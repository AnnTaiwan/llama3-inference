#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IO Latency / Throughput Microbenchmark for FS vs RAW (RawBlockKVBackend)

This script is an improved version of benchmark_io_latency_cdf.py with the goal of
measuring the "best possible" gap between:
  (1) File-system based reads (FS path)
  (2) Filesystem-bypass raw-block reads (RAW path via RawBlockKVBackend)

Key fixes / improvements vs the original script:
  - Correct throughput calculation for concurrency (use wall time, not sum(latency))
  - Configurable I/O sizes and thread counts via CLI
  - Configurable access pattern (random / sequential / stride)
  - Better cold-cache control for FS:
        * posix_fadvise(POSIX_FADV_RANDOM) to disable readahead
        * posix_fadvise(POSIX_FADV_DONTNEED) after each read to evict cache pages
        * optional global drop_caches (needs root / sudo)
  - RAW benchmark supports a configurable workset (many different blocks),
    so you can avoid "same-block SSD cache hit" artifacts.
  - Optional fs preadv-into preallocated buffers to reduce Python allocation noise.

⚠️  WARNING about RAW:
  RAW path will access the block device specified by KVCacheArgs.ssd_device_path.
  DO NOT point it at a device with valuable data. Prefer a dedicated partition,
  or a loop device backed by a file.

Example (maximize gap for random small I/O + concurrency):
  python benchmark_io_latency_cdf_improved.py \
    --profile small-rand \
    --threads 1,2,4,8,16 \
    --cold fadvise \
    --pattern random \
    --workset-mb 1024

Example (mimic weight-loading large reads, single thread):
  python benchmark_io_latency_cdf_improved.py \
    --profile weight \
    --threads 1 \
    --cold dropcaches \
    --pattern sequential
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Project imports (keep same as original) ---
sys.path.insert(0, str(Path(__file__).parent))
from llama3.config import KVCacheArgs  # type: ignore
from llama3.SSDBacked import RawBlockKVBackend  # type: ignore

try:
    import torch
except Exception as e:
    raise RuntimeError("This script requires PyTorch to test the RAW path.") from e


# ======================== Utilities ========================

def parse_size(s: str) -> int:
    """
    Parse sizes like: 4k, 64K, 256kb, 1m, 4MB, 16mb, 1g.
    Uses binary units (KiB=1024).
    """
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)([kmgt]?)(b)?", s)
    if not m:
        raise ValueError(f"Invalid size: {s}")
    val = int(m.group(1))
    unit = m.group(2)
    mul = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[unit]
    return val * mul


def fmt_size(n: int) -> str:
    if n % (1024**3) == 0 and n >= 1024**3:
        return f"{n // (1024**3)}GB"
    if n % (1024**2) == 0 and n >= 1024**2:
        return f"{n // (1024**2)}MB"
    if n % 1024 == 0 and n >= 1024:
        return f"{n // 1024}KB"
    return f"{n}B"


def align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def now_ns() -> int:
    return time.perf_counter_ns()


def try_posix_fadvise(fd: int, offset: int, length: int, advice: int) -> bool:
    """
    Best-effort wrapper: returns True if supported and succeeded.
    """
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        os.posix_fadvise(fd, offset, length, advice)  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def drop_caches_global() -> bool:
    """
    Drop Linux page cache / dentries / inodes (echo 3 > /proc/sys/vm/drop_caches).
    Requires root. We try:
      1) direct write (if running as root)
      2) sudo -n (no password prompt) fallback

    Returns True on success.
    """
    try:
        os.sync()
    except Exception:
        pass

    # direct
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except PermissionError:
        pass
    except Exception:
        pass

    # sudo -n fallback
    cmd = "sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1"
    ret = os.system(cmd)
    return ret == 0


def get_blockdev_size_bytes(path: str) -> Optional[int]:
    """
    Returns block device size in bytes if possible (Linux).
    """
    import fcntl
    import struct
    BLKGETSIZE64 = 0x80081272  # _IOR(0x12,114,size_t) on many arch

    try:
        fd = os.open(path, os.O_RDONLY)
    except Exception:
        return None
    try:
        buf = b"\x00" * 8
        out = fcntl.ioctl(fd, BLKGETSIZE64, buf)
        size = struct.unpack("Q", out)[0]
        return int(size)
    except Exception:
        return None
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


# ======================== Data structures ========================

@dataclass
class LatencyStats:
    io_type: str  # "fs" or "raw"
    io_size_bytes: int
    num_threads: int
    num_samples: int

    # latency stats (microseconds)
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    p50_us: float
    p90_us: float
    p95_us: float
    p99_us: float
    p99_9_us: float

    # throughput stats (system-level, wall-time based)
    wall_time_s: float
    iops: float
    bandwidth_mbps: float

    # raw latency samples for CDF plots
    raw_latencies_us: List[float]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("raw_latencies_us", None)  # avoid huge json
        return d


def calculate_percentiles(lat_us: List[float]) -> Dict[str, float]:
    arr = np.asarray(lat_us, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99.9": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def save_cdf_data(stats: LatencyStats, out_path: str) -> None:
    lat = np.asarray(stats.raw_latencies_us, dtype=np.float64)
    lat_sorted = np.sort(lat)
    cdf = np.arange(1, len(lat_sorted) + 1) / len(lat_sorted)
    data = {
        "io_type": stats.io_type,
        "io_size_bytes": stats.io_size_bytes,
        "num_threads": stats.num_threads,
        "latency_us": lat_sorted.tolist(),
        "cdf": cdf.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


# ======================== FS benchmark ========================

def ensure_test_file(path: str, size_bytes: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) >= size_bytes:
        return

    print(f"[FS] Creating test file: {path} ({size_bytes/(1024**2):.1f} MiB)")
    # Write random data so filesystem doesn't optimize sparse/compression
    chunk = 8 * 1024 * 1024
    remaining = size_bytes
    with open(path, "wb", buffering=0) as f:
        while remaining > 0:
            n = min(chunk, remaining)
            f.write(os.urandom(n))
            remaining -= n
        f.flush()
        os.fsync(f.fileno())


def gen_offsets(pattern: str, thread_id: int, n_ops: int, io_size: int, span: int, seed: int) -> List[int]:
    """
    span: max valid offset exclusive (file_size - io_size)
    """
    if span <= 0:
        return [0] * n_ops

    if pattern == "sequential":
        # each thread walks its own contiguous region
        base = (thread_id * n_ops * io_size) % span
        return [((base + i * io_size) % span) for i in range(n_ops)]
    if pattern == "stride":
        # global stride across threads (similar to original)
        return [(((thread_id + i) * io_size) % span) for i in range(n_ops)]
    if pattern == "random":
        rng = random.Random(seed ^ (thread_id * 0x9E3779B97F4A7C15))
        # random aligned offsets
        max_blocks = span // io_size
        return [rng.randrange(0, max_blocks) * io_size for _ in range(n_ops)]

    raise ValueError(f"Unknown pattern: {pattern}")


def benchmark_fs_latency(
    io_size: int,
    num_threads: int,
    iterations: int,
    warmup: int,
    test_dir: str,
    cold_mode: str,
    drop_every: int,
    pattern: str,
    use_preadv: bool,
    fadvise_random: bool,
    fadvise_dontneed: bool,
) -> LatencyStats:
    """
    FS benchmark:
      - default uses buffered reads (os.pread / os.preadv)
      - cold_mode controls cache behavior
    """
    test_file = os.path.join(test_dir, f"latency_test_{io_size}.bin")

    # Make file big enough to reduce cache hits. For random small reads, we still
    # rely on fadvise/drop_caches for cold behavior.
    min_size = max(io_size * iterations * num_threads * 2, 512 * 1024 * 1024)
    ensure_test_file(test_file, min_size)
    test_size = os.path.getsize(test_file)

    fd = os.open(test_file, os.O_RDONLY)
    try:
        # Optional: disable readahead when random (per POSIX_FADV_RANDOM).
        if fadvise_random and hasattr(os, "POSIX_FADV_RANDOM"):
            try_posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # type: ignore[attr-defined]

        # Warmup (not measured)
        for _ in range(max(0, warmup)):
            _ = os.pread(fd, min(io_size, 4096), 0)

        # If cold_mode uses global drop_caches, do one drop right before measured phase.
        if cold_mode == "dropcaches":
            ok = drop_caches_global()
            if not ok:
                print("[WARN] drop_caches failed (need root or passwordless sudo). Falling back to fadvise-only.")
                cold_mode = "fadvise"

        span = test_size - io_size
        iters_per_thread = math.ceil(iterations / num_threads)

        # Pre-allocate buffers for preadv mode (reduces Python alloc noise)
        buffers = [bytearray(io_size) for _ in range(num_threads)] if use_preadv else [None] * num_threads

        latencies: List[float] = []
        lat_lock = threading.Lock()

        ready_barrier = threading.Barrier(num_threads + 1)
        go = threading.Event()

        def worker(tid: int):
            offs = gen_offsets(pattern, tid, iters_per_thread, io_size, span, seed=0xC0FFEE)
            buf = buffers[tid] if use_preadv else None

            local: List[float] = []
            ready_barrier.wait()
            go.wait()

            # iterate
            for i, off in enumerate(offs):
                # keep cold if requested
                if cold_mode == "dropcaches" and drop_every > 0 and (i % drop_every == 0):
                    drop_caches_global()

                t0 = now_ns()
                if use_preadv:
                    n = os.preadv(fd, [buf], off)  # type: ignore[arg-type]
                    if n != io_size:
                        # ignore short reads (shouldn't happen)
                        pass
                else:
                    _ = os.pread(fd, io_size, off)
                t1 = now_ns()
                local.append((t1 - t0) / 1_000.0)

                # per-read eviction of cache pages (best-effort)
                if cold_mode == "fadvise" and fadvise_dontneed and hasattr(os, "POSIX_FADV_DONTNEED"):
                    try_posix_fadvise(fd, off, io_size, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]

            with lat_lock:
                latencies.extend(local)

        threads = [threading.Thread(target=worker, args=(tid,), daemon=True) for tid in range(num_threads)]
        for t in threads:
            t.start()

        ready_barrier.wait()  # all workers ready
        t_wall0 = now_ns()
        go.set()
        for t in threads:
            t.join()
        t_wall1 = now_ns()

        wall_s = (t_wall1 - t_wall0) / 1e9
        total_ops = len(latencies)
        total_bytes = total_ops * io_size

        pct = calculate_percentiles(latencies)
        return LatencyStats(
            io_type="fs",
            io_size_bytes=io_size,
            num_threads=num_threads,
            num_samples=total_ops,
            mean_us=pct["mean"],
            std_us=pct["std"],
            min_us=pct["min"],
            max_us=pct["max"],
            p50_us=pct["p50"],
            p90_us=pct["p90"],
            p95_us=pct["p95"],
            p99_us=pct["p99"],
            p99_9_us=pct["p99.9"],
            wall_time_s=wall_s,
            iops=total_ops / wall_s if wall_s > 0 else float("nan"),
            bandwidth_mbps=(total_bytes / wall_s / (1024**2)) if wall_s > 0 else float("nan"),
            raw_latencies_us=latencies,
        )
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


# ======================== RAW benchmark ========================

def benchmark_raw_latency(
    io_size: int,
    num_threads: int,
    iterations: int,
    warmup: int,
    device_path: str,
    workset_mb: int,
    pattern: str,
    max_concurrent_io: Optional[int],
) -> LatencyStats:
    """
    RAW benchmark via RawBlockKVBackend.

    We create a "workset" of many blocks (blk_per_layer = workset_blocks).
    Then each thread reads different blocks according to `pattern`.
    """
    if not os.path.exists(device_path):
        raise RuntimeError(f"RAW device not found: {device_path}")

    aligned_size = align_up(io_size, 4096)

    # compute workset blocks
    workset_bytes = int(workset_mb) * 1024 * 1024
    workset_blocks = max(num_threads * 16, workset_bytes // aligned_size)
    workset_blocks = max(workset_blocks, 1)

    dev_bytes = get_blockdev_size_bytes(device_path)
    if dev_bytes is not None:
        max_blocks = dev_bytes // aligned_size
        if max_blocks <= 0:
            raise RuntimeError(f"Device too small? dev_bytes={dev_bytes}, aligned_size={aligned_size}")
        if workset_blocks > max_blocks:
            print(f"[RAW] workset too large for device, capping: {workset_blocks} -> {max_blocks}")
            workset_blocks = int(max_blocks)

    # backend: 1 layer, many blocks
    if max_concurrent_io is None:
        max_concurrent_io = num_threads * 2

    backend = RawBlockKVBackend(
        dev_path=device_path,
        n_layers=1,
        blk_bytes=aligned_size,
        blk_per_layer=workset_blocks,
        max_concurrent_io=max_concurrent_io,
    )

    # pinned buffers: one per thread
    buffers = [
        torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
        for _ in range(num_threads)
    ]

    # Warmup reads (not measured)
    for i in range(max(0, warmup)):
        blk = (i % workset_blocks)
        backend.read_into_pinned_aligned(0, blk, buffers[0])

    iters_per_thread = math.ceil(iterations / num_threads)

    latencies: List[float] = []
    lat_lock = threading.Lock()

    ready_barrier = threading.Barrier(num_threads + 1)
    go = threading.Event()

    def gen_blocks(tid: int) -> List[int]:
        if workset_blocks <= 1:
            return [0] * iters_per_thread

        if pattern == "sequential":
            base = (tid * iters_per_thread) % workset_blocks
            return [((base + i) % workset_blocks) for i in range(iters_per_thread)]
        if pattern == "stride":
            return [((tid + i * num_threads) % workset_blocks) for i in range(iters_per_thread)]
        if pattern == "random":
            rng = random.Random(0xBADC0DE ^ (tid * 0x9E3779B97F4A7C15))
            return [rng.randrange(0, workset_blocks) for _ in range(iters_per_thread)]
        if pattern == "shared":
            return [0] * iters_per_thread

        raise ValueError(f"Unknown pattern: {pattern}")

    def worker(tid: int):
        blk_list = gen_blocks(tid)
        buf = buffers[tid]
        local: List[float] = []

        ready_barrier.wait()
        go.wait()

        for blk in blk_list:
            t0 = now_ns()
            backend.read_into_pinned_aligned(0, int(blk), buf)
            t1 = now_ns()
            local.append((t1 - t0) / 1_000.0)

        with lat_lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker, args=(tid,), daemon=True) for tid in range(num_threads)]
    for t in threads:
        t.start()

    ready_barrier.wait()
    t_wall0 = now_ns()
    go.set()
    for t in threads:
        t.join()
    t_wall1 = now_ns()

    wall_s = (t_wall1 - t_wall0) / 1e9
    total_ops = len(latencies)
    total_bytes = total_ops * aligned_size

    pct = calculate_percentiles(latencies)
    stats = LatencyStats(
        io_type="raw",
        io_size_bytes=aligned_size,
        num_threads=num_threads,
        num_samples=total_ops,
        mean_us=pct["mean"],
        std_us=pct["std"],
        min_us=pct["min"],
        max_us=pct["max"],
        p50_us=pct["p50"],
        p90_us=pct["p90"],
        p95_us=pct["p95"],
        p99_us=pct["p99"],
        p99_9_us=pct["p99.9"],
        wall_time_s=wall_s,
        iops=total_ops / wall_s if wall_s > 0 else float("nan"),
        bandwidth_mbps=(total_bytes / wall_s / (1024**2)) if wall_s > 0 else float("nan"),
        raw_latencies_us=latencies,
    )

    # cleanup
    del backend, buffers
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return stats


# ======================== Reporting ========================

def print_stats(stats: LatencyStats) -> None:
    print(f"\n[{stats.io_type.upper()}] io={fmt_size(stats.io_size_bytes)}, threads={stats.num_threads}, samples={stats.num_samples}")
    print(f"  Latency(us): p50={stats.p50_us:.2f}, p90={stats.p90_us:.2f}, p95={stats.p95_us:.2f}, p99={stats.p99_us:.2f}, p99.9={stats.p99_9_us:.2f}, mean={stats.mean_us:.2f}")
    print(f"  Throughput:  IOPS={stats.iops:.2f}, BW={stats.bandwidth_mbps:.2f} MiB/s, wall={stats.wall_time_s:.3f}s")


def print_speedup(fs: LatencyStats, raw: LatencyStats) -> None:
    def ratio(a: float, b: float) -> float:
        return a / b if (b and b > 0) else float("nan")

    print("\n--- RAW vs FS speedup (FS/RAW) ---")
    print(f"Latency p50:  {ratio(fs.p50_us, raw.p50_us):.2f}x")
    print(f"Latency p90:  {ratio(fs.p90_us, raw.p90_us):.2f}x")
    print(f"Latency p99:  {ratio(fs.p99_us, raw.p99_us):.2f}x")
    print(f"Bandwidth:    {ratio(raw.bandwidth_mbps, fs.bandwidth_mbps):.2f}x (RAW/FS)")
    print(f"IOPS:         {ratio(raw.iops, fs.iops):.2f}x (RAW/FS)")


# ======================== Main ========================

def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_size_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(parse_size(part))
    return out


def build_profile(profile: str) -> List[int]:
    if profile == "weight":
        return [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    if profile == "small-rand":
        return [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024]
    if profile == "mixed":
        return [4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    raise ValueError(f"Unknown profile: {profile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="weight", choices=["weight", "small-rand", "mixed"])
    ap.add_argument("--sizes", default="", help="Override sizes, e.g. 4k,16k,64k,256k,1m")
    ap.add_argument("--threads", default="1", help="comma list, e.g. 1,2,4,8,16")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--pattern", default="random", choices=["random", "sequential", "stride", "shared"])
    ap.add_argument("--cold", default="fadvise", choices=["none", "fadvise", "dropcaches"])
    ap.add_argument("--drop-every", type=int, default=1, help="for cold=dropcaches: drop cache every N ops per-thread")
    ap.add_argument("--fs-dir", default="/data1/fs_bench")
    ap.add_argument("--out-dir", default=str(Path.home() / "logs/io_latency_cdf_improved"))
    ap.add_argument("--fs-preadv", action="store_true", help="Use os.preadv into preallocated buffers to reduce Python alloc noise")
    ap.add_argument("--no-fadvise-random", action="store_true", help="Do not call POSIX_FADV_RANDOM for FS")
    ap.add_argument("--no-fadvise-dontneed", action="store_true", help="Do not call POSIX_FADV_DONTNEED per read for FS cold=fadvise")
    ap.add_argument("--workset-mb", type=int, default=1024, help="RAW workset size (MiB) to avoid same-block cache hits")
    ap.add_argument("--raw-max-concurrent-io", type=int, default=0, help="Override RawBlockKVBackend max_concurrent_io (0=auto)")

    args = ap.parse_args()

    sizes = build_profile(args.profile)
    if args.sizes.strip():
        sizes = parse_size_list(args.sizes)
    threads = parse_int_list(args.threads)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        raise RuntimeError("KVCacheArgs.ssd_device_path is not configured")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Output dir: {args.out_dir}")
    print(f"[INFO] Raw device: {raw_dev}")
    print(f"[INFO] Sizes: {[fmt_size(s) for s in sizes]}")
    print(f"[INFO] Threads: {threads}")
    print(f"[INFO] cold={args.cold}, pattern={args.pattern}, iterations={args.iterations}, warmup={args.warmup}")

    all_stats: List[LatencyStats] = []

    for io_size in sizes:
        for n_th in threads:
            print(f"\n{'='*90}")
            print(f"Config: io={fmt_size(io_size)}  threads={n_th}")
            print(f"{'='*90}")

            fs = benchmark_fs_latency(
                io_size=io_size,
                num_threads=n_th,
                iterations=args.iterations,
                warmup=args.warmup,
                test_dir=args.fs_dir,
                cold_mode=args.cold,
                drop_every=args.drop_every,
                pattern=args.pattern if args.pattern != "shared" else "random",
                use_preadv=args.fs_preadv,
                fadvise_random=not args.no_fadvise_random,
                fadvise_dontneed=not args.no_fadvise_dontneed,
            )
            print_stats(fs)
            all_stats.append(fs)
            save_cdf_data(fs, os.path.join(args.out_dir, f"cdf_fs_{fmt_size(io_size)}_t{n_th}.json"))

            raw = benchmark_raw_latency(
                io_size=io_size,
                num_threads=n_th,
                iterations=args.iterations,
                warmup=args.warmup,
                device_path=raw_dev,
                workset_mb=args.workset_mb,
                pattern=args.pattern,
                max_concurrent_io=(args.raw_max_concurrent_io if args.raw_max_concurrent_io > 0 else None),
            )
            print_stats(raw)
            all_stats.append(raw)
            save_cdf_data(raw, os.path.join(args.out_dir, f"cdf_raw_{fmt_size(io_size)}_t{n_th}.json"))

            print_speedup(fs, raw)

    with open(os.path.join(args.out_dir, "latency_summary.json"), "w") as f:
        json.dump([s.to_dict() for s in all_stats], f, indent=2)
    print(f"\n[INFO] Saved summary: {os.path.join(args.out_dir, 'latency_summary.json')}")


if __name__ == "__main__":
    main()
