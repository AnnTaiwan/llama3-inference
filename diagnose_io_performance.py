#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 Raw vs FS 性能差异的工具

可能导致 Raw 比 FS 慢的原因：
1. ❌ 没有使用 O_DIRECT (实际上代码已经使用了)
2. ❌ Python 开销 - 每次 IO 都需要 numpy 转换
3. ❌ 单线程顺序 IO - 没有充分利用并发
4. ❌ 小 IO 大小 - DirectIO 对小 IO 不友好
5. ❌ 同步等待 - 没有使用异步/批量 IO
6. ✅ FS 被缓存在 page cache - warm 场景下 FS 很快
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from llama3.config import KVCacheArgs
from llama3.SSDBacked import RawBlockKVBackend


def test_raw_io_overhead():
    """测试 Raw IO 的 Python 开销"""
    print("\n" + "="*80)
    print("测试 1: Raw IO Python 开销分析")
    print("="*80)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return

    io_size = 4 * 1024 * 1024  # 4MB
    aligned_size = ((io_size + 4095) // 4096) * 4096

    backend = RawBlockKVBackend(
        dev_path=raw_dev,
        n_layers=1,
        blk_bytes=aligned_size,
        blk_per_layer=1,
        max_concurrent_io=1,
    )

    buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
    test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
    backend.write(0, 0, test_data)

    iterations = 100

    # 测试 1: 纯 IO 时间
    print(f"\n测试 {iterations} 次 4MB 读取...")

    times_total = []
    for _ in range(iterations):
        start = time.perf_counter()
        backend.read_into_pinned_aligned(0, 0, buffer)
        end = time.perf_counter()
        times_total.append((end - start) * 1000)

    avg_total = np.mean(times_total)
    std_total = np.std(times_total)
    p99_total = np.percentile(times_total, 99)

    print(f"\n读取延迟 (包含所有开销):")
    print(f"  平均: {avg_total:.3f} ms")
    print(f"  标准差: {std_total:.3f} ms")
    print(f"  P99: {p99_total:.3f} ms")
    print(f"  带宽: {io_size / (avg_total / 1000) / (1024**2):.2f} MB/s")

    # 理论带宽参考
    print(f"\n参考:")
    print(f"  NVMe PCIe 3.0 x4: ~3500 MB/s")
    print(f"  NVMe PCIe 4.0 x4: ~7000 MB/s")
    print(f"  SATA SSD: ~550 MB/s")

    del backend, buffer, test_data
    torch.cuda.empty_cache()


def test_fs_cache_effect():
    """测试 FS page cache 的影响"""
    print("\n" + "="*80)
    print("测试 2: 文件系统 Page Cache 影响")
    print("="*80)

    test_dir = "/data1/fs_bench"
    os.makedirs(test_dir, exist_ok=True)

    io_size = 4 * 1024 * 1024  # 4MB
    test_file = os.path.join(test_dir, "cache_test.bin")

    # 创建测试文件
    if not os.path.exists(test_file):
        with open(test_file, 'wb') as f:
            f.write(os.urandom(100 * 1024 * 1024))

    iterations = 100

    # 测试 1: 冷启动（drop cache）
    print("\n尝试 drop cache (需要 sudo)...")
    os.system("sync")
    ret = os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
    if ret != 0:
        print("[WARN] drop_caches 失败，可能没有 sudo 权限")
    time.sleep(0.5)

    print(f"\n冷启动测试 ({iterations} 次读取)...")
    times_cold = []
    for i in range(iterations):
        # 每次读取不同位置，避免预读优化
        offset = (i * io_size) % (50 * 1024 * 1024)
        start = time.perf_counter()
        with open(test_file, 'rb') as f:
            f.seek(offset)
            _ = f.read(io_size)
        end = time.perf_counter()
        times_cold.append((end - start) * 1000)

    # 测试 2: 热启动（重复读同一位置）
    print(f"\n热启动测试 ({iterations} 次读取，全部命中 cache)...")
    times_hot = []
    for _ in range(iterations):
        start = time.perf_counter()
        with open(test_file, 'rb') as f:
            f.seek(0)
            _ = f.read(io_size)
        end = time.perf_counter()
        times_hot.append((end - start) * 1000)

    print(f"\n冷启动结果:")
    print(f"  平均: {np.mean(times_cold):.3f} ms")
    print(f"  P99: {np.percentile(times_cold, 99):.3f} ms")
    print(f"  带宽: {io_size / (np.mean(times_cold) / 1000) / (1024**2):.2f} MB/s")

    print(f"\n热启动结果 (Page Cache):")
    print(f"  平均: {np.mean(times_hot):.3f} ms")
    print(f"  P99: {np.percentile(times_hot, 99):.3f} ms")
    print(f"  带宽: {io_size / (np.mean(times_hot) / 1000) / (1024**2):.2f} MB/s")

    print(f"\n加速比 (Hot vs Cold): {np.mean(times_cold) / np.mean(times_hot):.2f}x")
    print(f"\n⚠️  如果你的 benchmark 测试的是 warm 场景，FS 会因为 page cache 非常快！")


def test_io_size_scaling():
    """测试不同 IO 大小的性能"""
    print("\n" + "="*80)
    print("测试 3: 不同 IO 大小的性能 (Raw vs FS)")
    print("="*80)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return

    io_sizes = [
        4 * 1024,       # 4KB
        64 * 1024,      # 64KB
        256 * 1024,     # 256KB
        1024 * 1024,    # 1MB
        4 * 1024 * 1024,   # 4MB
    ]

    test_dir = "/data1/fs_bench"
    os.makedirs(test_dir, exist_ok=True)

    print(f"\n{'IO大小':<10} {'Raw延迟':<15} {'FS延迟':<15} {'Raw带宽':<15} {'FS带宽':<15}")
    print("-" * 70)

    for io_size in io_sizes:
        aligned_size = ((io_size + 4095) // 4096) * 4096

        # Raw 测试
        backend = RawBlockKVBackend(
            dev_path=raw_dev,
            n_layers=1,
            blk_bytes=aligned_size,
            blk_per_layer=1,
            max_concurrent_io=1,
        )
        buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
        test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
        backend.write(0, 0, test_data)

        # 预热
        for _ in range(10):
            backend.read_into_pinned_aligned(0, 0, buffer)

        # 测试
        times_raw = []
        for _ in range(100):
            start = time.perf_counter()
            backend.read_into_pinned_aligned(0, 0, buffer)
            end = time.perf_counter()
            times_raw.append((end - start) * 1000)

        raw_lat = np.mean(times_raw)
        raw_bw = aligned_size / (raw_lat / 1000) / (1024**2)

        del backend, buffer, test_data
        torch.cuda.empty_cache()

        # FS 测试
        test_file = os.path.join(test_dir, f"size_test_{io_size}.bin")
        if not os.path.exists(test_file):
            with open(test_file, 'wb') as f:
                f.write(os.urandom(max(io_size * 100, 10 * 1024 * 1024)))

        # 预热
        for _ in range(10):
            with open(test_file, 'rb') as f:
                _ = f.read(io_size)

        # 测试
        times_fs = []
        for _ in range(100):
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                _ = f.read(io_size)
            end = time.perf_counter()
            times_fs.append((end - start) * 1000)

        fs_lat = np.mean(times_fs)
        fs_bw = io_size / (fs_lat / 1000) / (1024**2)

        io_size_str = f"{io_size/1024:.0f}KB" if io_size < 1024*1024 else f"{io_size/(1024**2):.0f}MB"
        print(f"{io_size_str:<10} {raw_lat:<15.3f} {fs_lat:<15.3f} {raw_bw:<15.2f} {fs_bw:<15.2f}")

    print("\n⚠️  注意:")
    print("  - 小 IO (< 64KB): DirectIO 开销大，可能比 FS 慢")
    print("  - 大 IO (> 1MB): DirectIO 应该接近或超过 FS (如果 FS 未缓存)")


def test_sequential_vs_parallel():
    """测试顺序 vs 并发 IO"""
    print("\n" + "="*80)
    print("测试 4: 顺序 vs 并发 IO (Raw)")
    print("="*80)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return

    io_size = 4 * 1024 * 1024  # 4MB
    aligned_size = ((io_size + 4095) // 4096) * 4096
    total_ops = 100

    print(f"\n{'并发度':<10} {'总时间(s)':<15} {'IOPS':<15} {'带宽(MB/s)':<15}")
    print("-" * 55)

    for num_threads in [1, 2, 4, 8]:
        backend = RawBlockKVBackend(
            dev_path=raw_dev,
            n_layers=num_threads,
            blk_bytes=aligned_size,
            blk_per_layer=1,
            max_concurrent_io=num_threads,
        )

        buffers = [torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
                   for _ in range(num_threads)]
        test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
        for layer in range(num_threads):
            backend.write(layer, 0, test_data)

        # 预热
        for _ in range(10):
            backend.read_into_pinned_aligned(0, 0, buffers[0])

        # 测试
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        def worker(tid, count):
            for _ in range(count):
                backend.read_into_pinned_aligned(tid % num_threads, 0, buffers[tid])

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i, total_ops // num_threads)
                       for i in range(num_threads)]
            for f in as_completed(futures):
                f.result()
        end = time.perf_counter()

        total_time = end - start
        iops = total_ops / total_time
        bandwidth = (total_ops * aligned_size) / total_time / (1024**2)

        print(f"{num_threads:<10} {total_time:<15.3f} {iops:<15.2f} {bandwidth:<15.2f}")

        del backend, buffers, test_data
        torch.cuda.empty_cache()

    print("\n⚠️  如果并发度=1时很慢，说明没有充分利用 SSD 的并发能力")


def main():
    print("\n" + "="*80)
    print("Raw vs FS 性能诊断工具")
    print("="*80)
    print("\n这个工具会帮你找出为什么 Raw 比 FS 慢")

    tests = [
        ("1", "Raw IO Python 开销分析", test_raw_io_overhead),
        ("2", "FS Page Cache 影响测试", test_fs_cache_effect),
        ("3", "不同 IO 大小性能测试", test_io_size_scaling),
        ("4", "顺序 vs 并发 IO 测试", test_sequential_vs_parallel),
    ]

    print("\n可用的测试:")
    for tid, name, _ in tests:
        print(f"  {tid}. {name}")
    print("  all. 运行所有测试")

    choice = input("\n请选择要运行的测试 (输入编号或 'all'): ").strip()

    if choice == 'all':
        for _, _, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n[ERROR] 测试失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        for tid, _, test_func in tests:
            if choice == tid:
                try:
                    test_func()
                except Exception as e:
                    print(f"\n[ERROR] 测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print(f"[ERROR] 无效的选择: {choice}")

    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)


if __name__ == "__main__":
    main()
