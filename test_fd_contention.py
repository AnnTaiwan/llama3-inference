#!/usr/bin/env python3
"""
测试文件描述符竞争问题

对比：
1. 共享单个 fd (当前实现)
2. 每线程独立 fd (修复方案)
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llama3.config import KVCacheArgs

# 测试配置
IO_SIZE = 4 * 1024 * 1024  # 4MB
ALIGNED_SIZE = ((IO_SIZE + 4095) // 4096) * 4096
NUM_THREADS = 8
ITERATIONS_PER_THREAD = 25

def test_shared_fd():
    """测试共享单个 fd（当前实现）"""
    print("\n" + "="*80)
    print("测试 1: 共享单个文件描述符")
    print("="*80)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if not raw_dev:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return

    # 打开单个 fd
    fd = os.open(raw_dev, os.O_RDONLY | os.O_DIRECT)
    print(f"[INFO] 打开共享 fd: {fd}")

    # 创建对齐的 buffer
    buffers = []
    for _ in range(NUM_THREADS):
        buf = np.empty(ALIGNED_SIZE, dtype=np.uint8)
        offset = (-buf.ctypes.data) % 4096
        if offset:
            buf = np.empty(ALIGNED_SIZE + 4096, dtype=np.uint8)
            offset = (-buf.ctypes.data) % 4096
            buf = buf[offset:offset+ALIGNED_SIZE]
        buffers.append(buf)

    latencies = []

    def worker(thread_id):
        thread_latencies = []
        buf = buffers[thread_id]
        offset = thread_id * ALIGNED_SIZE

        for _ in range(ITERATIONS_PER_THREAD):
            start = time.perf_counter()
            data = os.pread(fd, ALIGNED_SIZE, offset)  # ❌ 共享 fd
            buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
            end = time.perf_counter()
            thread_latencies.append((end - start) * 1_000_000)

        return thread_latencies

    print(f"[INFO] 测试中: {NUM_THREADS} threads, {ITERATIONS_PER_THREAD} iterations/thread")

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_THREADS)]
        for f in as_completed(futures):
            latencies.extend(f.result())
    end_total = time.perf_counter()

    os.close(fd)

    # 统计
    latencies = np.array(latencies)
    print(f"\n结果:")
    print(f"  总时间: {end_total - start_total:.3f} s")
    print(f"  平均延迟: {np.mean(latencies):.2f} μs")
    print(f"  P50: {np.percentile(latencies, 50):.2f} μs")
    print(f"  P99: {np.percentile(latencies, 99):.2f} μs")
    print(f"  总吞吐: {len(latencies) * ALIGNED_SIZE / (end_total - start_total) / (1024**2):.2f} MB/s")

    return latencies


def test_per_thread_fd():
    """测试每线程独立 fd（修复方案）"""
    print("\n" + "="*80)
    print("测试 2: 每线程独立文件描述符")
    print("="*80)

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if not raw_dev:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return

    # 创建对齐的 buffer
    buffers = []
    for _ in range(NUM_THREADS):
        buf = np.empty(ALIGNED_SIZE, dtype=np.uint8)
        offset = (-buf.ctypes.data) % 4096
        if offset:
            buf = np.empty(ALIGNED_SIZE + 4096, dtype=np.uint8)
            offset = (-buf.ctypes.data) % 4096
            buf = buf[offset:offset+ALIGNED_SIZE]
        buffers.append(buf)

    latencies = []

    def worker(thread_id):
        # ✅ 每个线程打开自己的 fd
        fd = os.open(raw_dev, os.O_RDONLY | os.O_DIRECT)

        thread_latencies = []
        buf = buffers[thread_id]
        offset = thread_id * ALIGNED_SIZE

        for _ in range(ITERATIONS_PER_THREAD):
            start = time.perf_counter()
            data = os.pread(fd, ALIGNED_SIZE, offset)  # ✅ 独立 fd
            buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
            end = time.perf_counter()
            thread_latencies.append((end - start) * 1_000_000)

        os.close(fd)
        return thread_latencies

    print(f"[INFO] 测试中: {NUM_THREADS} threads (每个独立 fd), {ITERATIONS_PER_THREAD} iterations/thread")

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_THREADS)]
        for f in as_completed(futures):
            latencies.extend(f.result())
    end_total = time.perf_counter()

    # 统计
    latencies = np.array(latencies)
    print(f"\n结果:")
    print(f"  总时间: {end_total - start_total:.3f} s")
    print(f"  平均延迟: {np.mean(latencies):.2f} μs")
    print(f"  P50: {np.percentile(latencies, 50):.2f} μs")
    print(f"  P99: {np.percentile(latencies, 99):.2f} μs")
    print(f"  总吞吐: {len(latencies) * ALIGNED_SIZE / (end_total - start_total) / (1024**2):.2f} MB/s")

    return latencies


def main():
    print("="*80)
    print("文件描述符竞争测试")
    print("="*80)
    print(f"\n配置:")
    print(f"  IO 大小: {ALIGNED_SIZE / (1024**2):.1f} MB")
    print(f"  并发线程: {NUM_THREADS}")
    print(f"  每线程迭代: {ITERATIONS_PER_THREAD}")

    # 测试 1: 共享 fd
    latencies_shared = test_shared_fd()

    # 测试 2: 独立 fd
    latencies_per_thread = test_per_thread_fd()

    # 对比
    if latencies_shared is not None and latencies_per_thread is not None:
        print("\n" + "="*80)
        print("对比结果")
        print("="*80)

        shared_p50 = np.percentile(latencies_shared, 50)
        per_thread_p50 = np.percentile(latencies_per_thread, 50)

        print(f"\n共享 fd - P50: {shared_p50:.2f} μs")
        print(f"独立 fd - P50: {per_thread_p50:.2f} μs")

        if shared_p50 > per_thread_p50:
            speedup = shared_p50 / per_thread_p50
            print(f"\n✅ 独立 fd 快 {speedup:.2f}x - 这就是为什么高并发时 Raw 变慢！")
        else:
            print(f"\n⚠️ 没有改善，可能有其他瓶颈")

        print("\n" + "="*80)
        print("结论:")
        print("="*80)
        print("如果独立 fd 明显更快，说明问题是:")
        print("  1. RawBlockKVBackend 的单个 fd 导致内核锁竞争")
        print("  2. 需要修改 RawBlockKVBackend 支持每线程独立 fd")
        print("  3. 或者在 benchmark 中绕过 RawBlockKVBackend，直接用 os.pread")


if __name__ == "__main__":
    main()
