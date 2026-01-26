#!/usr/bin/env python3
"""
对比测试：
1. 多线程读同一数据 (当前benchmark的做法)
2. 多线程读不同数据 (真正的并发)
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from llama3.config import KVCacheArgs
from llama3.SSDBacked import RawBlockKVBackend

def test_same_location():
    """测试：所有线程读同一位置（当前benchmark的做法）"""
    print("\n" + "="*80)
    print("测试 1: 所有线程读同一位置 (layer 0, slot 0)")
    print("="*80)

    dev = KVCacheArgs.ssd_device_path
    io_size = 4 * 1024 * 1024
    aligned_size = ((io_size + 4095) // 4096) * 4096
    num_threads = 8
    iterations = 25

    # 初始化 backend - 只有 1 layer
    backend = RawBlockKVBackend(
        dev_path=dev,
        n_layers=1,
        blk_bytes=aligned_size,
        blk_per_layer=1,
        max_concurrent_io=num_threads * 2,
    )

    # 创建buffers
    buffers = [torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
               for _ in range(num_threads)]

    # 写入数据
    test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
    backend.write(0, 0, test_data)

    # 预热
    for _ in range(5):
        backend.read_into_pinned_aligned(0, 0, buffers[0])

    latencies = []

    def worker(tid):
        thread_lats = []
        buf = buffers[tid]
        for _ in range(iterations):
            start = time.perf_counter()
            backend.read_into_pinned_aligned(0, 0, buf)  # ❌ 都读 layer 0, slot 0
            end = time.perf_counter()
            thread_lats.append((end - start) * 1_000_000)
        return thread_lats

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            latencies.extend(f.result())
    end_total = time.perf_counter()

    latencies = np.array(latencies)
    print(f"\n结果:")
    print(f"  总时间: {end_total - start_total:.3f} s")
    print(f"  样本数: {len(latencies)}")
    print(f"  平均延迟: {np.mean(latencies):.2f} μs")
    print(f"  P50: {np.percentile(latencies, 50):.2f} μs")
    print(f"  P99: {np.percentile(latencies, 99):.2f} μs")
    print(f"  单线程等效带宽: {aligned_size / (np.mean(latencies) / 1_000_000) / (1024**2):.2f} MB/s")

    del backend, buffers, test_data
    torch.cuda.empty_cache()

    return latencies


def test_different_locations():
    """测试：每个线程读不同位置（真正的并发）"""
    print("\n" + "="*80)
    print("测试 2: 每个线程读不同位置 (不同 layers)")
    print("="*80)

    dev = KVCacheArgs.ssd_device_path
    io_size = 4 * 1024 * 1024
    aligned_size = ((io_size + 4095) // 4096) * 4096
    num_threads = 8
    iterations = 25

    # 初始化 backend - 每个线程一个 layer
    backend = RawBlockKVBackend(
        dev_path=dev,
        n_layers=num_threads,  # ✅ 8 layers
        blk_bytes=aligned_size,
        blk_per_layer=1,
        max_concurrent_io=num_threads * 2,
    )

    # 创建buffers
    buffers = [torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
               for _ in range(num_threads)]

    # 写入数据到所有 layers
    print(f"写入 {num_threads} 个 layers...")
    test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
    for layer in range(num_threads):
        backend.write(layer, 0, test_data)

    # 预热
    for _ in range(5):
        backend.read_into_pinned_aligned(0, 0, buffers[0])

    latencies = []

    def worker(tid):
        thread_lats = []
        buf = buffers[tid]
        layer = tid  # ✅ 每个线程读自己的 layer
        for _ in range(iterations):
            start = time.perf_counter()
            backend.read_into_pinned_aligned(layer, 0, buf)  # ✅ 读不同 layer
            end = time.perf_counter()
            thread_lats.append((end - start) * 1_000_000)
        return thread_lats

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            latencies.extend(f.result())
    end_total = time.perf_counter()

    latencies = np.array(latencies)
    print(f"\n结果:")
    print(f"  总时间: {end_total - start_total:.3f} s")
    print(f"  样本数: {len(latencies)}")
    print(f"  平均延迟: {np.mean(latencies):.2f} μs")
    print(f"  P50: {np.percentile(latencies, 50):.2f} μs")
    print(f"  P99: {np.percentile(latencies, 99):.2f} μs")
    print(f"  单线程等效带宽: {aligned_size / (np.mean(latencies) / 1_000_000) / (1024**2):.2f} MB/s")
    print(f"  总吞吐: {len(latencies) * aligned_size / (end_total - start_total) / (1024**2):.2f} MB/s")

    del backend, buffers, test_data
    torch.cuda.empty_cache()

    return latencies


def main():
    print("="*80)
    print("并发读取测试：同一位置 vs 不同位置")
    print("="*80)

    # 测试1：读同一位置
    lats_same = test_same_location()

    # 测试2：读不同位置
    lats_diff = test_different_locations()

    # 对比
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    same_p50 = np.percentile(lats_same, 50)
    diff_p50 = np.percentile(lats_diff, 50)

    print(f"\n读同一位置 - P50: {same_p50:.2f} μs")
    print(f"读不同位置 - P50: {diff_p50:.2f} μs")

    if same_p50 > diff_p50:
        print(f"\n✅ 读不同位置快 {same_p50 / diff_p50:.2f}x")
        print("\n原因：")
        print("  - 读同一位置：所有线程竞争同一数据，串行化")
        print("  - 读不同位置：真正的并发，充分利用 NVMe 性能")
    else:
        print(f"\n⚠️ 读同一位置反而快 {diff_p50 / same_p50:.2f}x")
        print("\n可能原因：")
        print("  - SSD 控制器缓存了同一数据")
        print("  - 读不同位置涉及随机访问，性能下降")

    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("权重加载场景（单线程顺序）:")
    print("  - 不需要多线程并发")
    print("  - 单线程性能最重要")
    print("  - 多线程读同一数据反而变慢")
    print("\n当前 benchmark 的问题:")
    print("  - 测试了不适合的场景（多线程读同一数据）")
    print("  - 应该只测单线程性能")


if __name__ == "__main__":
    main()
