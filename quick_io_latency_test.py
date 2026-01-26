#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速 IO 延迟测试工具

用法示例:
    # 测试 FS，4MB IO，4 线程
    python quick_io_latency_test.py --type fs --size 4m --threads 4

    # 测试 Raw，16MB IO，8 线程
    python quick_io_latency_test.py --type raw --size 16m --threads 8

    # 同时测试 FS 和 Raw 并对比
    python quick_io_latency_test.py --type both --size 4m --threads 4
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch

sys.path.insert(0, str(Path(__file__).parent))
from llama3.config import KVCacheArgs
from llama3.SSDBacked import RawBlockKVBackend


def parse_size(size_str: str) -> int:
    """解析大小字符串，如 '4m', '256k'"""
    size_str = size_str.lower()
    if size_str.endswith('k'):
        return int(size_str[:-1]) * 1024
    elif size_str.endswith('m'):
        return int(size_str[:-1]) * 1024 * 1024
    elif size_str.endswith('g'):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def test_fs_latency(io_size: int, num_threads: int, iterations: int = 1000):
    """测试文件系统 IO 延迟"""
    print(f"\n{'='*80}")
    print(f"FS 测试: IO={io_size/(1024**2):.1f}MB, Threads={num_threads}, Iterations={iterations}")
    print(f"{'='*80}")

    test_dir = "/data1/fs_bench"
    os.makedirs(test_dir, exist_ok=True)

    test_file = os.path.join(test_dir, f"quick_test_{io_size}.bin")
    test_size = max(io_size * iterations, 100 * 1024 * 1024)

    # 创建测试文件
    if not os.path.exists(test_file):
        print(f"创建测试文件: {test_size / (1024**2):.2f} MB")
        with open(test_file, 'wb') as f:
            f.write(os.urandom(test_size))

    # 预热
    print("预热中...")
    for _ in range(10):
        with open(test_file, 'rb') as f:
            _ = f.read(io_size)

    # 测试
    latencies = []
    lock = threading.Lock()

    def worker(thread_id: int, iters: int):
        thread_latencies = []
        for i in range(iters):
            offset = (thread_id * io_size + i * io_size * num_threads) % (test_size - io_size)
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                f.seek(offset)
                _ = f.read(io_size)
            end = time.perf_counter()
            thread_latencies.append((end - start) * 1_000_000)
        with lock:
            latencies.extend(thread_latencies)

    print(f"测试中...")
    iters_per_thread = iterations // num_threads

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i, iters_per_thread) for i in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    end_total = time.perf_counter()

    return analyze_latencies(latencies, io_size, end_total - start_total, "FS")


def test_raw_latency(io_size: int, num_threads: int, iterations: int = 1000):
    """测试 Raw 块设备 IO 延迟"""
    print(f"\n{'='*80}")
    print(f"RAW 测试: IO={io_size/(1024**2):.1f}MB, Threads={num_threads}, Iterations={iterations}")
    print(f"{'='*80}")

    raw_dev = getattr(KVCacheArgs, "ssd_device_path", None)
    if raw_dev is None:
        print("[ERROR] KVCacheArgs.ssd_device_path 未配置")
        return None

    print(f"Raw 设备: {raw_dev}")

    # 对齐到 4KB
    aligned_size = ((io_size + 4095) // 4096) * 4096
    if aligned_size != io_size:
        print(f"IO 大小对齐: {io_size} -> {aligned_size} bytes")

    # 初始化 backend
    n_layers = num_threads
    backend = RawBlockKVBackend(
        dev_path=raw_dev,
        n_layers=n_layers,
        blk_bytes=aligned_size,
        blk_per_layer=1,
        max_concurrent_io=num_threads,
    )

    # 创建 buffers
    buffers = [torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
               for _ in range(num_threads)]

    # 预写入数据
    print("预写入测试数据...")
    test_data = torch.randn(aligned_size // 2, dtype=torch.bfloat16, pin_memory=True)
    for layer in range(n_layers):
        backend.write(layer, 0, test_data)

    # 预热
    print("预热中...")
    for _ in range(10):
        backend.read_into_pinned_aligned(0, 0, buffers[0])

    # 测试
    latencies = []
    lock = threading.Lock()

    def worker(thread_id: int, iters: int):
        thread_latencies = []
        layer = thread_id % n_layers
        buffer = buffers[thread_id]
        for _ in range(iters):
            start = time.perf_counter()
            backend.read_into_pinned_aligned(layer, 0, buffer)
            end = time.perf_counter()
            thread_latencies.append((end - start) * 1_000_000)
        with lock:
            latencies.extend(thread_latencies)

    print(f"测试中...")
    iters_per_thread = iterations // num_threads

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i, iters_per_thread) for i in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    end_total = time.perf_counter()

    # 清理
    del backend, buffers, test_data
    torch.cuda.empty_cache()

    return analyze_latencies(latencies, aligned_size, end_total - start_total, "RAW")


def analyze_latencies(latencies, io_size, total_time, io_type):
    """分析延迟数据"""
    arr = np.array(latencies)

    stats = {
        'io_type': io_type,
        'num_samples': len(latencies),
        'min': np.min(arr),
        'p50': np.percentile(arr, 50),
        'p90': np.percentile(arr, 90),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
        'p99.9': np.percentile(arr, 99.9),
        'max': np.max(arr),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'total_time': total_time,
        'iops': len(latencies) / total_time,
        'bandwidth_mbps': (len(latencies) * io_size) / total_time / (1024**2),
    }

    print(f"\n结果统计:")
    print(f"  样本数: {stats['num_samples']}")
    print(f"\n延迟 (microseconds):")
    print(f"  最小值:  {stats['min']:>10.2f} μs")
    print(f"  P50:     {stats['p50']:>10.2f} μs  (中位数)")
    print(f"  P90:     {stats['p90']:>10.2f} μs")
    print(f"  P95:     {stats['p95']:>10.2f} μs")
    print(f"  P99:     {stats['p99']:>10.2f} μs")
    print(f"  P99.9:   {stats['p99.9']:>10.2f} μs  (尾延迟)")
    print(f"  最大值:  {stats['max']:>10.2f} μs")
    print(f"  平均值:  {stats['mean']:>10.2f} μs")
    print(f"  标准差:  {stats['std']:>10.2f} μs")
    print(f"\n性能:")
    print(f"  总时间:  {total_time:.3f} s")
    print(f"  IOPS:    {stats['iops']:.2f}")
    print(f"  带宽:    {stats['bandwidth_mbps']:.2f} MB/s")

    # 延迟分布直方图
    print(f"\n延迟分布 (histogram):")
    hist, bins = np.histogram(arr, bins=20)
    max_bar_width = 50
    max_count = np.max(hist)
    for i in range(len(hist)):
        bar_width = int((hist[i] / max_count) * max_bar_width)
        print(f"  {bins[i]:>8.1f} - {bins[i+1]:>8.1f} μs: {'█' * bar_width} {hist[i]}")

    return stats


def compare_results(fs_stats, raw_stats):
    """对比 FS 和 Raw 结果"""
    print(f"\n{'='*80}")
    print("FS vs RAW 对比")
    print(f"{'='*80}")

    print(f"\n{'指标':<15} {'FS':<15} {'RAW':<15} {'加速比':<15}")
    print(f"{'-'*60}")

    metrics = ['p50', 'p90', 'p95', 'p99', 'p99.9', 'mean', 'iops', 'bandwidth_mbps']
    labels = {
        'p50': 'P50 (μs)',
        'p90': 'P90 (μs)',
        'p95': 'P95 (μs)',
        'p99': 'P99 (μs)',
        'p99.9': 'P99.9 (μs)',
        'mean': '平均 (μs)',
        'iops': 'IOPS',
        'bandwidth_mbps': '带宽 (MB/s)',
    }

    for metric in metrics:
        fs_val = fs_stats[metric]
        raw_val = raw_stats[metric]

        # 对于延迟，FS/Raw；对于吞吐，Raw/FS
        if metric in ['iops', 'bandwidth_mbps']:
            speedup = raw_val / fs_val
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = fs_val / raw_val
            speedup_str = f"{speedup:.2f}x"

        print(f"{labels[metric]:<15} {fs_val:<15.2f} {raw_val:<15.2f} {speedup_str:<15}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='快速 IO 延迟测试工具')
    parser.add_argument('--type', choices=['fs', 'raw', 'both'], default='both',
                        help='测试类型: fs, raw, 或 both')
    parser.add_argument('--size', type=str, default='4m',
                        help='IO 大小 (例如: 4k, 256k, 4m, 16m)')
    parser.add_argument('--threads', type=int, default=4,
                        help='并发线程数')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='测试迭代次数')

    args = parser.parse_args()

    io_size = parse_size(args.size)
    print(f"\n{'='*80}")
    print(f"快速 IO 延迟测试")
    print(f"{'='*80}")
    print(f"IO 大小: {io_size / (1024**2):.1f} MB ({io_size} bytes)" if io_size >= 1024*1024
          else f"IO 大小: {io_size / 1024:.1f} KB ({io_size} bytes)")
    print(f"并发线程: {args.threads}")
    print(f"迭代次数: {args.iterations}")
    print(f"测试类型: {args.type.upper()}")

    fs_stats = None
    raw_stats = None

    if args.type in ['fs', 'both']:
        fs_stats = test_fs_latency(io_size, args.threads, args.iterations)

    if args.type in ['raw', 'both']:
        raw_stats = test_raw_latency(io_size, args.threads, args.iterations)

    if fs_stats and raw_stats:
        compare_results(fs_stats, raw_stats)


if __name__ == "__main__":
    main()
