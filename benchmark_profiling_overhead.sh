#!/bin/bash
# 对比不同 profiling 配置的性能开销

SCRIPT="${1:-inferencellama3-1-70B_overlap_metrics.py}"

echo "========================================"
echo "Profiling Overhead Benchmark"
echo "Script: $SCRIPT"
echo "========================================"
echo ""

# 1. 基准：无 profiling
echo "[1/4] Running baseline (no profiling)..."
time python "$SCRIPT" 2>&1 | grep -E "e2e_ms=|throughput" | tee baseline.txt
echo ""
sleep 2

# 2. Lite 模式
echo "[2/4] Running with lite profiling..."
time ./profile_gpu_lite.sh "$SCRIPT" 2>&1 | grep -E "e2e_ms=|throughput" | tee lite.txt
echo ""
sleep 2

# 3. Simple 模式
echo "[3/4] Running with simple profiling..."
time ./profile_gpu_simple.sh "$SCRIPT" 2>&1 | grep -E "e2e_ms=|throughput" | tee simple.txt
echo ""
sleep 2

# 4. Full 模式（标准配置）
echo "[4/4] Running with standard profiling..."
time ./profile_gpu.sh "$SCRIPT" 2>&1 | grep -E "e2e_ms=|throughput" | tee standard.txt
echo ""

echo "========================================"
echo "Summary:"
echo "========================================"
echo "Baseline (no profiling):"
cat baseline.txt
echo ""
echo "Lite mode:"
cat lite.txt
echo ""
echo "Simple mode:"
cat simple.txt
echo ""
echo "Standard mode:"
cat standard.txt
echo "========================================"

# 清理临时文件
rm -f baseline.txt lite.txt simple.txt standard.txt
