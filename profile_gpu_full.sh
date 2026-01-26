#!/bin/bash
# GPU Profiling Script with Full GPU Metrics (requires sudo)
# This version includes GPU hardware metrics but needs elevated privileges

# 使用方法:
#   sudo ./profile_gpu_full.sh                          # 默认使用 inferencellama3-1-70B.py
#   sudo ./profile_gpu_full.sh inferencellama3-1-70B_overlap_metrics.py

# 检查是否有 root 权限
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires sudo/root privileges for GPU metrics."
    echo "Please run: sudo ./profile_gpu_full.sh"
    exit 1
fi

# 默认脚本
SCRIPT="${1:-inferencellama3-1-70B.py}"

# 生成时间戳和输出名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="nsight_profiles"
SCRIPT_BASE=$(basename "$SCRIPT" .py)
OUTPUT_NAME="${OUTPUT_DIR}/${SCRIPT_BASE}_full_${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Starting Nsight Systems profiling (FULL MODE)..."
echo "Script: $SCRIPT"
echo "Output: ${OUTPUT_NAME}.nsys-rep"
echo "================================================"

# 运行 Nsight Systems profiling with GPU metrics
nsys profile \
  -o "$OUTPUT_NAME" \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --gpu-metrics-devices=all \
  --gpu-metrics-frequency=10000 \
  --sample=cpu \
  --python-sampling=true \
  --force-overwrite=true \
  python "$SCRIPT"

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Profiling complete!"
else
    echo "Profiling finished with exit code: $EXIT_CODE"
fi
echo "================================================"
echo "Generated file: ${OUTPUT_NAME}.nsys-rep"
echo ""
echo "View results with:"
echo "  1. GUI:   nsys-ui ${OUTPUT_NAME}.nsys-rep"
echo "  2. Stats: nsys stats ${OUTPUT_NAME}.nsys-rep"
echo ""

# 自动生成统计报告
if [ -f "${OUTPUT_NAME}.nsys-rep" ]; then
    echo "Generating statistics report..."
    nsys stats "${OUTPUT_NAME}.nsys-rep" > "${OUTPUT_NAME}_stats.txt" 2>&1
    echo "Statistics saved to: ${OUTPUT_NAME}_stats.txt"
fi

echo "================================================"
