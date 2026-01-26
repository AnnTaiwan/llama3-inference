#!/bin/bash
# GPU Profiling Script for Llama3 Inference using Nsight Systems

# 使用方法:
#   ./profile_gpu.sh                                    # 默认使用 inferencellama3-1-70B.py
#   ./profile_gpu.sh inferencellama3-1-70B_overlap_metrics.py
#   ./profile_gpu.sh your_script.py

# 默认脚本
SCRIPT="${1:-inferencellama3-1-70B.py}"

# 生成时间戳和输出名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="nsight_profiles"
SCRIPT_BASE=$(basename "$SCRIPT" .py)
OUTPUT_NAME="${OUTPUT_DIR}/${SCRIPT_BASE}_${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Starting Nsight Systems profiling..."
echo "Script: $SCRIPT"
echo "Output: ${OUTPUT_NAME}.nsys-rep"
echo "================================================"

# 运行 Nsight Systems profiling
# 注意：如果需要 GPU metrics，需要 sudo 权限
# 不使用 --gpu-metrics-devices 以避免权限问题，其他 trace 足够分析 GPU 使用情况
nsys profile \
  -o "$OUTPUT_NAME" \
  --trace=cuda,nvtx,cudnn,cublas \
  --cuda-memory-usage=true \
  --sample=none \
  --backtrace=none \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --stop-on-exit=true \
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
echo "  3. Export to SQLite: nsys export --type=sqlite ${OUTPUT_NAME}.nsys-rep"
echo ""

# 自动生成统计报告
if [ -f "${OUTPUT_NAME}.nsys-rep" ]; then
    echo "Generating statistics report..."
    nsys stats "${OUTPUT_NAME}.nsys-rep" > "${OUTPUT_NAME}_stats.txt" 2>&1
    echo "Statistics saved to: ${OUTPUT_NAME}_stats.txt"
    echo ""

    # 显示 GPU 使用率统计（如果有）
    echo "Quick GPU Stats Summary:"
    grep -A 10 "GPU" "${OUTPUT_NAME}_stats.txt" | head -20 || echo "GPU stats not found in report"
fi

echo "================================================"
