#!/bin/bash
# Simplified GPU Profiling - More stable, less overhead

SCRIPT="${1:-inferencellama3-1-70B.py}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="nsight_profiles"
SCRIPT_BASE=$(basename "$SCRIPT" .py)
OUTPUT_NAME="${OUTPUT_DIR}/${SCRIPT_BASE}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Nsight Systems Profiling (Simple Mode)"
echo "Script: $SCRIPT"
echo "Output: ${OUTPUT_NAME}.nsys-rep"
echo "================================================"

# 最简化配置，只追踪 CUDA 核心功能
nsys profile \
  -o "$OUTPUT_NAME" \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --stop-on-exit=true \
  python "$SCRIPT"

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Profiling complete!"
else
    echo "✗ Exit code: $EXIT_CODE"
fi
echo "================================================"

if [ -f "${OUTPUT_NAME}.nsys-rep" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_NAME}.nsys-rep" | cut -f1)
    echo "Generated: ${OUTPUT_NAME}.nsys-rep ($FILE_SIZE)"
    echo ""
    echo "View with: nsys-ui ${OUTPUT_NAME}.nsys-rep"
    echo ""

    # 生成简单统计
    echo "Generating stats..."
    nsys stats "${OUTPUT_NAME}.nsys-rep" --report cuda_gpu_kern_sum > "${OUTPUT_NAME}_kernels.txt" 2>&1
    echo "Kernel stats: ${OUTPUT_NAME}_kernels.txt"
else
    echo "⚠ Warning: .nsys-rep file not found!"
fi

echo "================================================"
