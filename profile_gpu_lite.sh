#!/bin/bash
# Ultra-lightweight GPU profiling - minimal overhead
# Only traces CUDA kernels and NVTX markers, skips most API calls

SCRIPT="${1:-inferencellama3-1-70B.py}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="nsight_profiles"
SCRIPT_BASE=$(basename "$SCRIPT" .py)
OUTPUT_NAME="${OUTPUT_DIR}/${SCRIPT_BASE}_lite_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Nsight Lite Profiling (Minimal Overhead)"
echo "Script: $SCRIPT"
echo "Output: ${OUTPUT_NAME}.nsys-rep"
echo "================================================"

# 最小化开销配置：
# - 只追踪 CUDA kernels 和 NVTX
# - 不追踪 API 调用细节
# - 禁用所有采样
# - 禁用 backtrace
nsys profile \
  -o "$OUTPUT_NAME" \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --sample=none \
  --backtrace=none \
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
else
    echo "⚠ Warning: .nsys-rep file not found!"
fi

echo "================================================"
