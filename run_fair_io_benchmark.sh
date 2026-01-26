#!/bin/bash
# 公平的 IO benchmark 测试脚本
# 展示 Raw DirectIO 在冷启动场景下的优势

set -e

echo "=================================================================================================="
echo "Raw DirectIO vs FS Buffered IO 性能测试 - 公平对比版本"
echo "=================================================================================================="
echo ""
echo "测试配置:"
echo "  - IO 大小: 1MB, 4MB, 16MB (模型权重加载场景)"
echo "  - 并发度: 1, 4, 8 threads (充分发挥 NVMe 性能)"
echo "  - 测试模式: 冷启动 (每 10 次迭代 drop cache)"
echo "  - 迭代次数: 200 次/配置 (足够的样本用于 CDF)"
echo ""
echo "关键改进:"
echo "  ✅ 定期 drop cache，避免 FS page cache 不公平优势"
echo "  ✅ 大 IO 测试，DirectIO 擅长的场景"
echo "  ✅ 高并发测试，充分发挥 NVMe 并发能力"
echo "  ✅ 使用 os.pread() 减少 FS 开销，更公平对比"
echo ""
echo "预计运行时间: ~5-10 分钟 (取决于 drop cache 频率)"
echo "=================================================================================================="
echo ""

# 检查 sudo 权限
echo "检查 sudo 权限（需要用于 drop caches）..."
if sudo -n true 2>/dev/null; then
    echo "✅ sudo 权限可用"
else
    echo "⚠️  需要 sudo 权限来 drop caches，请输入密码："
    sudo -v
fi

echo ""
echo "开始测试..."
echo ""

# 运行 benchmark
python3 benchmark_io_latency_cdf.py

echo ""
echo "=================================================================================================="
echo "测试完成！"
echo "=================================================================================================="
echo ""
echo "结果已保存到: /home/roger/logs/io_latency_cdf/"
echo ""
echo "查看 CDF 曲线图："
echo "  python3 benchmark_io_latency_cdf.py --plot"
echo ""
