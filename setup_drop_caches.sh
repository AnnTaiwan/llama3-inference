#!/bin/bash
# 配置无密码 sudo drop_caches

echo "正在配置无密码 sudo drop_caches..."
echo ""
echo "需要输入一次 sudo 密码："
echo ""

# 创建 sudoers 配置
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/drop_caches > /dev/null

# 设置正确的权限
sudo chmod 440 /etc/sudoers.d/drop_caches

# 验证配置
if sudo -n bash -c "echo 3 | tee /proc/sys/vm/drop_caches > /dev/null 2>&1"; then
    echo "✅ 配置成功！现在可以无密码执行 drop_caches"
    echo ""
    echo "测试："
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo "✅ drop_caches 执行成功"
else
    echo "❌ 配置失败，请检查"
    exit 1
fi

echo ""
echo "现在可以运行 benchmark 了："
echo "  python benchmark_io_latency_cdf.py"
