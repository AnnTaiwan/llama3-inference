#!/usr/bin/env python3
"""
选择性 Profiling 包装器
只 profile inference 的核心部分，跳过模型加载等初始化阶段
"""

import sys
import torch

# 启用 CUDA Profiler API
torch.cuda.profiler.start()

# 初始化阶段不 profile
print("[Profiler] Skipping initialization phase...")

# 导入并运行主程序
if len(sys.argv) > 1:
    script_path = sys.argv[1]
else:
    script_path = "inferencellama3-1-70B_overlap_metrics.py"

# 读取并执行目标脚本
with open(script_path, 'r') as f:
    code = f.read()

# 在 profiling 范围内执行
print(f"[Profiler] Starting profiling of {script_path}")
torch.cuda.profiler.start()

try:
    exec(compile(code, script_path, 'exec'))
finally:
    torch.cuda.profiler.stop()
    print("[Profiler] Profiling stopped")
