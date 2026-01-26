#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载 HuggingFace 上的 openai/gdpval 数据集，并保存到本地：
1. 保存为 HuggingFace 原生格式（save_to_disk）
2. 每个 split 另外导出为 JSONL，方便自己用脚本处理

使用方法：
    python download_gdpval.py --out-dir ./gdpval_data
"""

import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="gdpval_data",
        help="保存数据集的输出目录（默认：gdpval_data）",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("🚀 正在从 HuggingFace 加载数据集 openai/gdpval ...")
    ds = load_dataset("Anthropic/AnthropicInterviewer")  # 会自动下载并缓存

    # 1) 保存为 HuggingFace 原生格式（以后可以用 load_from_disk 直接加载）
    hf_disk_path = os.path.join(out_dir, "hf_dataset")
    print(f"💾 保存 HF 原生格式到: {hf_disk_path}")
    ds.save_to_disk(hf_disk_path)

    # 2) 每个 split 额外导出为 JSONL，方便直接查看或自定义处理
    print("📄 正在导出各个 split 为 JSONL ...")
    for split_name, split_ds in ds.items():
        jsonl_path = os.path.join(out_dir, f"{split_name}.jsonl")
        print(f"  - 导出 {split_name} -> {jsonl_path}")
        split_ds.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print("✅ 完成！数据已下载并保存。")

if __name__ == "__main__":
    main()
