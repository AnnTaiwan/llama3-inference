from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelArgs
from .layers import (
    RMSNorm,
    EncoderBlock,
    precompute_theta_pos_frequencies,
)

@dataclass
class _RuntimeSwitch:
    """运行时开关：控制是否使用管线化 forward"""
    PIPELINED: bool = True   # 可通过外部配置/环境变量控制


class Transformer(nn.Module):
    """
    纯粹的 Forward 计算与 KV profiling，**不**负责权重载入 / 采样
    """

    def print_device_info(self, tokenizer=None):
        """
        打印设备和形状信息，用于调试设备一致性问题

        Args:
            tokenizer: 可选的 tokenizer，用于验证 vocab_size 一致性
        """
        print("=" * 80)
        print("Transformer 设备一致性检查")
        print("=" * 80)

        # 1. Tokenizer vocab size
        if tokenizer is not None:
            tokenizer_vocab_size = len(tokenizer)
            print(f"Tokenizer vocab_size:           {tokenizer_vocab_size}")

        # 2. Embedding 层信息
        print(f"embed_tokens.num_embeddings:    {self.embed_tokens.num_embeddings}")
        print(f"embed_tokens.embedding_dim:     {self.embed_tokens.embedding_dim}")
        print(f"embed_tokens.weight.shape:      {self.embed_tokens.weight.shape}")
        print(f"embed_tokens.weight.device:     {self.embed_tokens.weight.device}")
        print(f"embed_tokens.weight.dtype:      {self.embed_tokens.weight.dtype}")

        # 3. Output 层信息
        print(f"output.in_features:             {self.output.in_features}")
        print(f"output.out_features:            {self.output.out_features}")
        print(f"output.weight.shape:            {self.output.weight.shape}")
        print(f"output.weight.device:           {self.output.weight.device}")
        print(f"output.weight.dtype:            {self.output.weight.dtype}")

        # 4. 验证一致性
        print("\n" + "-" * 80)
        print("一致性验证:")
        print("-" * 80)

        # vocab_size 一致性
        if tokenizer is not None:
            if tokenizer_vocab_size == self.embed_tokens.num_embeddings:
                print(f"✅ Tokenizer vocab_size ({tokenizer_vocab_size}) == embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings})")
            else:
                print(f"❌ Tokenizer vocab_size ({tokenizer_vocab_size}) != embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings})")

        # embed vs output
        if self.embed_tokens.num_embeddings == self.output.out_features:
            print(f"✅ embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings}) == output.out_features ({self.output.out_features})")
        else:
            print(f"❌ embed_tokens.num_embeddings ({self.embed_tokens.num_embeddings}) != output.out_features ({self.output.out_features})")

        # 设备一致性
        embed_dev = self.embed_tokens.weight.device
        output_dev = self.output.weight.device
        if embed_dev == output_dev:
            print(f"✅ 设备一致: embed ({embed_dev}) == output ({output_dev})")
        else:
            print(f"❌ 设备不一致: embed ({embed_dev}) != output ({output_dev})")

        print("=" * 80)

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        # 确保 vocab_size 合法（必须 > 0）
        assert args.vocab_size > 0, f"vocab_size 必须 > 0，当前值: {args.vocab_size}"

        # 创建 embedding 层：num_embeddings = vocab_size，embedding_dim = dim
        # 注意：num_embeddings 必须等于后续加载权重的 shape[0]，否则前向查表会出错
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        # self.layers = nn.ModuleList(
        #     [EncoderBlock(args, i) for i in range(args.n_layers)]
        # )
        
        self.layers = nn.ModuleList()
        self.layer_infos: List[Dict[str, Any]] = []
        for i in range(args.n_layers):
            blk = EncoderBlock(args, i)
            self.layers.append(blk)
            self.layer_infos.append({
                "layer_id": i,
                "block": blk,          # 方便从 info 直接拿模块
                "extra": {}            # 留给后续任意字段
            })     
               
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # 输出投影层：必须与 embed_tokens 在 vocab 维度对齐
        # nn.Linear(in_features=dim, out_features=vocab_size)
        # output.weight.shape = [vocab_size, dim]，与 embed_tokens.weight 同形状
        # 许多模型会 tie weights（共享权重），但即使不共享，vocab_size 也必须一致
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            head_dim=args.dim // args.n_heads,
            seq_len=args.max_seq_len * 2,
            device=args.device,
            theta=args.rope_theta,
        )

        self.kv_times: List[float] = [0.0] * args.n_layers
        self.attn_times: List[float] = [0.0] * args.n_layers

    def _forward_pipelined(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        return_logits: bool = True,
    ) -> torch.Tensor | None:
        """
        跨层事件串接：全程异步提交 + 仅在最后等待。

        改进点：
        1) 使用 forward_async() 返回 (out, done_evt)
        2) L+1 的 forward_async 在提交时对 prev_done_evt 做 wait_event
        3) CPU 不在每层同步，而是在所有层排完后再等待最后一个事件
        4) 这样即使下一层不能提前算（数据依赖），CPU 也能提前把权重 H2D 挂到计算流

        Args:
            return_logits: 是否返回 logits。False 时只构建 KV cache，不计算输出（节省显存）
        """
        embed_dev = self.embed_tokens.weight.device
        if not embed_dev.type.startswith("cuda"):
            raise RuntimeError(
                f"embed_tokens.weight must be on CUDA, got {embed_dev}. "
                f"This should have been caught by _verify_and_fix_device_placement."
            )

        dev = embed_dev  # 使用实际的 CUDA device 对象，而不是字符串
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        # 1) embed：确保 tokens 与 embedding 在同一设备（与原实现一致）
        if tokens.device != embed_dev:
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
            tokens = tokens.to(embed_dev, non_blocking=True)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # 2) 执行 embedding
        h = self.embed_tokens(tokens)  # 与权重同设备（必然是 CUDA）

        # 3) 如有不一致，迁移到目标计算设备 + 统一 dtype
        if h.device != dev or h.dtype != dtype:
            h = h.to(device=dev, dtype=dtype, non_blocking=True)

        freqs_dev_key = str(dev)  # 用字符串作为缓存键
        if getattr(self, "_freqs_cached_dev", None) != freqs_dev_key:
            self._freqs_cached = self.freqs_complex.to(dev, non_blocking=True)
            self._freqs_cached_dev = freqs_dev_key
        freqs = self._freqs_cached

        prev_done = None
        for idx, info in enumerate(self.layer_infos):
            blk = info["block"]
            h, prev_done = blk.forward_async(h, start_pos, freqs, wait_on=prev_done)

            self.kv_times[idx] = blk.attention.kv_elapsed_time
            self.attn_times[idx] = blk.attention.attn_time

        # 4) 到真正要用输出时再等待最后一层事件（同步点）
        if prev_done is not None:
            with torch.cuda.device(dev):
                torch.cuda.current_stream(dev).wait_event(prev_done)
            # 释放最后一层的事件（可选：如果 stream_mnt 需要）
            try:
                pass
            except Exception:
                pass

        h = self.norm(h)

        if not return_logits:
            return None

        out = self.output(h).float()
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        return_logits: bool = True,
    ) -> torch.Tensor | None:
        """
        Args:
            return_logits: 是否返回 logits。False 时只构建 KV cache，不计算输出（节省显存）
        """

        use_pipe = getattr(self, "_runtime_switch", None)
        if use_pipe is None:
            self._runtime_switch = _RuntimeSwitch(PIPELINED=True)
            use_pipe = self._runtime_switch

        if use_pipe.PIPELINED:
            return self._forward_pipelined(tokens, start_pos, return_logits=return_logits)

        bsz, seqlen = tokens.shape
        embed_dev = self.embed_tokens.weight.device
        if not embed_dev.type.startswith("cuda"):
            raise RuntimeError(
                f"embed_tokens.weight must be on CUDA, got {embed_dev}. "
                f"This should have been caught by _verify_and_fix_device_placement."
            )

        dev = embed_dev  
        dtype = getattr(self, "param_dtype", torch.bfloat16)

        if not hasattr(self, '_device_info_printed'):
            print(f"[DEVICE] 计算设备: {dev}")
            print(f"[DEVICE] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
            if len(self.layer_infos) > 0 and hasattr(self.layer_infos[0]['block'], 'attention'):
                first_attn = self.layer_infos[0]['block'].attention
                if hasattr(first_attn, 'wq') and hasattr(first_attn.wq, 'weight'):
                    print(f"[DEVICE] layer[0].attention.wq.device: {first_attn.wq.weight.device}")
            self._device_info_printed = True
            
        # 1) embed：确保 tokens 与 embedding 在同一设备（与原实现一致）
        if tokens.device != embed_dev:
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
            tokens = tokens.to(embed_dev, non_blocking=True)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # 2) 执行 embedding 查表：此时 tokens 和 weight 已在同一设备
        h = self.embed_tokens(tokens)  # -> 与 embedding 同设备（必然是 CUDA）

        # 3) 如有不一致，迁移到目标计算设备 + 统一 dtype
        if h.device != dev or h.dtype != dtype:
            # 打印调试信息（首次调用时）
            if not hasattr(self, '_device_warning_printed'):
                print(f"[DEBUG] Embedding 输出设备: {h.device}, 目标计算设备: {dev}")
                print(f"[DEBUG] embed_tokens.weight.device: {self.embed_tokens.weight.device}")
                print("[DEBUG] 正在将激活转换到目标设备...")
                self._device_warning_printed = True
            h = h.to(device=dev, dtype=dtype, non_blocking=True)

        # 2) freqs 只在设备不一致时搬一次，避免每步重复 .to()
        freqs_dev_key = str(dev)  # 用字符串作为缓存键
        if getattr(self, "_freqs_cached_dev", None) != freqs_dev_key:
            self._freqs_cached = self.freqs_complex.to(dev, non_blocking=True)
            self._freqs_cached_dev = freqs_dev_key
        freqs = self._freqs_cached

        # 3) 运行时防呆：任何层若把激活搬回 CPU 立即报错（早失败）
        for idx, info in enumerate(self.layer_infos):
            blk: EncoderBlock = info["block"]
            h = blk(h, start_pos, freqs)
            if h.device != dev:
                raise RuntimeError(
                    f"EncoderBlock[{idx}] returned activation on {h.device}, "
                    f"expected {dev}. Check for any '.to(\"cpu\")' fallback."
                )
            self.kv_times[idx]  = blk.attention.kv_elapsed_time
            self.attn_times[idx] = blk.attention.attn_time

        h = self.norm(h)                  # 确认 norm 模块常驻 GPU
        if not return_logits:
            return None
        out = self.output(h).float()      # 输出用 float 以便后续 logits 运算
        return out
