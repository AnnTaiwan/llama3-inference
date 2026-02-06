import os
from pathlib import Path
import threading
import types
from typing import Any, Dict, List, Optional
import json
import csv
import uuid
import platform
import time
import re
from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 限制分块大小，减少碎片
os.environ["WSM_NO_FALLBACK"] = "1"

#  启用层级性能 profiling（CUDA timer 统计 attn/ffn/kv_fetch 等详细时间）
os.environ["LLM_PROFILE"] = "1"
CHUNK_SIZE = int(os.environ.setdefault("PREFILL_T_CHUNK", "512")) # prefill 分块大小
MIRCO_BATCH_SIZE  = os.environ.setdefault("MIRCO_BATCH_SIZE", "8") # micro batch size
ATTN_MICRO_B = os.environ.setdefault("ATTN_MICRO_B", "8") # attention micro batch size
import torch  # noqa: E402

# Optional NVTX for GPU decode-step ranges
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except Exception:
    nvtx = None
    NVTX_AVAILABLE = False

LOG_DIR = Path("/home/roger/logs")   # 自动创建
RUN_TAG = ""                        

from llama3.generator import LLaMA  # noqa: E402
from llama3.config import KVCacheArgs, load_runtime_config, runtime_config_to_dict  # noqa: E402
from llama3 import generator as _gen, stream_mnt  # noqa: E402
try:
    from llama3.layers import PERF_TRACKER 
except Exception:
    PERF_TRACKER = None

# ========== build()  ==========
_orig_build = _gen.LLaMA.build

def _debug_build(*args, **kw):
    mode       = kw.get("mode", None)
    load_model = kw.get("load_model", None)
    mode_cfg   = (kw.get("mode_config", {}) or {})
    raw_dev    = mode_cfg.get("raw_device")
    manifest   = mode_cfg.get("manifest_path") or mode_cfg.get("ssd_manifest_path")

    print(f"[MODE-DECISION] LLaMA.build(mode={mode}, load_model={load_model})")
    use_raw_ssd = (mode in {"ssd", "mixed"}) or (mode_cfg.get("weight_source") == "raw-ssd")
    print(f"[MODE-DECISION] use_raw_ssd={use_raw_ssd} raw_device={raw_dev} manifest={manifest}")

    llama = _orig_build(*args, **kw)

    has_wsm = hasattr(llama, "weight_streaming_manager")
    if has_wsm:
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        print(f"[MODE-DECISION] built: WSM present, ssd_enabled={ssd}")
    else:
        print("[MODE-DECISION] built: NO WSM ")
    return llama

_gen.LLaMA.build = staticmethod(_debug_build)

# =======  Profiler =======
PROFILER = None  

def _now_utc():
    return datetime.now(timezone.utc).isoformat()

def _flatten_extras(extras: dict):
    out = {}
    for k,v in (extras or {}).items():
        out[k] = v if (isinstance(v,(int,float,str,bool)) or v is None) else str(v)
    return out

class InferenceProfiler:
    def __init__(self, run_name: str | None = None):
        self.run_id   = run_name or f"run-{uuid.uuid4().hex[:8]}"
        self.t0_ns    = time.perf_counter_ns()
        self.timeline = []   
        self.active   = False
        self.cuda     = torch.cuda.is_available()
        self.forward_events = []      # GPU：[(kind,batch,seqlen,start_ev,end_ev)]
        self.forward_events_cpu = []  # CPU 回退：[(kind,batch,seqlen,dt_ms)]
        self.bookkeep  = {}
        
        # decode step 计数 + 是否启用 NVTX
        self.decode_step_idx = 0
        self.use_nvtx = bool(self.cuda and NVTX_AVAILABLE)

        # 供 finalize 合并的 WSM 运行期统计（由 main() 写入）
        self.wsm_runtime = None

        # 当前阶段：setup / prefill / decode
        self.phase: str = "setup"

        # 累加两个阶段的 compute 时间（单位：us）
        self.prefill_compute_us: float = 0.0
        self.decode_compute_us: float = 0.0

        # 累加两个阶段的 IO 时间（单位：ms）
        self.prefill_io_ms = {"ssd_to_cpu": 0.0, "h2d_param": 0.0}
        self.decode_io_ms = {"ssd_to_cpu": 0.0, "h2d_param": 0.0}

        # GPU timing events for overall inference (用于 finalize 中的 overlap 计算)
        self.gpu_t0 = None
        self.gpu_t1 = None

        # H2D GPU events for overlap analysis (记录每次 H2D 传输的 CUDA events)
        self.h2d_gpu_events = []

        # NVML samples for GPU utilization monitoring
        self.nvml_samples = []
        self._nvml_stop = None
        self._nvml_thread = None

        self.meta      = {
            "started_at_utc": _now_utc(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "device": ("cuda" if self.cuda else "cpu"),
        }
        if self.cuda:
            try:
                self.meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                self.meta["cuda_cc"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, category: str, **extras):
        s = time.perf_counter_ns()
        try:
            yield
        finally:
            e = time.perf_counter_ns()
            dur_ms = (e - s) / 1e6
            rec = {
                "name": name, "cat": category,
                "t_start_ms": (s - self.t0_ns) / 1e6,
                "t_end_ms":   (e - self.t0_ns) / 1e6,
                "dur_ms":     dur_ms,
            }
            rec.update(_flatten_extras(extras))
            self.timeline.append(rec)
            if name == "inference_e2e":
                self.bookkeep["inference_s_ns"] = s
                self.bookkeep["inference_e_ns"] = e

            # 实时按 phase 汇总 IO 时间
            phase = extras.get("phase") or self.phase
            if name.startswith("wsm.ssd_to_cpu"):
                if phase == "prefill":
                    self.prefill_io_ms["ssd_to_cpu"] += dur_ms
                elif phase == "decode":
                    self.decode_io_ms["ssd_to_cpu"] += dur_ms
            elif name.startswith("wsm.h2d_param"):
                if phase == "prefill":
                    self.prefill_io_ms["h2d_param"] += dur_ms
                elif phase == "decode":
                    self.decode_io_ms["h2d_param"] += dur_ms

    @contextmanager
    def inference_scope(self):
        self.active = True

        # 创建 GPU timing events（如果支持 CUDA）
        if self.cuda:
            self.gpu_t0 = torch.cuda.Event(enable_timing=True)
            self.gpu_t1 = torch.cuda.Event(enable_timing=True)
            self.gpu_t0.record()

        with self.span("inference_e2e", "inference"):
            yield

        # 记录结束 event
        if self.cuda and self.gpu_t1 is not None:
            self.gpu_t1.record()

        self.active = False
    
    def now_ms(self) -> float:
        """当前相对 t0 的墙钟时间（毫秒），供外部补丁使用。"""
        return (time.perf_counter_ns() - self.t0_ns) / 1e6

    # ---------------- NVML util  ----------------
    def _start_nvml_sampler(self):
        try:
            import pynvml  # pip install nvidia-ml-py
        except Exception as e:
            print(f"[NVML] pynvml not available: {e}")
            return

        interval_s = float(os.getenv("PROFILER_NVML_INTERVAL_S", "0.2"))
        interval_s = max(0.05, interval_s)

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        stop_evt = threading.Event()
        self._nvml_stop = stop_evt

        def _worker():
            while not stop_evt.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.nvml_samples.append({
                        "t_ms": self.now_ms(),
                        "gpu": int(getattr(util, "gpu", 0)),  # 修正：使用 "gpu" 而不是 "gpu_util"
                        "mem": int(getattr(util, "memory", 0)),  # 修正：使用 "mem" 而不是 "mem_util"
                        "mem_used_B": int(getattr(mem, "used", 0)),
                        "mem_total_B": int(getattr(mem, "total", 0)),
                    })
                except Exception:
                    pass
                time.sleep(interval_s)

        th = threading.Thread(target=_worker, name="nvml_sampler", daemon=True)
        self._nvml_thread = th
        th.start()

    def _stop_nvml_sampler(self):
        try:
            import pynvml
        except Exception:
            pynvml = None

        if self._nvml_stop is not None:
            try:
                self._nvml_stop.set()
            except Exception:
                pass
        if self._nvml_thread is not None:
            try:
                self._nvml_thread.join(timeout=2.0)
            except Exception:
                pass

        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def set_phase(self, phase: str):
        """切换当前 phase：setup / prefill / decode"""
        assert phase in ("setup", "prefill", "decode"), f"Invalid phase: {phase}"
        self.phase = phase

    def wrap_model_forward(self, model):
        orig = model.forward

        def _classify_args(args, kwargs):
            """
            解析出 tokens / B / T / start_pos
            返回: (kind, B, T, start_pos) 其中 kind: "prefill" or "decode"
            """
            # 1) 找到 tokens tensor
            cand = None
            for k in ("tokens", "input_ids"):
                t = kwargs.get(k, None)
                if torch.is_tensor(t) and t.dim() == 2:
                    cand = t
                    break  
            if cand is None:
                for a in args:
                    if torch.is_tensor(a) and a.dtype in (torch.long, torch.int32, torch.int64) and a.dim() == 2:
                        cand = a
                        break  
            if cand is None:
                return "unknown", None, None, None

            B, T = int(cand.size(0)), int(cand.size(1))

            # 2) 尝试解析 start_pos
            start_pos = kwargs.get("start_pos", None)
            if start_pos is None and len(args) >= 2:
                # 可能是位置参数：forward(tokens, start_pos, ...)
                if isinstance(args[1], int):
                    start_pos = args[1]

            # 3) 判断 kind
            kind = None
            if T is not None and T > 1:
                # 不管 start_pos 是 0 还是 >0，都当 prefill（包含 chunk prefill）
                kind = "prefill"
            elif T == 1:
                # 单 token，一律认为是 decode
                kind = "decode"
            elif start_pos is not None and start_pos > 0:
                # T 取不到、但 start_pos>0 时再兜底当 decode
                kind = "decode"
            else:
                kind = "unknown"

            return kind, B, T, start_pos

        def wrapped(*args, **kwargs):
            if not self.active:
                return orig(*args, **kwargs)

            kind, B, T, start_pos = _classify_args(args, kwargs)

            # ---- 按 kind 自动设置 profiler phase ----
            if kind == "prefill":
                self.set_phase("prefill")
            elif kind == "decode":
                self.set_phase("decode")

            # ---- 统一维护 decode_step_idx（不依赖 NVTX 是否开启）----
            step_idx = None
            if kind == "decode":
                step_idx = self.decode_step_idx
                self.decode_step_idx += 1

            # 在进入 forward 前，告诉 PERF_TRACKER 当前 phase + step
            if PERF_TRACKER is not None:
                # prefill 不需要记录 step，就给 None；decode 给具体 step_idx
                PERF_TRACKER.set_step_context(kind, step_idx if kind == "decode" else None)

            if self.cuda:
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)

                label = None
                if self.use_nvtx and NVTX_AVAILABLE and kind == "decode":
                    # 这里直接用上面算好的 step_idx
                    label = f"decode_step_{step_idx}_B{B}_T{T}"
                    nvtx.range_push(label)

                s_ev.record()
                try:
                    out = orig(*args, **kwargs)
                finally:
                    e_ev.record()
                    if NVTX_AVAILABLE and label is not None:
                        nvtx.range_pop()
                    # 退出 forward 后，清掉 step context
                    if PERF_TRACKER is not None:
                        PERF_TRACKER.set_step_context(None, None)

                self.forward_events.append((kind, B, T, s_ev, e_ev))
                return out
            else:
                t0 = time.time()
                try:
                    out = orig(*args, **kwargs)
                finally:
                    t1 = time.time()
                    # CPU 版本也结束时清 context
                    if PERF_TRACKER is not None:
                        PERF_TRACKER.set_step_context(None, None)

                self.forward_events_cpu.append((kind, B, T, (t1 - t0) * 1000.0))
                return out

        model.forward = wrapped


    # 供 WSM 补丁使用
    def span_if_active(self, name, category, **extras):
        return self.span(name, category, **extras) if self is not None else nullcontext()

    def _compute_decode_stats(self, arr, batch_size: int = 1):
        """
        通用的延迟统计工具。

        参数:
            arr: 一个包含若干耗时（单位: ms）的列表。
            batch_size: 每次调用 forward 处理的样本数（用于换算 token/s）。

        返回:
            统计字典，字段包括:
              - count: 样本数
              - sum_ms: 总耗时 (ms)
              - mean_ms: 平均值 (ms)
              - p50_ms / p90_ms / p99_ms / p999_ms: 分位数 (ms)
              - max_ms: 最大值 (ms)
              - decode_toks_per_s: 如果 batch_size>0，则认为每个样本生成 batch_size 个 token，
                                   给出对应的 token/s（否则为 None）
        """
        if not arr:
            return {
                "count": 0,
                "sum_ms": 0.0,
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p99_ms": None,
                "p999_ms": None,
                "max_ms": None,
                "decode_toks_per_s": None,
            }

        # 过滤非法值并转成 float
        arr = [float(x) for x in arr if isinstance(x, (int, float))]
        if not arr:
            return {
                "count": 0,
                "sum_ms": 0.0,
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p99_ms": None,
                "p999_ms": None,
                "max_ms": None,
                "decode_toks_per_s": None,
            }

        s = sorted(arr)
        n = len(s)

        def q(p: float) -> float:
            if n == 1:
                return s[0]
            # 简单按 index 取值即可，足够用于统计
            idx = int((n - 1) * p)
            idx = max(0, min(n - 1, idx))
            return s[idx]

        total_ms = float(sum(s))
        mean_ms = total_ms / n
        max_ms = s[-1]

        total_s = total_ms / 1000.0
        total_tokens = n * max(int(batch_size), 1)
        toks_per_s = total_tokens / total_s if total_s > 0 else None

        return {
            "count": n,
            "sum_ms": total_ms,
            "mean_ms": mean_ms,
            "p50_ms": q(0.50),
            "p90_ms": q(0.90),
            "p99_ms": q(0.99),
            "p999_ms": q(0.999),
            "max_ms": max_ms,
            "decode_toks_per_s": toks_per_s,
        }


    def finalize(
        self,
        tokens_in: int,
        tokens_out: int,
        extra_meta: Optional[Dict[str, Any]] = None,
        kv_stats: Optional[Dict[str, Any]] = None,
        wsm_runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        把 timeline + forward_events 汇总成结构化结果，并补充一组更「论文友好」的指标:

        - TE2E: 端到端延迟 (Time End-to-End)
        - TTFT: Time To First Token
        - TBT: Time Between Tokens（稳定阶段，忽略第一个 token）
        - decode 期间 IO / Compute 的近似 overlap ratio
        - 每层 decoder 的 compute / IO 汇总
        """
        # -------- meta / runtime 补充 --------
        if extra_meta:
            self.meta.update(extra_meta)

        if wsm_runtime is not None:
            self.wsm_runtime = wsm_runtime
        elif self.wsm_runtime is None:
            self.wsm_runtime = {}

        # 从 meta 中获取 batch_size（用于所有 token/s 相关统计）
        bsz = int(self.meta.get("batch_size") or 1)

        # -------- 1. 从 CUDA events 解析 prefill / decode 每 step 耗时 --------
        decode_ms: List[float] = []
        prefill_ms: List[float] = []
        if torch.cuda.is_available():
            for ev in self.forward_events:
                # forward_events: (kind, B, T, start_evt, end_evt)
                kind, B, T, start_evt, end_evt = ev
                if start_evt is None or end_evt is None:
                    continue
                try:
                    end_evt.synchronize()
                    dt_ms = float(start_evt.elapsed_time(end_evt))
                except Exception:
                    continue

                if kind == "decode":
                    decode_ms.append(dt_ms)
                elif kind == "prefill":
                    prefill_ms.append(dt_ms)

        # -------- 2. E2E / prefill / decode 墙钟时间 --------
        # inference_scope() 用 "inference_e2e" 包裹整个推理
        infer_spans = [ev for ev in self.timeline if ev.get("name") == "inference_e2e"]
        infer_t0_ms: Optional[float] = None
        infer_t1_ms: Optional[float] = None
        if infer_spans:
            infer_t0_ms = min(float(ev["t_start_ms"]) for ev in infer_spans)
            infer_t1_ms = max(float(ev["t_end_ms"]) for ev in infer_spans)
            e2e_ms = infer_t1_ms - infer_t0_ms
        else:
            e2e_ms = None

        prefill_total_ms = float(sum(prefill_ms)) if prefill_ms else 0.0
        decode_total_ms = float(sum(decode_ms)) if decode_ms else 0.0

        # 预热 GPU decoder window（warmup layer）时间
        warmup_spans = [ev for ev in self.timeline if ev.get("name") == "gpu_window_warmup"]
        warmup_total_ms = sum(float(ev.get("dur_ms", 0.0)) for ev in warmup_spans)
        warmup_calls = len(warmup_spans)

        # -------- 3. 按 category 统计各阶段墙钟时间 --------
        by_cat: Dict[str, float] = {}
        for ev in self.timeline:
            cat = ev.get("cat")
            dur = float(ev.get("dur_ms", 0.0))
            if not cat or dur <= 0:
                continue
            by_cat[cat] = by_cat.get(cat, 0.0) + dur

        sum_cat = {
            "total_ms": sum(by_cat.values()) if by_cat else 0.0,
            "by_cat_ms": by_cat,
        }

        # -------- 4. decode per-step 统计（用于 TBT、decode throughput）--------
        decode_stats = self._compute_decode_stats(decode_ms, batch_size=bsz)

        # TBT 采用「稳定阶段」的 token 间隔，所以跳过第一个 token
        tbt_stats = None
        if len(decode_ms) > 1:
            tbt_stats = self._compute_decode_stats(decode_ms[1:], batch_size=bsz)

        # TTFT: prefill 总时间 + 第一个 decode step
        # 这和常见文献/文档中“从请求到第一个 token 的延迟”一致
        ttft_ms = None
        if prefill_total_ms and decode_ms:
            ttft_ms = prefill_total_ms + decode_ms[0]
        elif prefill_total_ms:
            ttft_ms = prefill_total_ms

        # -------- 5. slack_ms 统计（group ready 到真正使用之间的间隔）--------
        slack_times: List[float] = []
        for ev in self.timeline:
            if ev.get("name") == "wsm.wait_group_ready":
                s = ev.get("slack_ms")
                if isinstance(s, (int, float)) and s >= 0:
                    slack_times.append(float(s))

        slack_stats = (
            self._compute_decode_stats(slack_times)
            if slack_times
            else {
                "count": 0,
                "sum_ms": 0.0,
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p99_ms": None,
                "p999_ms": None,
                "max_ms": None,
                "decode_toks_per_s": None,
            }
        )

        # -------- 6. WSM 维度的总 IO 时间 / 等待时间 --------
        def _sum_by_name(name: str) -> float:
            total = 0.0
            for ev in self.timeline:
                if ev.get("name") != name:
                    continue
                total += float(ev.get("dur_ms", 0.0))
            return total

        def _sum_io(name: str, group_key: Optional[str] = None) -> Dict[str, Any]:
            total = 0.0
            by_group: Dict[str, float] = {}
            for ev in self.timeline:
                if ev.get("name") != name:
                    continue
                dur = float(ev.get("dur_ms", 0.0))
                total += dur
                if group_key is not None:
                    g = str(ev.get(group_key, "unknown"))
                    by_group[g] = by_group.get(g, 0.0) + dur
            out: Dict[str, Any] = {"total_ms": total}
            if group_key is not None:
                out["by_group"] = by_group
            return out

        # -------- 5. WSM 高层统计（和你之前一样）--------
        wait_total_ms = _sum_by_name("wsm.wait_group_ready")
        ensure_total_ms = _sum_by_name("wsm.ensure_module_on_gpu")

        # IO 按类型统计（SSD->CPU / CPU->GPU），带 phase 方便看 prefill / decode
        ssd_io = _sum_io("wsm.ssd_to_cpu_layer", group_key="phase")
        h2d_io = _sum_io("wsm.h2d_param", group_key="phase")

        w_io = {
            "ssd_to_cpu_ms": ssd_io,
            "h2d_param_ms": h2d_io,
        }

        wsm_stats: Dict[str, Any] = {
            "wait_group_ready": {"total_ms": wait_total_ms},
            "ensure_module_on_gpu": {"total_ms": ensure_total_ms},
            "io": w_io,
        }
        if self.wsm_runtime:
            wsm_stats["runtime"] = self.wsm_runtime

        # -------- 6.x 论文友好指标：用 interval-union 计算 IO busy time / bubble / overlap --------
        # 关键点：
        # - sum(dur_ms) 会把并发的 I/O 请求 latency 叠加，从而严重夸大 IO_total；
        # - 对吞吐/利用率/overlap 更自洽的是用区间并集 (union) 代表“设备忙碌时长”。
        def _merge_intervals(intervals: List[tuple[float, float]]) -> List[tuple[float, float]]:
            if not intervals:
                return []
            intervals = sorted(intervals)
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            return [(float(s), float(e)) for s, e in merged]

        def _union_ms(intervals: List[tuple[float, float]]) -> float:
            m = _merge_intervals(intervals)
            return float(sum(e - s for s, e in m))

        def _clip_to_infer_window(s: float, e: float) -> Optional[tuple[float, float]]:
            if infer_t0_ms is None or infer_t1_ms is None:
                return (s, e)
            # 严格限制在 [infer_t0, infer_t1]
            s2 = max(float(s), float(infer_t0_ms))
            e2 = min(float(e), float(infer_t1_ms))
            if e2 <= s2:
                return None
            return (s2, e2)

        def _collect_intervals(name: str, *, phase: Optional[str] = None) -> List[tuple[float, float]]:
            out: List[tuple[float, float]] = []
            for ev in self.timeline:
                if ev.get("name") != name:
                    continue
                if phase is not None and str(ev.get("phase", "unknown")) != phase:
                    continue
                s = float(ev.get("t_start_ms", 0.0))
                e = float(ev.get("t_end_ms", 0.0))
                clipped = _clip_to_infer_window(s, e)
                if clipped is not None:
                    out.append(clipped)
            return out

        def _sum_dur_ms(name: str, *, phase: Optional[str] = None) -> float:
            total = 0.0
            for ev in self.timeline:
                if ev.get("name") != name:
                    continue
                if phase is not None and str(ev.get("phase", "unknown")) != phase:
                    continue
                total += float(ev.get("dur_ms", 0.0))
            return float(total)

        def _sum_bytes(name: str, *, phase: Optional[str] = None) -> float:
            total = 0.0
            for ev in self.timeline:
                if ev.get("name") != name:
                    continue
                if phase is not None and str(ev.get("phase", "unknown")) != phase:
                    continue
                total += float(ev.get("bytes", 0.0))
            return float(total)

        # CPU time domain：SSD reads / (fallback) H2D enqueue / wait (bubble)
        ssd_intv = _collect_intervals("wsm.ssd_to_cpu_layer")
        h2d_intv_cpu = _collect_intervals("wsm.h2d_param")
        wait_intv = _collect_intervals("wsm.wait_group_ready")

        ssd_union_ms = _union_ms(ssd_intv)
        h2d_union_ms_cpu = _union_ms(h2d_intv_cpu)
        wait_union_ms = _union_ms(wait_intv)
        io_union_ms_cpu = _union_ms(ssd_intv + h2d_intv_cpu)

        ssd_bytes = _sum_bytes("wsm.ssd_to_cpu_layer")
        h2d_bytes = _sum_bytes("wsm.h2d_param")

        # 以 inference wall 作为平均吞吐的分母；以 union 作为 active 吞吐的分母
        wall_s = (e2e_ms / 1000.0) if (e2e_ms is not None and e2e_ms > 0) else None
        ssd_active_s = ssd_union_ms / 1000.0 if ssd_union_ms > 0 else None
        h2d_active_s_cpu = h2d_union_ms_cpu / 1000.0 if h2d_union_ms_cpu > 0 else None

        def _bw_gbs(bytes_: float, seconds: Optional[float]) -> Optional[float]:
            if seconds is None or seconds <= 0 or bytes_ <= 0:
                return None
            return float(bytes_ / seconds / 1e9)

        bubble_ratio = (wait_union_ms / e2e_ms) if (e2e_ms and e2e_ms > 0) else None
        io_hidden_ratio_est = None
        io_hidden_ms_est = None
        if io_union_ms_cpu > 0:
            io_hidden_ms_est = max(0.0, io_union_ms_cpu - wait_union_ms)
            io_hidden_ratio_est = max(0.0, min(1.0, io_hidden_ms_est / io_union_ms_cpu))

        paper_cpu_io = {
            "inference_window_ms": {
                "t0_ms": infer_t0_ms,
                "t1_ms": infer_t1_ms,
                "wall_ms": e2e_ms,
            },
            "bubble": {
                "wait_group_ready_union_ms": wait_union_ms,
                "wait_group_ready_sum_ms": _sum_dur_ms("wsm.wait_group_ready"),
                "bubble_ratio": bubble_ratio,
            },
            "ssd": {
                "bytes": ssd_bytes,
                "active_union_ms": ssd_union_ms,
                "busy_fraction": (ssd_union_ms / e2e_ms) if (e2e_ms and e2e_ms > 0) else None,
                "avg_bw_GBps": _bw_gbs(ssd_bytes, wall_s),
                "active_bw_GBps": _bw_gbs(ssd_bytes, ssd_active_s),
                "sum_dur_ms": _sum_dur_ms("wsm.ssd_to_cpu_layer"),
                "concurrency_factor": (
                    (_sum_dur_ms("wsm.ssd_to_cpu_layer") / ssd_union_ms) if ssd_union_ms > 0 else None
                ),
            },
            "h2d_cpu_enq": {
                "bytes": h2d_bytes,
                "active_union_ms": h2d_union_ms_cpu,
                "busy_fraction": (h2d_union_ms_cpu / e2e_ms) if (e2e_ms and e2e_ms > 0) else None,
                "avg_bw_GBps": _bw_gbs(h2d_bytes, wall_s),
                "active_bw_GBps": _bw_gbs(h2d_bytes, h2d_active_s_cpu),
                "sum_dur_ms": _sum_dur_ms("wsm.h2d_param"),
                "note": "This is CPU-side enqueue/driver overhead; real PCIe DMA time should be measured with CUDA events (see h2d_gpu section).",
            },
            "io_overlap_est": {
                "io_union_ms": io_union_ms_cpu,
                "io_hidden_ms_est": io_hidden_ms_est,
                "io_hidden_ratio_est": io_hidden_ratio_est,
            },
        }

        # 按 phase 拆分的 union 指标（prefill / decode）
        phase_union_metrics: Dict[str, Any] = {}
        for _ph in ("prefill", "decode"):
            _ssd = _collect_intervals("wsm.ssd_to_cpu_layer", phase=_ph)
            _h2d = _collect_intervals("wsm.h2d_param", phase=_ph)
            _wait = _collect_intervals("wsm.wait_group_ready", phase=_ph)
            _io_union = _union_ms(_ssd + _h2d)
            _wait_union = _union_ms(_wait)
            _hidden_ms = max(0.0, _io_union - _wait_union) if _io_union > 0 else None
            _hidden_ratio = (
                max(0.0, min(1.0, (_hidden_ms / _io_union))) if (_io_union and _hidden_ms is not None) else None
            )
            phase_union_metrics[_ph] = {
                "ssd_union_ms": _union_ms(_ssd),
                "h2d_union_ms_cpu": _union_ms(_h2d),
                "wait_union_ms": _wait_union,
                "io_union_ms_cpu": _io_union,
                "io_hidden_ms_est": _hidden_ms,
                "io_hidden_ratio_est": _hidden_ratio,
            }

        # -------- 6. 每个 decoder layer 的 compute / IO / 带宽统计 --------
        decoder_layers_global: Dict[str, float] = {}
        decoder_layers_per_layer: Dict[str, Any] = {}

        # 6.1 GPU 计算时间（layers.py 里的 PERF_TRACKER）
        perf_per_layer: Dict[int, Dict[str, float]] = {}
        perf_stats: Dict[str, Any] = {}
        # decode 阶段按 layer 聚合的 MHA / FFN 时间，用来从总量中扣出 prefill 部分
        decode_attn_per_layer: Dict[int, float] = {}
        decode_ffn_per_layer: Dict[int, float] = {}

        if PERF_TRACKER is not None:
            try:
                # {"global": {...}, "per_layer": {...}, "per_step": {...}}
                perf_stats = PERF_TRACKER.get_stats()
                decoder_layers_global = dict(perf_stats.get("global", {}))
                perf_per_layer = perf_stats.get("per_layer", {}) or {}

                # 6.1.1 从 per_step 里按 phase 汇总各层的时间
                per_step = perf_stats.get("per_step", {}) or {}
                for phase, steps in per_step.items():
                    for step_idx, layers in steps.items():
                        for layer_id_raw, stats in layers.items():
                            # layer_id 可能是 int 或 str，这里统一成 int
                            try:
                                layer_id = int(layer_id_raw)
                            except Exception:
                                continue

                            attn_us = float(stats.get("attn_us", 0.0))
                            ffn_us = float(stats.get("ffn_us", 0.0))
                            kv_us = float(stats.get("kv_fetch_us", 0.0))
                            mem_us = float(stats.get("memory_alloc_us", 0.0))
                            w_hbm_us = float(stats.get("weights_hbm_us", 0.0))

                            # per_step 里通常没有单独的 total_forward_us，用子项之和近似
                            forward_us = float(stats.get("total_forward_us", 0.0))
                            if forward_us <= 0.0:
                                forward_us = attn_us + ffn_us + kv_us + mem_us + w_hbm_us

                            if forward_us > 0.0:
                                if phase == "prefill":
                                    self.prefill_compute_us += forward_us
                                elif phase == "decode":
                                    self.decode_compute_us += forward_us

                            # 只在 decode 阶段，把各层的 MHA/FFN 累加起来，后面用来做 prefill/decoder 拆分
                            if phase == "decode":
                                if attn_us > 0.0:
                                    decode_attn_per_layer[layer_id] = (
                                        decode_attn_per_layer.get(layer_id, 0.0) + attn_us
                                    )
                                if ffn_us > 0.0:
                                    decode_ffn_per_layer[layer_id] = (
                                        decode_ffn_per_layer.get(layer_id, 0.0) + ffn_us
                                    )
            except Exception:
                perf_stats = {}
                decoder_layers_global = {}
                perf_per_layer = {}
                decode_attn_per_layer = {}
                decode_ffn_per_layer = {}

        # 6.2 从 WSM timeline 按 layer_idx 聚合 IO 时间 + 字节数
        layer_io: Dict[int, Dict[str, float]] = {}
        for ev in self.timeline:
            lid = ev.get("layer_idx")
            if lid is None:
                continue
            name = ev.get("name", "")
            if not (
                name.startswith("wsm.ssd_to_cpu_layer")
                or name.startswith("wsm.h2d_param")
                or name.startswith("wsm.wait_group_ready")
            ):
                continue

            lid = int(lid)
            entry = layer_io.setdefault(
                lid,
                {
                    "ssd_to_cpu_ms": 0.0,
                    "ssd_to_cpu_bytes": 0.0,
                    "h2d_param_ms": 0.0,
                    "h2d_param_bytes": 0.0,
                    "wait_group_ready_ms": 0.0,
                },
            )
            dur = float(ev.get("dur_ms", 0.0))
            b = float(ev.get("bytes", 0.0))

            if name.startswith("wsm.ssd_to_cpu_layer"):
                entry["ssd_to_cpu_ms"] += dur
                entry["ssd_to_cpu_bytes"] += b
            elif name.startswith("wsm.h2d_param"):
                entry["h2d_param_ms"] += dur
                entry["h2d_param_bytes"] += b
            elif name.startswith("wsm.wait_group_ready"):
                entry["wait_group_ready_ms"] += dur

        # 6.3 合并 per-layer compute + IO + 带宽，并拆出 prefill / decode 的 MHA / FFN
        def _us_to_ms(x: float) -> Optional[float]:
            return x / 1000.0 if x and x > 0.0 else None

        all_layer_ids = sorted(set(list(perf_per_layer.keys()) + list(layer_io.keys())))
        for lid in all_layer_ids:
            lp = perf_per_layer.get(lid, {})  # 来自 cuda_timer 的各类 us 统计
            li = layer_io.get(lid, {})

            attn_us = float(lp.get("attn_us", 0.0))
            ffn_us = float(lp.get("ffn_us", 0.0))
            kv_us = float(lp.get("kv_fetch_us", 0.0))
            total_forward_us = float(lp.get("total_forward_us", 0.0))
            if total_forward_us == 0.0 and (attn_us or ffn_us or kv_us):
                total_forward_us = attn_us + ffn_us + kv_us

            mem_us = float(lp.get("memory_alloc_us", 0.0))
            weights_hbm_us = float(lp.get("weights_hbm_us", 0.0))

            # 根据 per_step(decode) 里累积的数据拆出 prefill / decode 的 MHA / FFN
            dec_attn_us = float(decode_attn_per_layer.get(lid, 0.0))
            dec_ffn_us = float(decode_ffn_per_layer.get(lid, 0.0))
            pre_attn_us = max(0.0, attn_us - dec_attn_us)
            pre_ffn_us = max(0.0, ffn_us - dec_ffn_us)

            ssd_ms = float(li.get("ssd_to_cpu_ms", 0.0))
            ssd_bytes = float(li.get("ssd_to_cpu_bytes", 0.0))
            h2d_ms = float(li.get("h2d_param_ms", 0.0))
            h2d_bytes = float(li.get("h2d_param_bytes", 0.0))
            wait_ms = float(li.get("wait_group_ready_ms", 0.0))

            io_total_ms = ssd_ms + h2d_ms + wait_ms

            # 平均带宽 (GB/s, 1GB=1e9 bytes)，注意 ms -> s
            ssd_gbps = (
                ssd_bytes / ssd_ms / 1e6 if (ssd_ms > 0.0 and ssd_bytes > 0.0) else None
            )
            h2d_gbps = (
                h2d_bytes / h2d_ms / 1e6 if (h2d_ms > 0.0 and h2d_bytes > 0.0) else None
            )

            decoder_layers_per_layer[str(lid)] = {
                # MHA / FFN / KV 的累计计算时间（us）
                "compute_us": {
                    "attn_us": attn_us,
                    "ffn_us": ffn_us,
                    "kv_fetch_us": kv_us,
                    "total_forward_us": total_forward_us,
                    "memory_alloc_us": mem_us,
                    "weights_hbm_us": weights_hbm_us,
                    "prefill_attn_us": pre_attn_us,
                    "decode_attn_us": dec_attn_us,
                    "prefill_ffn_us": pre_ffn_us,
                    "decode_ffn_us": dec_ffn_us,
                },
                "compute_ms_split": {
                    "prefill_attn_ms": _us_to_ms(pre_attn_us),
                    "decode_attn_ms": _us_to_ms(dec_attn_us),
                    "prefill_ffn_ms": _us_to_ms(pre_ffn_us),
                    "decode_ffn_ms": _us_to_ms(dec_ffn_us),
                },
                # 兼容老字段：只放时间（ms）
                "io_ms": {
                    "ssd_to_cpu_ms": ssd_ms,
                    "h2d_param_ms": h2d_ms,
                    "wait_group_ready_ms": wait_ms,
                },
                # 新增：IO 字节数
                "io_bytes": {
                    "ssd_to_cpu_bytes": ssd_bytes if ssd_bytes > 0 else None,
                    "h2d_param_bytes": h2d_bytes if h2d_bytes > 0 else None,
                },
                # 新增：平均带宽（GB/s）
                "io_bandwidth_gbps": {
                    "ssd_to_cpu_gbps": ssd_gbps,
                    "h2d_param_gbps": h2d_gbps,
                },
                "summary": {
                    # 单层累计前向时间（prefill+decode，近似）
                    "compute_ms_pure": total_forward_us / 1000.0
                    if total_forward_us
                    else None,
                    "io_ms_total": io_total_ms,
                },
            }


        # 6.4 MHA / FFN decouple 的全局统计（prefill / decode / overall）
        # 使用前面已经写好的 decoder_layers_per_layer 里的 prefill / decode MHA/FFN 拆分
        global_prefill_attn_us = 0.0
        global_prefill_ffn_us = 0.0
        global_decode_attn_us = 0.0
        global_decode_ffn_us = 0.0

        # 理论对比：
        # - sequential_ms: MHA + FFN 完全串行执行的总时间
        # - ideal_decoupled_ms: MHA / FFN 完全重叠时的理论最短时间 = max(MHA, FFN)
        seq_prefill_ms = 0.0
        ideal_prefill_ms = 0.0
        seq_decode_ms = 0.0
        ideal_decode_ms = 0.0

        for _lid_str, layer_stats in decoder_layers_per_layer.items():
            csplit = layer_stats.get("compute_us", {}) or {}

            pre_attn = float(csplit.get("prefill_attn_us") or 0.0)
            pre_ffn = float(csplit.get("prefill_ffn_us") or 0.0)
            dec_attn = float(csplit.get("decode_attn_us") or 0.0)
            dec_ffn = float(csplit.get("decode_ffn_us") or 0.0)

            global_prefill_attn_us += pre_attn
            global_prefill_ffn_us += pre_ffn
            global_decode_attn_us += dec_attn
            global_decode_ffn_us += dec_ffn

            # Prefill 理论串行 / 理想 overlap 时间（按 layer 累加）
            if pre_attn > 0.0 or pre_ffn > 0.0:
                seq_prefill_ms += (pre_attn + pre_ffn) / 1000.0
                ideal_prefill_ms += max(pre_attn, pre_ffn) / 1000.0

            # Decode 同理
            if dec_attn > 0.0 or dec_ffn > 0.0:
                seq_decode_ms += (dec_attn + dec_ffn) / 1000.0
                ideal_decode_ms += max(dec_attn, dec_ffn) / 1000.0

        def _safe_speedup(seq_ms: float, ideal_ms: float) -> Optional[float]:
            if ideal_ms <= 0.0 or seq_ms <= 0.0:
                return None
            return seq_ms / ideal_ms

        mha_ffn_stats: Dict[str, Any] = {
            "prefill": {
                "attn_ms": global_prefill_attn_us / 1000.0 if global_prefill_attn_us > 0.0 else None,
                "ffn_ms": global_prefill_ffn_us / 1000.0 if global_prefill_ffn_us > 0.0 else None,
                # 理论：如果 MHA/FFN 完全串行
                "sequential_ms": seq_prefill_ms or None,
                # 理论：如果 MHA/FFN 在每个 layer 内完全 overlap
                "ideal_decoupled_ms": ideal_prefill_ms or None,
                # 理论 decouple 加速比 = 串行时间 / 理想重叠时间
                "theoretical_speedup": _safe_speedup(seq_prefill_ms, ideal_prefill_ms),
            },
            "decode": {
                "attn_ms": global_decode_attn_us / 1000.0 if global_decode_attn_us > 0.0 else None,
                "ffn_ms": global_decode_ffn_us / 1000.0 if global_decode_ffn_us > 0.0 else None,
                "sequential_ms": seq_decode_ms or None,
                "ideal_decoupled_ms": ideal_decode_ms or None,
                "theoretical_speedup": _safe_speedup(seq_decode_ms, ideal_decode_ms),
            },
        }

        total_attn_us = global_prefill_attn_us + global_decode_attn_us
        total_ffn_us = global_prefill_ffn_us + global_decode_ffn_us
        total_seq_ms = seq_prefill_ms + seq_decode_ms
        total_ideal_ms = ideal_prefill_ms + ideal_decode_ms

        mha_ffn_stats["overall"] = {
            "attn_ms": total_attn_us / 1000.0 if total_attn_us > 0.0 else None,
            "ffn_ms": total_ffn_us / 1000.0 if total_ffn_us > 0.0 else None,
            "sequential_ms": total_seq_ms or None,
            "ideal_decoupled_ms": total_ideal_ms or None,
            "theoretical_speedup": _safe_speedup(total_seq_ms, total_ideal_ms),
        }

        # 6.5 全局 decode 阶段 IO-Compute overlap 估计（原来的 6.4）
        decoder_compute_ms_total = sum(
            v["summary"]["compute_ms_pure"] or 0.0
            for v in decoder_layers_per_layer.values()
        )
        decoder_io_ms_total = sum(
            v["summary"]["io_ms_total"] for v in decoder_layers_per_layer.values()
        )
        decoder_wall_ms = float(decode_stats.get("sum_ms") or decode_total_ms or 0.0)

        overlap_ratio: Optional[float] = None
        overlap_ms: Optional[float] = None
        uncovered_io_ms: Optional[float] = None
        if decoder_wall_ms > 0.0 and decoder_io_ms_total > 0.0:
            # 理论：wall_time ≈ max(compute, io) + 其它开销
            # 近似：compute + io - wall ≈ 被覆盖掉的 IO 时间
            raw_overlap = decoder_compute_ms_total + decoder_io_ms_total - decoder_wall_ms
            overlap_ms = max(0.0, raw_overlap)
            overlap_ratio = max(0.0, min(1.0, overlap_ms / decoder_io_ms_total))
            uncovered_io_ms = max(0.0, decoder_io_ms_total - overlap_ms)

        decoder_layers_summary = {
            "compute_ms_total": decoder_compute_ms_total,
            "io_ms_total": decoder_io_ms_total,
            "decode_wall_ms": decoder_wall_ms,
            "overlap_ms_approx": overlap_ms,
            "uncovered_io_ms_approx": uncovered_io_ms,
            "overlap_ratio": overlap_ratio,
        }

        # 把 overlap 信息塞进 decode 这块，方便 timings.decode.* 直接访问
        if overlap_ratio is not None:
            decode_stats["overlap_ratio"] = overlap_ratio
            decode_stats["approx_compute_ms_total"] = decoder_compute_ms_total
            decode_stats["approx_io_ms_total"] = decoder_io_ms_total
            decode_stats["approx_overlap_ms"] = overlap_ms
            decode_stats["approx_uncovered_io_ms"] = uncovered_io_ms

        # 最终 decoder_layers 结构：多一个 mha_ffn_decouple，专门给论文用
        decoder_layers = {
            "global": decoder_layers_global,
            "per_layer": decoder_layers_per_layer,
            "summary": decoder_layers_summary,
            "mha_ffn_decouple": mha_ffn_stats,
        }

        # -------- 8. throughput 统计 --------
        # 先准备 token 统计
        tokens_in_total = tokens_in * bsz if tokens_in is not None else None
        tokens_out_total = tokens_out * bsz if tokens_out is not None else None

        # 生成的新 token 数（decode 阶段输出）
        decode_tokens_total = None
        if tokens_in is not None and tokens_out is not None and tokens_out > tokens_in:
            decode_tokens_per_seq = tokens_out - tokens_in
            decode_tokens_total = decode_tokens_per_seq * bsz

        # 各阶段时间（秒）
        prefill_s = prefill_total_ms / 1000.0 if prefill_total_ms > 0 else None
        decode_s = decode_total_ms / 1000.0 if decode_total_ms > 0 else None
        e2e_s = e2e_ms / 1000.0 if e2e_ms > 0 else None

        # 1) Prefill 吞吐：只看输入 token
        prefill_input_toks_per_s = (
            (tokens_in_total / prefill_s)
            if (tokens_in_total is not None and prefill_s and prefill_s > 0)
            else None
        )

        # 2) Decode 阶段吞吐：只看新生成 token 和 decode 时间
        decode_output_toks_per_s = (
            (decode_tokens_total / decode_s)
            if (decode_tokens_total is not None and decode_s and decode_s > 0)
            else None
        )

        # 3) 端到端输出吞吐：把 prefill 时间也算进来
        e2e_output_toks_per_s = (
            (decode_tokens_total / e2e_s)
            if (decode_tokens_total is not None and e2e_s and e2e_s > 0)
            else None
        )

        # 4) 端到端 total token 吞吐（有些论文会用）
        total_processed_tokens = None
        if tokens_in_total is not None or decode_tokens_total is not None:
            total_processed_tokens = (tokens_in_total or 0) + (decode_tokens_total or 0)

        e2e_total_toks_per_s = (
            (total_processed_tokens / e2e_s)
            if (total_processed_tokens is not None and e2e_s and e2e_s > 0)
            else None
        )

        throughput = {
            # 保留原字段，方便兼容
            "prefill_toks_per_s": prefill_input_toks_per_s,
            "decode_toks_per_s": decode_output_toks_per_s,

            # 新字段：语义更清晰
            "prefill_input_toks_per_s": prefill_input_toks_per_s,
            "decode_output_toks_per_s": decode_output_toks_per_s,
            "e2e_output_toks_per_s": e2e_output_toks_per_s,
            "e2e_total_toks_per_s": e2e_total_toks_per_s,
        }


        # -------- 9. TE2E / TTFT / TBT 等汇总到 timings --------

        # 9.1 计算 phase breakdown (prefill / decode 的 IO + compute + overlap)
        def calc_overlap(compute_ms: float, io_ms: float, wall_ms: float):
            """计算 overlap：理论时间 (compute+io) vs 实际墙钟时间 (wall)"""
            if wall_ms <= 0:
                return 0.0, io_ms, 0.0
            overlap_ms = max(0.0, compute_ms + io_ms - wall_ms)
            uncovered_io_ms = max(0.0, io_ms - overlap_ms)
            overlap_ratio = overlap_ms / io_ms if io_ms > 0 else 0.0
            return overlap_ms, uncovered_io_ms, overlap_ratio

        # Prefill
        prefill_compute_ms = self.prefill_compute_us / 1000.0
        prefill_io_ms = self.prefill_io_ms["ssd_to_cpu"] + self.prefill_io_ms["h2d_param"]
        prefill_wall_ms = prefill_total_ms if prefill_total_ms else 0.0
        prefill_union = phase_union_metrics.get("prefill", {}) if "phase_union_metrics" in locals() else {}
        prefill_io_union_ms = float(prefill_union.get("io_union_ms_cpu") or 0.0)
        prefill_wait_union_ms = float(prefill_union.get("wait_union_ms") or 0.0)
        prefill_io_hidden_ratio_est = prefill_union.get("io_hidden_ratio_est")
        pref_ovl_ms, pref_uncovered_ms, pref_ratio = calc_overlap(
            prefill_compute_ms, prefill_io_ms, prefill_wall_ms
        )

        # Decode
        decode_compute_ms = self.decode_compute_us / 1000.0
        decode_io_ms = self.decode_io_ms["ssd_to_cpu"] + self.decode_io_ms["h2d_param"]
        decode_wall_ms = decode_stats.get("sum_ms", 0.0) if decode_stats else 0.0
        decode_union = phase_union_metrics.get("decode", {}) if "phase_union_metrics" in locals() else {}
        decode_io_union_ms = float(decode_union.get("io_union_ms_cpu") or 0.0)
        decode_wait_union_ms = float(decode_union.get("wait_union_ms") or 0.0)
        decode_io_hidden_ratio_est = decode_union.get("io_hidden_ratio_est")
        dec_ovl_ms, dec_uncovered_ms, dec_ratio = calc_overlap(
            decode_compute_ms, decode_io_ms, decode_wall_ms
        )

        # Per-token metrics
        tokens_in_total = tokens_in * bsz if tokens_in is not None else None
        tokens_out_total = tokens_out * bsz if tokens_out is not None else None
        prefill_ms_per_token = (
            prefill_wall_ms / tokens_in_total if tokens_in_total and tokens_in_total > 0 else None
        )
        decode_ms_per_token = (
            decode_wall_ms / tokens_out_total if tokens_out_total and tokens_out_total > 0 else None
        )

        timings: Dict[str, Any] = {
            "e2e_ms": e2e_ms,
            "te2e_ms": e2e_ms,  # alias，方便直接在论文里引用
            "prefill_total_ms": prefill_total_ms,
            "ttft_ms": ttft_ms,
            "first_token_latency_ms": ttft_ms,  # 保持向后兼容
            "decode": decode_stats,
            "tbt_steady_ms": tbt_stats,
            "warmup_total_ms": warmup_total_ms,
            "warmup_calls": warmup_calls,
            "by_category_ms": sum_cat,
            # Phase breakdown
            "phase_breakdown": {
                "prefill": {
                    "wall_ms": prefill_wall_ms,
                    "compute_ms": prefill_compute_ms,
                    "io_ms": prefill_io_ms,
                    # 更自洽：用 interval union 表示“IO busy time”
                    "io_union_ms_cpu": prefill_io_union_ms,
                    "bubble_union_ms": prefill_wait_union_ms,
                    "io_hidden_ratio_est": prefill_io_hidden_ratio_est,
                    "io_breakdown_ms": {
                        "ssd_to_cpu": self.prefill_io_ms["ssd_to_cpu"],
                        "h2d_param": self.prefill_io_ms["h2d_param"],
                    },
                    "overlap_ms": pref_ovl_ms,
                    "uncovered_io_ms": pref_uncovered_ms,
                    "overlap_ratio": pref_ratio,
                    "ms_per_token": prefill_ms_per_token,
                },
                "decode": {
                    "wall_ms": decode_wall_ms,
                    "compute_ms": decode_compute_ms,
                    "io_ms": decode_io_ms,
                    # 更自洽：用 interval union 表示“IO busy time”
                    "io_union_ms_cpu": decode_io_union_ms,
                    "bubble_union_ms": decode_wait_union_ms,
                    "io_hidden_ratio_est": decode_io_hidden_ratio_est,
                    "io_breakdown_ms": {
                        "ssd_to_cpu": self.decode_io_ms["ssd_to_cpu"],
                        "h2d_param": self.decode_io_ms["h2d_param"],
                    },
                    "overlap_ms": dec_ovl_ms,
                    "uncovered_io_ms": dec_uncovered_ms,
                    "overlap_ratio": dec_ratio,
                    "ms_per_token": decode_ms_per_token,
                },
            },
        }

        # -------- 10. GPU 内存峰值 --------
        memory_stats: Dict[str, Any] = {}
        if torch.cuda.is_available():
            try:
                alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                memory_stats = {
                    "gpu_peak_allocated_gb": float(alloc_gb),
                    "gpu_peak_reserved_gb": float(reserved_gb),
                }
            except Exception:
                memory_stats = {}

        # -------- 10.5 GPU/H2D overlap（CUDA event，跨 stream 计算）--------
        # 说明：CUDA event 时间戳在同一 device 上是全局一致的，可用 elapsed_time 做跨 stream 对齐。
        paper_gpu_h2d: Dict[str, Any] = {}
        if self.cuda and self.gpu_t0 is not None and self.gpu_t1 is not None:
            try:
                # 仅同步 default stream 的 t1，避免无关后台 stream 让 finalize 卡死。
                self.gpu_t1.synchronize()
                gpu_infer_ms = float(self.gpu_t0.elapsed_time(self.gpu_t1))

                def _merge(intervals: List[tuple[float, float]]) -> List[tuple[float, float]]:
                    if not intervals:
                        return []
                    intervals = sorted(intervals)
                    out = [list(intervals[0])]
                    for s, e in intervals[1:]:
                        if s <= out[-1][1]:
                            out[-1][1] = max(out[-1][1], e)
                        else:
                            out.append([s, e])
                    return [(float(s), float(e)) for s, e in out]

                def _union_len(intervals: List[tuple[float, float]]) -> float:
                    m = _merge(intervals)
                    return float(sum(e - s for s, e in m))

                def _intersect_len(a: List[tuple[float, float]], b: List[tuple[float, float]]) -> float:
                    a = _merge(a)
                    b = _merge(b)
                    i = j = 0
                    total = 0.0
                    while i < len(a) and j < len(b):
                        s = max(a[i][0], b[j][0])
                        e = min(a[i][1], b[j][1])
                        if e > s:
                            total += (e - s)
                        if a[i][1] <= b[j][1]:
                            i += 1
                        else:
                            j += 1
                    return float(total)

                # compute intervals：用 forward wrapper 的 start/end events
                compute_intv: List[tuple[float, float]] = []
                for kind, _bsz, _seqlen, s_ev, e_ev in self.forward_events:
                    # forward 的 end_evt 在 finalize 前已经 synchronize 过；这里再 query 一下做保险
                    try:
                        if hasattr(e_ev, "query") and (not e_ev.query()):
                            continue
                        s = float(self.gpu_t0.elapsed_time(s_ev))
                        e = float(self.gpu_t0.elapsed_time(e_ev))
                        # clip 到 [0, gpu_infer_ms]
                        s = max(0.0, min(s, gpu_infer_ms))
                        e = max(0.0, min(e, gpu_infer_ms))
                        if e > s:
                            compute_intv.append((s, e))
                    except Exception:
                        continue

                # h2d intervals：来自 WSM _install_group_on_gpu patch
                h2d_intv: List[tuple[float, float]] = []
                h2d_bytes_infer = 0.0
                for ev in self.h2d_gpu_events:
                    try:
                        e_ev = ev["e_ev"]
                        s_ev = ev["s_ev"]
                        if hasattr(e_ev, "query") and (not e_ev.query()):
                            # 如果 transfer 还没完成（可能是推理结束后的预取尾巴），就先跳过
                            continue
                        s = float(self.gpu_t0.elapsed_time(s_ev))
                        e = float(self.gpu_t0.elapsed_time(e_ev))
                        # clip 到 [0, gpu_infer_ms]
                        s_c = max(0.0, min(s, gpu_infer_ms))
                        e_c = max(0.0, min(e, gpu_infer_ms))
                        if e_c > s_c:
                            h2d_intv.append((s_c, e_c))
                            # bytes 也按同样的 clip（近似）：只要 interval 有交集就计入 bytes
                            h2d_bytes_infer += float(ev.get("bytes", 0.0))
                    except Exception:
                        continue

                compute_union_ms = _union_len(compute_intv)
                h2d_union_ms = _union_len(h2d_intv)
                overlap_ms = _intersect_len(compute_intv, h2d_intv)
                overlap_ratio = (overlap_ms / h2d_union_ms) if h2d_union_ms > 0 else None

                def _bw(bytes_: float, ms: float) -> Optional[float]:
                    if bytes_ <= 0 or ms <= 0:
                        return None
                    return float(bytes_ / (ms / 1000.0) / 1e9)

                paper_gpu_h2d = {
                    "gpu_infer_ms": gpu_infer_ms,
                    "compute_union_ms": compute_union_ms,
                    "h2d_union_ms": h2d_union_ms,
                    "h2d_bytes": h2d_bytes_infer,
                    "h2d_active_bw_GBps": _bw(h2d_bytes_infer, h2d_union_ms) if h2d_union_ms > 0 else None,
                    "h2d_avg_bw_GBps": _bw(h2d_bytes_infer, gpu_infer_ms) if gpu_infer_ms > 0 else None,
                    "h2d_compute_overlap_ms": overlap_ms,
                    "h2d_overlap_ratio": overlap_ratio,
                }
            except Exception:
                paper_gpu_h2d = {}

        # NVML util summary（如果启用 NVML sampler）
        nvml_summary: Dict[str, Any] = {}
        if self.nvml_samples:
            try:
                gpu_utils = [int(s.get("gpu", 0)) for s in self.nvml_samples if s.get("gpu") is not None]
                mem_utils = [int(s.get("mem", 0)) for s in self.nvml_samples if s.get("mem") is not None]
                def _pct(xs, p):
                    if not xs:
                        return None
                    xs2 = sorted(xs)
                    i = int((len(xs2)-1)*p)
                    return xs2[i]
                nvml_summary = {
                    "num_samples": len(self.nvml_samples),
                    "gpu_util_avg_pct": (sum(gpu_utils)/len(gpu_utils)) if gpu_utils else None,
                    "gpu_util_p50_pct": _pct(gpu_utils, 0.50),
                    "gpu_util_p90_pct": _pct(gpu_utils, 0.90),
                    "mem_util_avg_pct": (sum(mem_utils)/len(mem_utils)) if mem_utils else None,
                }
            except Exception:
                nvml_summary = {}

        # -------- 11. 汇总成 result --------
        self.result = {
            "run": self.meta
            | {
                "run_id": self.run_id,
                "finished_at_utc": _now_utc(),
            },
            "counts": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_in_total": tokens_in * bsz if tokens_in is not None else None,
                "tokens_out_total": tokens_out * bsz if tokens_out is not None else None,
            },
            "timings": timings,
            "throughput": throughput,
            "paper_metrics": {
                "cpu_io": paper_cpu_io,
                "phase_union": phase_union_metrics,
                "gpu_h2d": paper_gpu_h2d,
                "nvml": nvml_summary,
            },
            "wsm": wsm_stats,
            "decoder_layers": decoder_layers,
            "decode_step_ms": decode_ms,
            "timeline": self.timeline,
            "memory": memory_stats,
        }

        if kv_stats is not None:
            self.result["kv_cache"] = kv_stats

        return self.result




    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.result, f, ensure_ascii=False, indent=2)
        elif path.lower().endswith(".csv"):
            rows = []
            for ev in self.timeline:
                r = {"kind":"span","name":ev["name"],"cat":ev["cat"],
                     "t_start_ms":ev["t_start_ms"],"t_end_ms":ev["t_end_ms"],"dur_ms":ev["dur_ms"]}
                rows.append(r)
            for i,dt in enumerate(self.result.get("decode_step_ms", [])):
                rows.append({"kind":"decode_step","name":f"decode_{i:04d}","cat":"inference","t_start_ms":"", "t_end_ms":"", "dur_ms":dt})
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["kind","name","cat","t_start_ms","t_end_ms","dur_ms"])
                w.writeheader()
                w.writerows(rows)
        else:
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump(self.result, f, ensure_ascii=False, indent=2)

# ===== 路径与常量（按你的环境） =====
PROMPT_TXT = Path(os.getenv(
    "PROMPT_TXT",
    "/home/roger/llama3-inference/prompts/prompts_batch512_len2048.txt",
))
RAW_DEV  = os.getenv("RAW_DEV",  "/dev/nvme0n1p4")
MANIFEST = os.getenv("MANIFEST", "/data1/70b-fixed.runtime_manifest.json")
CKPT_DIR = os.getenv("CKPT_DIR", "/home/roger/.llama/checkpoints/Llama3.1-70B")

# ===== Workload 配置：统一改 batch / 生成 token =====
# 可以直接改这里的默认值，也可以用环境变量覆盖



# ---------- 系统/GPU 内存快照 ----------
def _read_status():
    def _grep(path, keys):
        out = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    for k in keys:
                        if line.startswith(k + ":"):
                            out[k] = line.split(":")[1].strip()
        except Exception:
            pass
        return out
    s = _grep("/proc/self/status", ["VmRSS","VmHWM","VmLck"])
    m = _grep("/proc/meminfo", ["MemAvailable","CommitLimit","Committed_AS","Cached","Buffers"])
    return s, m

def _gpu_mem():
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    st  = torch.cuda.memory_stats(dev)
    return {
        "alloc_GB": st.get("allocated_bytes.all.current", 0)/(1<<30),
        "rsrv_GB":  st.get("reserved_bytes.all.current", 0)/(1<<30),
    }

def probe(stage: str):
    s, m = _read_status()
    g    = _gpu_mem()
    print(f"\n[MEM] {stage}")
    print(f"  VmRSS={s.get('VmRSS','?')}  VmLck(pinned)={s.get('VmLck','?')}  "
          f"CommitLimit={m.get('CommitLimit','?')}  Committed_AS={m.get('Committed_AS','?')}  "
          f"MemAvailable={m.get('MemAvailable','?')}")
    if g:
        print(f"  GPU: allocated={g['alloc_GB']:.2f} GiB  reserved={g['rsrv_GB']:.2f} GiB")
    print()

# ---------- 扫描参数在 meta/cpu/cuda 上的占用 ----------
def dump_param_inventory(model, tag):
    buckets = {"cpu":0, "cuda":0, "meta":0, "other":0}
    big_cpu = []
    for n,p in model.named_parameters(recurse=True):
        b = p.numel() * p.element_size()
        if getattr(p, "is_meta", False):
            buckets["meta"] += b
        elif hasattr(p, "device"):
            t = p.device.type
            if t == "cpu":
                buckets["cpu"] += b
                if b >= (64<<20):
                    big_cpu.append((n,b))
            elif t == "cuda":
                buckets["cuda"] += b
            else:
                buckets["other"] += b
        else:
            buckets["other"] += b
    def f(x):
        return f"{x/(1<<30):.2f} GiB"
    print(f"[PARAMS] {tag}: cpu={f(buckets['cpu'])}, cuda={f(buckets['cuda'])}, meta={f(buckets['meta'])}, other={f(buckets['other'])}")
    if big_cpu:
        big_cpu.sort(key=lambda x:-x[1])
        print("  [big-cpu] top:")
        for n,b in big_cpu[:10]:
            print(f"   - {n}  {b/(1<<20):.1f} MiB")

# ---------- 运行时覆盖：收敛 pinned/注册池 ----------
def apply_runtime_overrides():
    """
    把注册总量钳在 ≤256MiB，并把 EXTENT_BYTES 降到 1MiB，降低高阶页 order 压力。
    """
    cfg = load_runtime_config({
        "pinned": {
            "WEIGHT_PINNED_BYTES":      8  << 30,
            "KV_PINNED_BYTES":          8  << 30,
            "EXTENT_BYTES":             1  << 20,   # 1MiB
            "PINNED_REGISTER_CHUNK":   16  << 20,   # 16MiB
            "PINNED_REGISTER_N":            8,      # 128MiB
        },
        "regpool": {
            "REG_POOL_N_BUFFERS":           8,
            "REG_POOL_BUF_BYTES":     16 << 20,     # ~128MiB 传送带
        },
        "io": {
            "RAW_IO_QD_WRITE":             24,      # 写队列深度
            "IO_RAW_THROTTLE_MS":          30,      # 写带宽窗口
        }
    })
    D = runtime_config_to_dict(cfg)
    p = D["pinned"]
    need  = int(p["WEIGHT_PINNED_BYTES"])
    chunk = int(p["PINNED_REGISTER_CHUNK"])
    target_total = min(need // 2, 256 << 20)  # 目标 ≤ 256MiB
    newN = max(1, target_total // chunk)
    p["PINNED_REGISTER_N"] = newN
    cfg = load_runtime_config({"pinned": p, "io": D["io"]})
    print("[RuntimeConfig] pinned =", runtime_config_to_dict(cfg)["pinned"])
    print("[RuntimeConfig] io =", runtime_config_to_dict(cfg)["io"])
    return cfg

# ---------- KV 池：懒分配 + 单块 ≥ 单个 KV 块 ----------
def configure_kv_pool():
    # DRAM 配置
    KVCacheArgs.dram_limit_gb     = 32.0
    KVCacheArgs.dram_sizing_batch = 32
    KVCacheArgs.block_bytes       = 1 * 1024 * 1024
    KVCacheArgs.preallocate       = False
    KVCacheArgs.lazy_init         = True

    # 关闭 push 即时镜像，采用后移/聚合写（避免与权重 H2D 冲突）
    KVCacheArgs.mirror_on_push = False

    # I/O 节流与写速率配置（与权重 H2D 仲裁）
    KVCacheArgs.IO_RAW_THROTTLE_MS     = 25
    KVCacheArgs.NVME_WRITE_TARGET_MBPS = 1500

    if hasattr(KVCacheArgs, "prefer_bf16"):
        KVCacheArgs.prefer_bf16 = True

    print(f"[KVArgs] dram_limit={KVCacheArgs.dram_limit_gb} GiB, "
          f"block_bytes={KVCacheArgs.block_bytes//(1<<20)} MiB, prealloc={KVCacheArgs.preallocate}")
    print(f"[KVArgs] mirror_on_push={KVCacheArgs.mirror_on_push}, "
          f"IO_RAW_THROTTLE_MS={KVCacheArgs.IO_RAW_THROTTLE_MS}, "
          f"NVME_WRITE_TARGET_MBPS={KVCacheArgs.NVME_WRITE_TARGET_MBPS}")

# ---------- 识别“实际运行的模式” ----------
def classify_mode(llama) -> str:
    """
    返回：'ssd-streaming' / 'cpu-gpu-streaming' / 'full-gpu' / 'full-cpu' / 'meta-only'
    """
    m = llama.model
    if hasattr(llama, "weight_streaming_manager"):
        wsm = llama.weight_streaming_manager
        ssd = bool(getattr(wsm, "ssd_enabled", False) or getattr(wsm, "ssd", None))
        cpu_warm = getattr(wsm, "disable_cpu_warm", None)
        mode = "ssd-streaming" if ssd else "cpu-gpu-streaming"
        print(f"[MODE] detected={mode}  (has WSM, ssd={ssd}, disable_cpu_warm={cpu_warm})")
        return mode
    cpu, cuda, meta = 0,0,0
    for _, p in m.named_parameters():
        b = p.numel() * p.element_size()

        if getattr(p, "is_meta", False) or p.device.type == "meta":
            meta += b
        elif p.device.type == "cpu":
            cpu += b
        elif p.device.type == "cuda":
            cuda += b

    if cuda > 0 and cpu == 0 and meta == 0:
        print("[MODE] detected=full-gpu")
        mode = "full-gpu"
    elif cpu > 0 and cuda == 0 and meta == 0:
        print("[MODE] detected=full-cpu")
        mode = "full-cpu"
    elif meta > 0 and cpu == 0 and cuda == 0:
        print("[MODE] detected=meta-only")
        mode = "meta-only"
    else:
        print("[MODE] detected=mixed")
        mode = "mixed"
    print("[MODE] mixed/unrecognized (check PARAMS dump below)")
    return "unknown"


# ===== WSM wait_group_ready 包装（仅添加 Profiler 计时，不改逻辑） =====
def _wrap_wait_group_ready(original_method):
    """
    包装 WSM.wait_group_ready，添加：
      - profiler 计时埋点；
      - 从 group ready 到首次 compute 使用的 slack_ms 估计；
      - pipeline 水位（ring / inflight / in_use 的最大值）。
    """
    def wrapped(self, layer_idx: int, group: str, compute_stream=None):
        prof = globals().get("PROFILER")
        extras = {
            "layer_idx": int(layer_idx),
            "group": str(group),
        }

        # phase：prefill / decode（用于论文/分析拆分）
        if prof is not None:
            try:
                extras["phase"] = getattr(self, "_phase", None) or getattr(prof, "phase", "unknown")
            except Exception:
                extras["phase"] = getattr(prof, "phase", "unknown")

        # 1) slack_ms = 现在时间 - 该 group ready 事件记录时间
        if prof is not None and hasattr(self, "_group_ready_wallclock"):
            try:
                key = (int(layer_idx), str(group))
                ready_ms = self._group_ready_wallclock.get(key)
                if isinstance(ready_ms, (int, float)):
                    now_ms = prof.now_ms()
                    extras["slack_ms"] = max(0.0, now_ms - float(ready_ms))
            except Exception:
                pass

        # 2) pipeline 水位统计（最大 ring/inflight/in_use）
        if hasattr(self, "_pipeline_watermark"):
            try:
                lock = getattr(self, "_group_lock", None)
                if lock is not None:
                    with lock:
                        ring_len     = len(getattr(self, "_gpu_group_ring", []))
                        inflight_len = len(getattr(self, "_gpu_group_inflight", set()))
                        in_use_len   = len(getattr(self, "_gpu_group_in_use", {}))
                else:
                    ring_len     = len(getattr(self, "_gpu_group_ring", []))
                    inflight_len = len(getattr(self, "_gpu_group_inflight", set()))
                    in_use_len   = len(getattr(self, "_gpu_group_in_use", {}))

                wm = self._pipeline_watermark
                wm["max_gpu_ring"]  = max(wm.get("max_gpu_ring", 0), ring_len)
                wm["max_inflight"]  = max(wm.get("max_inflight", 0), inflight_len)
                wm["max_in_use"]    = max(wm.get("max_in_use", 0), in_use_len)
            except Exception:
                pass

        ctx = prof.span("wsm.wait_group_ready", "wsm", **extras) if prof is not None else nullcontext()
        with ctx:
            return original_method(layer_idx, group, compute_stream)

    return wrapped



def _patched_ensure_module_on_gpu(self, m: torch.nn.Module, layer_idx: int | None = None, module_name: str | None = None):
    """
    扩展：把 **0-size CPU stub** 当作 meta 一样处理，优先从 CPU cache 取回并上卡。
    其它情况仍复用原先的 _ensure_param_on_gpu() 路径。
    """
    with (PROFILER.span("wsm.ensure_module_on_gpu", "wsm", layer_idx=(None if layer_idx is None else int(layer_idx)), module=str(module_name))
          if (globals().get("PROFILER") is not None) else nullcontext()):
        params_to_replace = {}
        params_full_names = {}

        def _full_name(layer_idx: int, module_name: str, local_param_name: str) -> str:
            if module_name in ("wq", "wk", "wv", "wo"):
                parent = "attention"
            elif module_name in ("w1", "w2", "w3"):
                parent = "feed_forward"
            else:
                parent = module_name or ""
            return f"layers.{layer_idx}.{parent}.{module_name}.{local_param_name}" if parent else f"layers.{layer_idx}.{module_name}.{local_param_name}"

        def _fetch_from_cpu_cache(name: str):
            if (layer_idx is not None) and (layer_idx in self.cpu_cache):
                return self.cpu_cache[layer_idx].get(name)
            return None

        for local_param_name, p in m.named_parameters(recurse=False):
            full_name = None
            if (layer_idx is not None) and (module_name is not None):
                full_name = _full_name(layer_idx, module_name, local_param_name)

            is_meta     = (p.device.type == "meta") or getattr(p, "is_meta", False)
            is_cpu_stub = (p.device.type == "cpu")  and (p.numel() == 0)

            if (is_meta or is_cpu_stub) and self.ssd_enabled and full_name:
                # 确保本层已有 CPU cache（没有就立即加载）
                if (layer_idx not in self.cpu_cache):
                    try:
                        self._load_layer_to_cpu(int(layer_idx))
                    except Exception:
                        pass

                cached = _fetch_from_cpu_cache(full_name)
                expected = tuple(getattr(getattr(m, local_param_name), "shape", ()))
                chosen_name, chosen_tensor = None, None

                def _try_pick(names: list[str]):
                    nonlocal chosen_name, chosen_tensor
                    for nm in names:
                        t = _fetch_from_cpu_cache(nm)
                        if t is not None and (not expected or tuple(t.shape) == expected):
                            chosen_name, chosen_tensor = nm, t
                            break

                if cached is not None and (not expected or tuple(cached.shape) == expected):
                    chosen_name, chosen_tensor = full_name, cached
                else:
                    cand = []
                    if module_name in ("wq", "wk", "wv"):
                        cand = [f"layers.{layer_idx}.attention.{x}.{local_param_name}" for x in ("wq","wk","wv")]
                    elif module_name in ("w1", "w2", "w3"):
                        cand = [f"layers.{layer_idx}.feed_forward.{x}.{local_param_name}" for x in ("w1","w2","w3")]
                    else:
                        cand = [full_name]
                    _try_pick(cand)
                    if chosen_tensor is None and cached is not None:
                        chosen_name, chosen_tensor = full_name, cached  # 退而求其次

                if chosen_tensor is not None:
                    with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
                        p_gpu = chosen_tensor.to(self.device, non_blocking=True)
                    params_to_replace[local_param_name] = torch.nn.Parameter(p_gpu, requires_grad=p.requires_grad)
                    params_full_names[local_param_name] = chosen_name or full_name
                    if getattr(self, "verbose", False):
                        print(f"[WSM DEBUG] ✓ Loaded {'meta' if is_meta else 'stub'} param {params_full_names[local_param_name]} to GPU: {tuple(p_gpu.shape)}")
                else:
                    if getattr(self, "verbose", False):
                        print(f"[WSM WARN] CPU cache miss for {full_name} (layer {layer_idx}); will rely on ensure_group_on_gpu() later")
                continue  # 该参数处理完毕

            # 其它情况：沿用原来的 CPU→GPU 逻辑
            self._ensure_param_on_gpu(p, layer_idx, full_name)

        # 安装替换后的 Parameter，并维护 name 映射
        for pname, new_param in params_to_replace.items():
            m._parameters[pname] = new_param
            full = params_full_names.get(pname)
            if full:
                try:
                    pobj = getattr(m, pname)
                except Exception:
                    pobj = new_param
                self.name_to_param[full] = pobj
                self.param_owner[full]   = (m, pname)

        # buffer 维持原有策略：meta→materialize，CPU→上卡
        for b in m.buffers(recurse=True):
            if getattr(b, "is_meta", False):
                try:
                    b = b.to_empty(device=self.device)
                except Exception:
                    pass
            elif b.device.type == "cpu":
                with torch.cuda.stream(self._select_h2d_stream_for(module_name=module_name)):
                    b_gpu = b.detach().to(self.device, non_blocking=True)
                try:
                    b.data = b_gpu
                except Exception:
                    pass
                
def _patch_wsm_for_profiling(wsm):
    """
    给 WeightStreamingManager 打补丁，补齐：
      - SSD -> pinned CPU 读层时间（含字节数）
      - pinned CPU -> GPU 组级 H2D 时间（含字节数、attn/ffn）
      - wait_group_ready 事件（保持原来统计）
    注意：只依赖 run 时的 wsm 实例，不改 wsm 源码。
    """
    global PROFILER
    prof = PROFILER
    if prof is None:
        # 没开 profiler 就不打点，保持零开销
        return

    # ---------- 1) wait_group_ready：保留你现有的等待统计 ----------
    if hasattr(wsm, "_record_group_ready_event"):
        orig_rg = wsm._record_group_ready_event

        def _record_group_ready_event_patched(self, layer_idx, group, *args, **kwargs):
            # 这里只记录 wait 自身的阻塞时间；更细的 slack 你之前已经在 _decode_timeline 里算了
            s = time.perf_counter_ns()
            try:
                return orig_rg(layer_idx, group, *args, **kwargs)
            finally:
                e = time.perf_counter_ns()
                rec = {
                    "name": "wsm.group_ready_event",
                    "cat": "wsm",
                    "t_start_ms": (s - prof.t0_ns) / 1e6,
                    "t_end_ms":   (e - prof.t0_ns) / 1e6,
                    "dur_ms":     (e - s) / 1e6,
                    "layer_idx":  int(layer_idx),
                    "group":      str(group),
                    "phase":      getattr(self, "_phase", None) or getattr(prof, "phase", "unknown"),
                }
                prof.timeline.append(rec)

        wsm._record_group_ready_event = types.MethodType(_record_group_ready_event_patched, wsm)

    # ---------- 2) SSD -> pinned CPU：按 layer 统计 ----------
    def _ssd_bytes_for_layer(self, layer_idx: int) -> int:
        try:
            params = self.layers_params.get(int(layer_idx), [])
        except Exception:
            return 0
        total = 0
        for p in params:
            try:
                if p.get("policy") == "stream":
                    total += int(p.get("nbytes", 0))
            except Exception:
                pass
        return int(total)

    if getattr(wsm, "ssd_enabled", False):
        # 同步版本（可能被 warmup 用到）
        if hasattr(wsm, "_read_layer_from_ssd"):
            orig_read = wsm._read_layer_from_ssd

            def _read_layer_from_ssd_patched(self, layer_idx: int):
                lid = int(layer_idx)
                total_bytes = _ssd_bytes_for_layer(self, lid)
                phase = getattr(self, "_phase", None) or "unknown"
                with prof.span_if_active(
                    "wsm.ssd_to_cpu_layer",
                    "io",
                    layer_idx=lid,
                    bytes=total_bytes,
                    phase=phase,
                    thread="main",
                ):
                    return orig_read(lid)

            wsm._read_layer_from_ssd = types.MethodType(_read_layer_from_ssd_patched, wsm)

        # 线程安全版本：真正的 CPU 预取线程走的是这个
        if hasattr(wsm, "_read_layer_from_ssd_threadsafe"):
            orig_read_ts = wsm._read_layer_from_ssd_threadsafe

            def _read_layer_from_ssd_threadsafe_patched(self, layer_idx: int):
                lid = int(layer_idx)
                total_bytes = _ssd_bytes_for_layer(self, lid)
                phase = getattr(self, "_phase", None) or "unknown"
                with prof.span_if_active(
                    "wsm.ssd_to_cpu_layer",
                    "io",
                    layer_idx=lid,
                    bytes=total_bytes,
                    phase=phase,
                    thread="cpu_pf_worker",
                ):
                    return orig_read_ts(lid)

            wsm._read_layer_from_ssd_threadsafe = types.MethodType(
                _read_layer_from_ssd_threadsafe_patched, wsm
            )

    # ---------- 3) pinned CPU -> GPU：组级 H2D ----------
    if hasattr(wsm, "_install_group_on_gpu"):
        orig_install = wsm._install_group_on_gpu

        def _install_group_on_gpu_patched(self, layer_idx: int, group: str, *, h2d_override=None):
            lid = int(layer_idx)
            grp = str(group)
            phase = getattr(self, "_phase", None) or "prefill"

            # 估算这次 H2D 的字节数：只看 CPU cache 里这层当前组的权重
            total_bytes = 0
            try:
                suffixes = ()
                if grp == "attn":
                    suffixes = (
                        "attention.wq.weight",
                        "attention.wk.weight",
                        "attention.wv.weight",
                        "attention.wo.weight",
                    )
                elif grp == "ffn":
                    suffixes = (
                        "feed_forward.w1.weight",
                        "feed_forward.w2.weight",
                        "feed_forward.w3.weight",
                    )

                if suffixes:
                    with self.cpu_cache_lock:
                        layer_data = dict(self.cpu_cache.get(lid, {}))

                    for suf in suffixes:
                        pname = f"layers.{lid}.{suf}"
                        t = layer_data.get(pname)
                        if t is not None and torch.is_tensor(t) and t.numel() > 0:
                            total_bytes += int(t.numel() * t.element_size())
            except Exception:
                total_bytes = 0

            with prof.span_if_active(
                "wsm.h2d_param",
                "io",
                layer_idx=lid,
                group=grp,
                bytes=int(total_bytes),
                phase=phase,
            ):
                # 关键：CPU wall time 只能测到 enqueue/调度开销，不能代表真实的 PCIe H2D 传输时间。
                # 这里额外用 CUDA events 在 **实际 H2D stream** 上打点，最后在 finalize 统一同步计算。
                if torch.cuda.is_available() and getattr(prof, "gpu_t0", None) is not None:
                    try:
                        # 选择与原实现一致的 H2D stream（不改变逻辑，仅显式拿到 stream 做 event record）
                        stream = h2d_override
                        if stream is None and hasattr(self, "_select_h2d_stream_for"):
                            stream = self._select_h2d_stream_for(module_name=grp)

                        if stream is not None:
                            s_ev = torch.cuda.Event(enable_timing=True)
                            e_ev = torch.cuda.Event(enable_timing=True)
                            with torch.cuda.stream(stream):
                                s_ev.record()

                            ret = orig_install(layer_idx, group, h2d_override=stream)

                            with torch.cuda.stream(stream):
                                e_ev.record()

                            prof.h2d_gpu_events.append({
                                "phase": phase,
                                "layer_idx": lid,
                                "group": grp,
                                "bytes": int(total_bytes),
                                "s_ev": s_ev,
                                "e_ev": e_ev,
                            })
                            return ret
                    except Exception:
                        # 出错就退回原逻辑（不影响正确性）
                        pass

                return orig_install(layer_idx, group, h2d_override=h2d_override)

        wsm._install_group_on_gpu = types.MethodType(_install_group_on_gpu_patched, wsm)

    # ---------- 4) 兼容老路径：_load_layer_to_cpu / _h2d_transfer_with_retry ----------
    # 这些在你新版 pipeline 中基本不会走到，但留着以防以后 fallback
    if hasattr(wsm, "_load_layer_to_cpu"):
        orig_load = wsm._load_layer_to_cpu

        def _load_layer_to_cpu_patched(self, layer_idx: int):
            lid = int(layer_idx)
            total_bytes = _ssd_bytes_for_layer(self, lid)
            phase = getattr(self, "_phase", None) or "unknown"
            with prof.span_if_active(
                "wsm.ssd_to_cpu_layer",
                "io",
                layer_idx=lid,
                bytes=total_bytes,
                phase=phase,
                thread="fallback_sync",
            ):
                return orig_load(lid)

        wsm._load_layer_to_cpu = types.MethodType(_load_layer_to_cpu_patched, wsm)

    if hasattr(wsm, "_h2d_transfer_with_retry"):
        orig_h2d = wsm._h2d_transfer_with_retry

        def _h2d_transfer_with_retry_patched(self, src_cpu_tensor, param_name, h2d_stream):
            bytes_ = 0
            try:
                if torch.is_tensor(src_cpu_tensor) and src_cpu_tensor.numel() > 0:
                    bytes_ = int(src_cpu_tensor.numel() * src_cpu_tensor.element_size())
            except Exception:
                bytes_ = 0

            pname = str(param_name)
            if "attention." in pname:
                grp = "attn"
            elif "feed_forward." in pname:
                grp = "ffn"
            else:
                grp = "other"

            phase = getattr(self, "_phase", None) or "prefill"

            with prof.span_if_active(
                "wsm.h2d_param",
                "io",
                param=pname,
                group=grp,
                bytes=bytes_,
                phase=phase,
            ):
                return orig_h2d(src_cpu_tensor, param_name, h2d_stream)

        wsm._h2d_transfer_with_retry = types.MethodType(
            _h2d_transfer_with_retry_patched, wsm
        )
        
        
def extract_kv_cache_stats(llama):
    """
    从 LLaMA wrapper 中提取 KV cache 统计信息（如果使用了 KVOffloader）。

    返回示例：
    {
        "fetch_blocks_total": 1234,
        "hits": 1200,
        "misses": 34,
        "hit_ratio": 0.9724,
        "ssd_load_blocks_prefetch": 56,
        "evictions": 789,
    }
    """
    try:
        model = getattr(llama, "model", None)
        if model is None:
            return None

        # 1) 有些实现会把 offloader 挂在 model 上
        off = getattr(model, "kv_offloader", None)

        # 2) 否则从第一层 attention 上找 offloader
        if off is None and hasattr(model, "layers"):
            for blk in getattr(model, "layers", []):
                attn = getattr(blk, "attention", None)
                if attn is None:
                    continue
                off = getattr(attn, "offloader", None)
                if off is not None:
                    break

        if off is None or not hasattr(off, "get_cache_stats"):
            return None

        stats = off.get_cache_stats()
        if not isinstance(stats, dict):
            return None

        # 清洗成 JSON-friendly 的简单类型
        cleaned = {}
        for k, v in stats.items():
            if isinstance(v, (int, float)) or v is None:
                cleaned[k] = v
            else:
                try:
                    cleaned[k] = float(v)
                except Exception:
                    cleaned[k] = str(v)
        return cleaned

    except Exception as e:
        print(f"[KV][WARN] extract_kv_cache_stats() failed: {e}")
        return None

# ---------- 辅助：固定规则生成 JSON/CSV 路径 ----------
def _sanitize_for_filename(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9_.+-]", "-", s)

def build_output_paths(log_dir: Path, run_id: str, mode: str) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = _sanitize_for_filename(RUN_TAG)
    stem = f"{ts}_{run_id}_{mode}" if not tag else f"{ts}_{tag}_{run_id}_{mode}"
    json_path = log_dir / f"{stem}.json"
    csv_path  = log_dir / f"{stem}.csv"
    return json_path, csv_path

# ---------- 运行主流程 ----------
def main():
    global PROFILER
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_tag = RUN_TAG.strip() or None
    PROFILER = InferenceProfiler(run_name=base_tag)

    os.environ.setdefault("OMP_NUM_THREADS",  "8")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    GPU_AHEAD_LAYERS = 8
    GPU_MAX_GROUPS   = 12
    GPU_WARMUP_LAYERS = 10
    CPU_CACHE_LAYERS = 47# 
    DEFAULT_BATCH_SIZE  = int(os.getenv("PROMPT_BATCH", "64"))   # batch size
    DEFAULT_MAX_GEN_LEN = int(os.getenv("GEN_TOKENS", "32"))    # 生成 token 数

    os.environ.setdefault("WSM_H2D_BASE_CONCURRENCY",  "32")   
    os.environ.setdefault("WSM_H2D_PREFILL_MULT",      "4")  
    os.environ.setdefault("WSM_H2D_DECODE_MULT",       "2")  
    os.environ.setdefault("WSM_MAX_INFLIGHT_GROUPS",   "128")  
    os.environ.setdefault("WSM_H2D_GROUP_BACKLOG_MAX", "256")  

    # === 异步逐出机制 ===
    os.environ.setdefault("WSM_EVICT_QUEUE_SIZE",      "96")   # 逐出队列容量
    os.environ.setdefault("WSM_BG_WORKERS",            "8")    # 后台线程池

    # === GPU 窗口配置 ===
    os.environ.setdefault("WSM_GPU_MAX_GROUPS",        str(GPU_MAX_GROUPS))
    os.environ.setdefault("WSM_GPU_AHEAD_GROUPS",      str(GPU_AHEAD_LAYERS))

    # Group 级预取深度：prefill 保守，decode aggressive
    os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH_PREFILL",  "10")  
    os.environ.setdefault("WSM_GROUP_PREFETCH_DEPTH_DECODE",   "10") 
    os.environ.setdefault("WSM_GPU_AHEAD",             str(GPU_AHEAD_LAYERS))
    os.environ.setdefault("WSM_GPU_BEHIND",            "2")    

    # === 预取策略 ===
    os.environ.setdefault("WSM_BALANCE_PREFETCH",      "1")
    os.environ.setdefault("WSM_PAIR_AHEAD",            "2")
    os.environ.setdefault("WSM_KIND_AHEAD_CAP",        "2")
    os.environ.setdefault("WSM_EVICT_FINISHED",        "1")   
    os.environ.setdefault("WSM_CPU_EVICT_AFTER_USE",   "0")  

    # === 调试与监控 ===
    os.environ.setdefault("WSM_GRP_RETAIN_MS",         "0")
    os.environ.setdefault("WSM_SKIP_PRELOAD_WAIT",     "1")    
    os.environ.setdefault("WSM_DEBUG_PREFETCH",        "1")  
    os.environ.setdefault("WSM_VERBOSE_MISMATCH",      "0")   

    # === CPU 预取优化（RAM 可容纳 60 层） ===
    os.environ.setdefault("WSM_POOLED_CPU_READ",       "1")
    os.environ.setdefault("WSM_CPU_PF_WORKERS",        "12")   
    os.environ.setdefault("WSM_REBALANCE_SYNC",        "0")  

    # === SSD→CPU 流水线 ===
    os.environ.setdefault("WSM_CPU_PREFETCH_DISTANCE", str(CPU_CACHE_LAYERS))   
    os.environ.setdefault("WSM_SSD_CONCURRENCY",       "12") 

    # === Prefill 特定优化 ===
    os.environ.setdefault("PREFILL_CPU_LAYERS",        str(CPU_CACHE_LAYERS))   
    os.environ.setdefault("PREFILL_GPU_LAYERS",        str(GPU_WARMUP_LAYERS))  
    os.environ.setdefault("PREFILL_PREFETCH_DISTANCE", "10")   
    os.environ.setdefault("DECODE_PREFETCH_DISTANCE",  "4")   
    os.environ.setdefault("WSM_WARMUP_LAYERS_GPU",     str(GPU_WARMUP_LAYERS))  
    os.environ.setdefault("WSM_WRAPAROUND_WARMUP",     str(GPU_WARMUP_LAYERS))  
    
    # chunk & micro-batch 大小（与并发匹配）
    os.environ.setdefault("PREFILL_T_CHUNK", str(CHUNK_SIZE)) 
    os.environ.setdefault("FFN_MICRO_B", str(MIRCO_BATCH_SIZE)) 
    os.environ.setdefault("ATTN_MICRO_B", str(ATTN_MICRO_B)) 


    # ============================================================
    # CPU 窗口额外配置（复用上面的 CPU_CACHE_LAYERS）
    # ============================================================
    os.environ.setdefault("WSM_CPU_RING_MODE",     "1")
    os.environ.setdefault("WSM_CPU_RING_OFFSET",   "0")
    os.environ.setdefault("WSM_CPU_CACHE_LAYERS",  str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_CAP_LAYERS", str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_HWM_LAYERS", str(CPU_CACHE_LAYERS))
    os.environ.setdefault("WSM_CPU_CACHE_LWM_LAYERS", str(max(2, CPU_CACHE_LAYERS - 5)))
    os.environ.setdefault("WSM_CPU_BACK_MARGIN",   "1")
    os.environ.setdefault("WSM_KV_THROTTLE_THRESHOLD", "2")
    os.environ.setdefault("WSM_KV_THROTTLE_MS",        "16")

    # 配置总结（仅打印）
    print("=" * 80)
    print("🚀 异步滑动窗口 - RTX 5080 (16GB) + 125GB RAM 优化配置")
    print("=" * 80)
    print(f"GPU 预取深度:  {GPU_AHEAD_LAYERS} 组")
    print(f"GPU 组预算:    {GPU_MAX_GROUPS} 组 (最多 ~9GB)")
    print(f"CPU 缓存容量:  {CPU_CACHE_LAYERS} 层 (~79.5GB)")
    print("H2D 并发度:    Prefill 24 | Decode 16")
    print("异步逐出队列:  64 任务")
    print("后台线程池:    6 workers")
    print("CPU 预取线程:  10 workers")
    print("=" * 80)
    print("✅ 异步窗口特性: 逐出/预取/CPU推进 全部在后台线程执行")
    print("✅ 主线程窗口滑动延迟: <1ms (vs 同步模式 ~20ms)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with PROFILER.span("apply_runtime_overrides", "setup"):
        apply_runtime_overrides()
    with PROFILER.span("configure_kv_pool", "setup"):
        configure_kv_pool()
    with PROFILER.span("probe_after_runtime_clamp", "non_inference"):
        probe("after runtime clamp")

    PRIME_WINDOW = int(os.getenv("WSM_PRIME_WINDOW", "6"))  
    mode_config = {
        "raw_device": RAW_DEV,
        "ssd_manifest_path": MANIFEST,
        "max_cached_layers": CPU_CACHE_LAYERS,         
        "cpu_cache_layers": CPU_CACHE_LAYERS,         
        "warmup_layers": max(PRIME_WINDOW, GPU_AHEAD_LAYERS + 2),  
        "staging_mb": 64,
        "verbose": True,
        "gpu_max_groups": GPU_MAX_GROUPS,
    }

    # 3) 构建（meta + SSD 流式）
    with PROFILER.span("probe_before_build", "non_inference"):
        probe("before LLaMA.build")
    with PROFILER.span("LLaMA.build", "setup"):
        llama = LLaMA.build(
            checkpoints_dir=CKPT_DIR,
            load_model=False,           
            device=device,
            max_seq_len=4096,
            max_batch_size=64,
            topk_blk=8,
            mode="mixed",
            mode_config=mode_config
        )
        s = stream_mnt.get_streams("cuda:0")
        print("h2d_mha:", s.weight_h2d_mha)
        print("h2d_ffn:", s.weight_h2d_ffn)
        print("cmp_mha:", s.compute_mha)
        print("cmp_ffn:", s.compute_ffn)
        print("kv_h2d:", s.kv_h2d, "kv_d2h:", s.kv_d2h)
    with PROFILER.span("probe_after_build", "non_inference"):
        probe("after LLaMA.build")

    wsm = getattr(llama, "weight_streaming_manager", None)
    if wsm is not None:
        wsm._pipeline_watermark = {}
        _patch_wsm_for_profiling(wsm)
        original_wait = wsm.wait_group_ready
        wsm.wait_group_ready = types.MethodType(_wrap_wait_group_ready(original_wait), wsm)

        wsm._ensure_module_on_gpu = types.MethodType(_patched_ensure_module_on_gpu, wsm)
        print("[WSM PATCH] Profiler wrapper + CPU stub loader enabled")

        # GPU窗口预热（避免冷启动，前N层并行H2D）
        with PROFILER.span("gpu_window_warmup", "setup"):
            warmup_layers = GPU_WARMUP_LAYERS  #  使用配置的 12 层
            print(f"[WSM WARMUP] Preloading first {warmup_layers} layers to GPU...")
            for layer_idx in range(min(warmup_layers, wsm.n_layers)):
                try:
                    # 异步预取attn和ffn组（不阻塞，让H2D在后台并行）
                    wsm.prefetch_group_async(layer_idx, "attn", reason="warmup")
                    wsm.prefetch_group_async(layer_idx, "ffn", reason="warmup")
                except Exception as e:
                    print(f"[WSM WARMUP] Layer {layer_idx} prefetch failed: {e}")
            print(f"[WSM WARMUP] Warmup requests sent (async), first {warmup_layers} layers (24 groups) will be ready before inference")


    PROFILER.wrap_model_forward(llama.model)

    # 读取 prompt + 安全裁剪（max_gen_len=32）
    batch_size = DEFAULT_BATCH_SIZE
    max_gen_len = DEFAULT_MAX_GEN_LEN

    with PROFILER.span("read_prompt_file", "prompt"):
        try:
            prompt_path = PROMPT_TXT
            file_content = prompt_path.read_text(encoding="utf-8").strip()

            # 解析多个prompts（按 "===== PROMPT XXXX =====" 分隔）
            import re
            prompt_blocks = re.split(r'=====\s*PROMPT\s+\d+\s+.*?=====\s*\n', file_content)
            # 过滤空字符串
            prompt_blocks = [p.strip() for p in prompt_blocks if p.strip()]

            # 取前batch_size个prompts
            prompts = prompt_blocks[:batch_size]
            if len(prompts) < batch_size:
                # 如果prompts不足，重复最后一个prompt来填充
                print(f"Warning: Only {len(prompts)} prompts found, padding to {batch_size}")
                while len(prompts) < batch_size:
                    prompts.append(prompts[-1])

            print(f"Loaded {len(prompts)} prompts for batch_size={batch_size}")

        except Exception as e:
            raise RuntimeError(f"无法读取 {prompt_path}: {e}")

    with PROFILER.span("tokenize_and_clip", "prompt"):
        max_prompt_tokens = llama.args.max_seq_len - max_gen_len

        # 对每个prompt进行tokenize和裁剪
        clipped_prompts = []
        for prompt in prompts:
            tok = llama.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tok) > max_prompt_tokens:
                tok = tok[-max_prompt_tokens:]
                prompt = llama.tokenizer.decode(tok)
            clipped_prompts.append(prompt)

        prompts = clipped_prompts
        # 使用第一个prompt的token数作为统计（假设所有prompt长度相似）
        tokens_in_count = len(llama.tokenizer.encode(prompts[0], add_special_tokens=False))

    # 5) 真正推理（decode）
    with PROFILER.span("probe_before_infer", "non_inference"):
        probe("before inference (decode)")
    with PROFILER.inference_scope():  # 端到端推理时间
        out_tokens, out_texts = llama.text_completion(
            prompts=prompts,
            temperature=0.0,
            max_gen_len=max_gen_len,
            batch_size=batch_size,
        )
    with PROFILER.span("probe_after_infer", "non_inference"):
        probe("after inference (decode)")

    # ==== 统计 tokens_out ====
    def _count_output_tokens(out_tokens_obj):
        """
        统计输出 token 数。
        对于 batch 输出（list of lists），返回第一个样本的长度作为代表。
        实际总 token 数 = tokens_out × batch_size
        """
        try:
            if isinstance(out_tokens_obj, (list, tuple)):
                if len(out_tokens_obj) > 0 and isinstance(out_tokens_obj[0], (list, tuple)):
                    # 返回第一个样本的长度（假设所有样本长度相同或相近）
                    return len(out_tokens_obj[0])
                return len(out_tokens_obj)
            if torch.is_tensor(out_tokens_obj):
                return int(out_tokens_obj.numel())
        except Exception:
            pass
        return None

    tokens_out_count = _count_output_tokens(out_tokens)

    # ==== 汇总与保存 ====
    mode = classify_mode(llama)
    kv_stats = extract_kv_cache_stats(llama)

    # 新增：从 WSM 对象上采集一次运行期统计，传给 Profiler
    wsm_runtime = None
    if hasattr(llama, "weight_streaming_manager"):
        try:
            wsm = llama.weight_streaming_manager
            rt = {}

            # pipeline 水位（来自 _wrap_wait_group_ready / _patch_wsm_for_profiling）
            pipe = getattr(wsm, "_pipeline_watermark", None)
            if isinstance(pipe, dict):
                rt["pipeline"] = dict(pipe)

            # SSD backend 静态 / 运行状态
            if hasattr(wsm, "get_ssd_stats"):
                try:
                    rt["ssd"] = wsm.get_ssd_stats()
                except Exception:
                    pass

            # H2D timeout/retry 统计（如果实现了）
            if hasattr(wsm, "get_h2d_timeout_stats"):
                try:
                    rt["h2d"] = wsm.get_h2d_timeout_stats()
                except Exception:
                    pass

            if rt:
                wsm_runtime = rt
        except Exception:
            wsm_runtime = None

    PROFILER.finalize(
        tokens_in=tokens_in_count,
        tokens_out=tokens_out_count,
        extra_meta={"llama_mode": mode, "device_str": str(device), "batch_size": batch_size},
        kv_stats=kv_stats or None,
        wsm_runtime=wsm_runtime,
    )


    # 自动生成 JSON/CSV 路径并各保存一次
    json_path, csv_path = build_output_paths(LOG_DIR, PROFILER.run_id, mode)
    PROFILER.save(str(json_path))
    PROFILER.save(str(csv_path))
    print(f"[Profiler] JSON: {json_path}")
    print(f"[Profiler] CSV : {csv_path}")
    
    
    # 控制台摘要，方便直接抄到论文表格
    summary = PROFILER.result
    t = summary.get("timings", {})
    kv = summary.get("kv_cache", {})
    print(
        "\n[SUMMARY] e2e_ms={e2e}, warmup_ms={warmup}, "
        "ftl_ms={ftl}".format(
            e2e=t.get("e2e_ms"),
            warmup=t.get("warmup_total_ms"),
            ftl=t.get("first_token_latency_ms"),
        )
    )
    if kv:
        print(
            "[SUMMARY] KV cache: hits={hits}, misses={misses}, "
            "evictions={evictions}, hit_ratio={ratio}".format(
                hits=kv.get("hits"),
                misses=kv.get("misses"),
                evictions=kv.get("evictions"),
                ratio=kv.get("hit_ratio"),
            )
        )
    else:
        print("[SUMMARY] KV cache: <no stats found on llama / kv_cache object>")

    # ==== 输出生成文本（不影响计时）====
    print(f"\n========== Generation (batch_size={batch_size}, len={max_gen_len}) ==========")
    # 只显示前3个和最后1个，避免输出太长
    for i in [0, 1, 2, batch_size-1]:
        if i < len(out_texts):
            print(f"\n--- Batch {i} ---")
            print(out_texts[i][:200] + "..." if len(out_texts[i]) > 200 else out_texts[i])
    print("=========================================")

if __name__ == "__main__":
    main()
