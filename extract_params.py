#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 inference profiler 的 JSON/CSV 中抽取论文用的关键指标。

用法示例：
    python extract_paper_metrics.py 20251126-150912_run-1b685d46_ssd-streaming.json
    python extract_paper_metrics.py 20251126-150912_run-1b685d46_ssd-streaming.csv --out paper_table.csv

如果传入的是 CSV，脚本会尝试在同目录下找到同名 JSON（后缀改为 .json）。
推荐：总是用 JSON 作为主输入，这样能拿到最完整的统计（prefill / decode / overlap）。
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _percentile(xs: List[float], q: float) -> float:
    """简单百分位实现，q 取 0~100。"""
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    pos = (len(xs_sorted) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs_sorted[lo]
    frac = pos - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 标准时间字段转成 float，方便后面可能的复用
            for k in ("t_start_ms", "t_end_ms", "dur_ms"):
                v = row.get(k, "")
                if v == "" or v is None:
                    row[k] = None
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def guess_decode_steps_from_csv(rows: List[Dict[str, Any]]) -> List[float]:
    """从 CSV 中提取 decode 步长（如果没有 JSON 的 decode_step_ms）。"""
    step_ms: List[float] = []
    for r in rows:
        if r.get("kind") == "decode_step":
            d = r.get("dur_ms")
            if d is not None:
                step_ms.append(float(d))
    return step_ms


def _summarize_from_data(data: Dict[str, Any],
                         csv_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """核心汇总逻辑：给定已经 load 好的 JSON dict + 可选 CSV 行，返回一坨论文用指标。"""
    result: Dict[str, Any] = {}

    run_meta = data.get("run", {})
    counts = data.get("counts", {})
    timings = data.get("timings", {})
    throughput = data.get("throughput", {})

    # ===== 基本信息 =====
    result["run_id"] = run_meta.get("run_id", "")
    result["mode"] = run_meta.get("llama_mode", "")
    result["device"] = run_meta.get("cuda_device_name", run_meta.get("device_str", ""))
    result["batch_size"] = run_meta.get("batch_size", None)

    # token 相关
    result["tokens_in"] = counts.get("tokens_in")
    result["tokens_out"] = counts.get("tokens_out")
    result["tokens_in_total"] = counts.get("tokens_in_total")
    result["tokens_out_total"] = counts.get("tokens_out_total")

    # ===== 时延整体指标 =====
    result["e2e_ms"] = timings.get("e2e_ms", timings.get("inference_e2e_ms"))
    result["te2e_ms"] = timings.get("te2e_ms")
    result["prefill_total_ms"] = timings.get("prefill_total_ms")
    result["ttft_ms"] = timings.get("ttft_ms") or timings.get("first_token_latency_ms")

    # decode / TBT 统计
    decode_stats = timings.get("decode") or timings.get("decode_stats") or {}
    tbt_stats = timings.get("tbt_steady_ms") or {}

    result["decode_total_ms"] = decode_stats.get("sum_ms")
    result["decode_step_count"] = decode_stats.get("count")
    result["decode_step_mean_ms"] = decode_stats.get("mean_ms")
    result["decode_step_p50_ms"] = decode_stats.get("p50_ms")
    result["decode_step_p90_ms"] = decode_stats.get("p90_ms")
    result["decode_step_p99_ms"] = decode_stats.get("p99_ms")
    result["decode_step_p999_ms"] = decode_stats.get("p999_ms")
    result["decode_step_max_ms"] = decode_stats.get("max_ms")

    # TBT（去掉首 token 之后）
    result["tbt_p50_ms"] = tbt_stats.get("p50_ms")
    result["tbt_p90_ms"] = tbt_stats.get("p90_ms")
    result["tbt_p99_ms"] = tbt_stats.get("p99_ms")

    # 吞吐
    result["prefill_toks_per_s"] = throughput.get("prefill_toks_per_s")
    result["decode_toks_per_s"] = throughput.get("decode_toks_per_s")

    # ===== by-category 时间分解 =====
    by_cat = (timings.get("by_category_ms") or {}).get("by_cat_ms", {})
    result["setup_ms"] = by_cat.get("setup")
    result["prompt_ms"] = by_cat.get("prompt")
    result["io_ms"] = by_cat.get("io")
    result["wsm_ms"] = by_cat.get("wsm")
    result["inference_ms"] = by_cat.get("inference")
    result["non_inference_ms"] = by_cat.get("non_inference")

    # ===== decoder 端 IO / compute / overlap =====
    dec_layers = data.get("decoder_layers") or {}
    layers_summary = dec_layers.get("summary") or {}
    per_layer = dec_layers.get("per_layer") or {}
    global_stats = dec_layers.get("global") or {}

    # 来自 profiler 中的 summary（注意：目前是 prefill+decode 总和的近似）
    result["decode_compute_ms_total"] = layers_summary.get("compute_ms_total")
    result["decode_io_ms_total"] = layers_summary.get("io_ms_total")
    result["decode_overlap_ratio"] = layers_summary.get("overlap_ratio")
    result["decode_overlap_ms_approx"] = layers_summary.get("overlap_ms_approx")
    result["decode_uncovered_io_ms_approx"] = layers_summary.get("uncovered_io_ms_approx")
    result["decode_wall_ms"] = layers_summary.get("decode_wall_ms")

    # 如果 timings["decode"] 里还有 approximate 的信息，也顺便带上
    for k in (
        "approx_compute_ms_total",
        "approx_io_ms_total",
        "approx_overlap_ms",
        "approx_uncovered_io_ms",
        "overlap_ratio",
    ):
        if k in decode_stats and f"decode_{k}" not in result:
            result[f"decode_{k}"] = decode_stats.get(k)

    # 归一化成 compute / IO 的占比
    comp = result.get("decode_compute_ms_total")
    io_tot = result.get("decode_io_ms_total")
    if comp is not None and io_tot is not None:
        denom = comp + io_tot
        if denom > 0:
            result["decode_compute_share"] = comp / denom
            result["decode_io_share"] = io_tot / denom

    # per-layer breakdown：SSD / H2D / wait_group + 带宽
    if per_layer:
        ssd_ms = h2d_ms = wait_ms = 0.0
        ssd_bytes = h2d_bytes = 0
        for st in per_layer.values():
            io_ms = st.get("io_ms") or {}
            ssd_ms += float(io_ms.get("ssd_to_cpu_ms") or 0.0)
            h2d_ms += float(io_ms.get("h2d_param_ms") or 0.0)
            wait_ms += float(io_ms.get("wait_group_ready_ms") or 0.0)
            io_bytes = st.get("io_bytes") or {}
            ssd_bytes += int(io_bytes.get("ssd_to_cpu_bytes") or 0)
            h2d_bytes += int(io_bytes.get("h2d_param_bytes") or 0)

        result["decode_ssd_to_cpu_ms_total"] = ssd_ms
        result["decode_h2d_param_ms_total"] = h2d_ms
        result["decode_wait_group_ready_ms_total"] = wait_ms
        result["ssd_to_cpu_bytes_total"] = ssd_bytes or None
        result["h2d_param_bytes_total"] = h2d_bytes or None

        if ssd_ms > 0 and ssd_bytes > 0:
            result["ssd_to_cpu_effective_GBps"] = (ssd_bytes / 1e9) / (ssd_ms / 1000.0)
        if h2d_ms > 0 and h2d_bytes > 0:
            result["h2d_param_effective_GBps"] = (h2d_bytes / 1e9) / (h2d_ms / 1000.0)

    # 全局 compute breakdown：attn / ffn / kv_fetch
    if global_stats:
        def _us_to_ms(x: Optional[float]) -> Optional[float]:
            return (float(x) / 1000.0) if x is not None else None

        attn_us = global_stats.get("attn_us")
        ffn_us = global_stats.get("ffn_us")
        kv_us = global_stats.get("kv_fetch_us")
        total_us = global_stats.get("total_forward_us")

        result["decoder_global_attn_ms"] = _us_to_ms(attn_us)
        result["decoder_global_ffn_ms"] = _us_to_ms(ffn_us)
        result["decoder_global_kv_fetch_ms"] = _us_to_ms(kv_us)
        result["decoder_global_total_forward_ms"] = _us_to_ms(total_us)

        parts = [x for x in (attn_us, ffn_us, kv_us) if x is not None]
        denom_us = float(sum(parts)) if parts else 0.0
        if denom_us > 0:
            result["decoder_global_attn_share"] = float(attn_us or 0.0) / denom_us
            result["decoder_global_ffn_share"] = float(ffn_us or 0.0) / denom_us
            result["decoder_global_kv_fetch_share"] = float(kv_us or 0.0) / denom_us

    # ===== WSM / IO（prefill / decode phase 粗略统计）=====
    wsm = data.get("wsm") or {}
    wsm_io = wsm.get("io") or {}
    # 兼容两种 key 命名：ssd_to_cpu_ms / ssd_to_cpu_layer, h2d_param_ms / h2d_param
    ssd_entry = wsm_io.get("ssd_to_cpu_ms") or wsm_io.get("ssd_to_cpu_layer") or {}
    h2d_entry = wsm_io.get("h2d_param_ms") or wsm_io.get("h2d_param") or {}

    result["ssd_to_cpu_ms_total"] = ssd_entry.get("total_ms")
    result["h2d_param_ms_total"] = h2d_entry.get("total_ms")

    by_phase_ssd = ssd_entry.get("by_group") or {}
    by_phase_h2d = h2d_entry.get("by_group") or {}

    # 如果你的代码里按 phase 标了 prefill / decode，这里就能拆开；否则这些字段可能为 None
    result["prefill_ssd_to_cpu_ms"] = by_phase_ssd.get("prefill")
    result["decode_ssd_to_cpu_ms"] = by_phase_ssd.get("decode")
    result["prefill_h2d_param_ms"] = by_phase_h2d.get("prefill")
    result["decode_h2d_param_ms"] = by_phase_h2d.get("decode")

    if result.get("prefill_ssd_to_cpu_ms") is not None or result.get("prefill_h2d_param_ms") is not None:
        result["prefill_io_ms_from_wsm"] = float(result.get("prefill_ssd_to_cpu_ms") or 0.0) + float(result.get("prefill_h2d_param_ms") or 0.0)
    if result.get("decode_ssd_to_cpu_ms") is not None or result.get("decode_h2d_param_ms") is not None:
        result["decode_io_ms_from_wsm"] = float(result.get("decode_ssd_to_cpu_ms") or 0.0) + float(result.get("decode_h2d_param_ms") or 0.0)

    # ===== 一些归一化指标：ms/token, share, overall throughput =====
    tokens_in_total = result.get("tokens_in_total")
    if tokens_in_total is None and result.get("tokens_in") is not None and result.get("batch_size") is not None:
        tokens_in_total = int(result["tokens_in"]) * int(result["batch_size"])
        result["tokens_in_total"] = tokens_in_total

    tokens_out_total = result.get("tokens_out_total")
    if tokens_out_total is None and result.get("tokens_out") is not None and result.get("batch_size") is not None:
        tokens_out_total = int(result["tokens_out"]) * int(result["batch_size"])
        result["tokens_out_total"] = tokens_out_total

    pf_ms = result.get("prefill_total_ms")
    dec_ms = result.get("decode_total_ms")
    if pf_ms is not None and tokens_in_total:
        result["prefill_ms_per_input_token"] = pf_ms / float(tokens_in_total)
    if dec_ms is not None and tokens_out_total:
        result["decode_ms_per_output_token"] = dec_ms / float(tokens_out_total)

    if pf_ms is not None or dec_ms is not None:
        pf_val = float(pf_ms or 0.0)
        dec_val = float(dec_ms or 0.0)
        denom_inf = pf_val + dec_val
        if denom_inf > 0:
            result["prefill_share_of_inference"] = pf_val / denom_inf
            result["decode_share_of_inference"] = dec_val / denom_inf

    if tokens_in_total and tokens_out_total and result.get("e2e_ms"):
        total_tokens = float(tokens_in_total + tokens_out_total)
        result["total_tokens"] = int(total_tokens)
        e2e = float(result["e2e_ms"])
        if e2e > 0:
            result["total_toks_per_s_over_e2e"] = total_tokens / (e2e / 1000.0)

    # ===== 如果 JSON 里没有 decode_step 统计，尝试从 CSV 重算 =====
    if not decode_stats and csv_rows is not None:
        step_ms = guess_decode_steps_from_csv(csv_rows)
        if step_ms:
            result["decode_step_count"] = len(step_ms)
            result["decode_step_mean_ms"] = sum(step_ms) / len(step_ms)
            result["decode_step_p50_ms"] = _percentile(step_ms, 50)
            result["decode_step_p90_ms"] = _percentile(step_ms, 90)
            result["decode_step_p99_ms"] = _percentile(step_ms, 99)
            result["decode_step_max_ms"] = max(step_ms)
            result["decode_total_ms"] = sum(step_ms)

    return result


def summarize_one_run(json_path: Optional[Path],
                      csv_path: Optional[Path]) -> Dict[str, Any]:
    """从 JSON + CSV 文件路径中读入，再调用上面的核心逻辑。"""
    data: Dict[str, Any] = {}
    if json_path is not None and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    csv_rows: Optional[List[Dict[str, Any]]] = None
    if csv_path is not None and csv_path.exists():
        csv_rows = load_csv_rows(csv_path)

    return _summarize_from_data(data, csv_rows)


def print_human_readable(summary: Dict[str, Any]) -> None:
    """人类可读的打印（终端看用）。"""
    print("==== Run summary ====")
    print(f"run_id      : {summary.get('run_id')}")
    print(f"mode        : {summary.get('mode')}")
    print(f"device      : {summary.get('device')}")
    print(f"batch_size  : {summary.get('batch_size')}")
    print(f"tokens_in   : {summary.get('tokens_in')} "
          f"(total {summary.get('tokens_in_total')})")
    print(f"tokens_out  : {summary.get('tokens_out')} "
          f"(total {summary.get('tokens_out_total')})")
    print()

    print("== Latency ==")
    print(f"e2e_ms      : {summary.get('e2e_ms')}")
    print(f"TTFT_ms     : {summary.get('ttft_ms')}")
    print(f"prefill_ms  : {summary.get('prefill_total_ms')} "
          f"(per in tok: {summary.get('prefill_ms_per_input_token')})")
    print(f"decode_ms   : {summary.get('decode_total_ms')} "
          f"(per out tok: {summary.get('decode_ms_per_output_token')})")
    print(f"prefill/decode share (of inference): "
          f"{summary.get('prefill_share_of_inference')}, "
          f"{summary.get('decode_share_of_inference')}")
    print(f"TBT p50/p90/p99 ms : "
          f"{summary.get('tbt_p50_ms')}, "
          f"{summary.get('tbt_p90_ms')}, "
          f"{summary.get('tbt_p99_ms')}")
    print()

    print("== Throughput ==")
    print(f"prefill toks/s     : {summary.get('prefill_toks_per_s')}")
    print(f"decode  toks/s     : {summary.get('decode_toks_per_s')}")
    print(f"total  toks/s (e2e): {summary.get('total_toks_per_s_over_e2e')}")
    print()

    print("== Decode per-step latency (ms) ==")
    print(f"count      : {summary.get('decode_step_count')}")
    print(f"mean       : {summary.get('decode_step_mean_ms')}")
    print(f"p50/p90/p99: {summary.get('decode_step_p50_ms')}, "
          f"{summary.get('decode_step_p90_ms')}, "
          f"{summary.get('decode_step_p99_ms')}")
    print(f"max        : {summary.get('decode_step_max_ms')}")
    print()

    print("== Decode IO/compute overlap ==")
    print(f"wall_ms (decode)   : {summary.get('decode_wall_ms')}")
    print(f"compute_ms_total   : {summary.get('decode_compute_ms_total')}")
    print(f"io_ms_total        : {summary.get('decode_io_ms_total')}")
    print(f"compute/io share   : {summary.get('decode_compute_share')}, "
          f"{summary.get('decode_io_share')}")
    print(f"overlap_ratio      : {summary.get('decode_overlap_ratio')}")
    print(f"approx_overlap_ms  : {summary.get('decode_overlap_ms_approx')}")
    print(f"uncovered_io_ms    : {summary.get('decode_uncovered_io_ms_approx')}")
    print()

    print("== IO breakdown (all decoder layers) ==")
    print(f"ssd_ms / h2d_ms / wait_ms : "
          f"{summary.get('decode_ssd_to_cpu_ms_total')}, "
          f"{summary.get('decode_h2d_param_ms_total')}, "
          f"{summary.get('decode_wait_group_ready_ms_total')}")
    print(f"ssd_bytes / h2d_bytes    : "
          f"{summary.get('ssd_to_cpu_bytes_total')}, "
          f"{summary.get('h2d_param_bytes_total')}")
    print(f"ssd_GBps / h2d_GBps      : "
          f"{summary.get('ssd_to_cpu_effective_GBps')}, "
          f"{summary.get('h2d_param_effective_GBps')}")
    print()

    print("== Global compute breakdown (attn / ffn / kv) ==")
    print(f"attn_ms / ffn_ms / kv_ms : "
          f"{summary.get('decoder_global_attn_ms')}, "
          f"{summary.get('decoder_global_ffn_ms')}, "
          f"{summary.get('decoder_global_kv_fetch_ms')}")
    print(f"shares (attn/ffn/kv)     : "
          f"{summary.get('decoder_global_attn_share')}, "
          f"{summary.get('decoder_global_ffn_share')}, "
          f"{summary.get('decoder_global_kv_fetch_share')}")
    print()

    print("== WSM IO (prefill/decode, if available) ==")
    print(f"ssd_to_cpu_ms_total : {summary.get('ssd_to_cpu_ms_total')}")
    print(f"h2d_param_ms_total  : {summary.get('h2d_param_ms_total')}")
    print(f"prefill ssd/h2d ms  : {summary.get('prefill_ssd_to_cpu_ms')}, "
          f"{summary.get('prefill_h2d_param_ms')}")
    print(f"decode  ssd/h2d ms  : {summary.get('decode_ssd_to_cpu_ms')}, "
          f"{summary.get('decode_h2d_param_ms')}")
    print()

    print("== By category (ms) ==")
    for k in ("setup_ms", "prompt_ms", "io_ms",
              "wsm_ms", "inference_ms", "non_inference_ms"):
        print(f"{k:14s}: {summary.get(k)}")
    print()


def write_one_row_to_csv(summary: Dict[str, Any], out_path: Path) -> None:
    """把 summary 作为一行写到 CSV（如果文件不存在就创建并写 header）。"""
    # 定义我们希望出现在表格里的列顺序（适合直接丢进论文里的大表）
    field_order = [
        # 基本信息
        "run_id", "mode", "device", "batch_size",
        # token 统计
        "tokens_in", "tokens_out", "tokens_in_total", "tokens_out_total", "total_tokens",
        # 时延 + 吞吐
        "e2e_ms", "ttft_ms", "prefill_total_ms", "decode_total_ms",
        "prefill_share_of_inference", "decode_share_of_inference",
        "prefill_toks_per_s", "decode_toks_per_s", "total_toks_per_s_over_e2e",
        "prefill_ms_per_input_token", "decode_ms_per_output_token",
        # per-step / TBT
        "decode_step_count", "decode_step_mean_ms",
        "decode_step_p50_ms", "decode_step_p90_ms",
        "decode_step_p99_ms", "decode_step_max_ms",
        "tbt_p50_ms", "tbt_p90_ms", "tbt_p99_ms",
        # decode 端 IO+compute
        "decode_wall_ms",
        "decode_compute_ms_total", "decode_io_ms_total",
        "decode_compute_share", "decode_io_share",
        "decode_overlap_ratio", "decode_overlap_ms_approx",
        "decode_uncovered_io_ms_approx",
        # per-layer IO breakdown + 带宽
        "decode_ssd_to_cpu_ms_total", "decode_h2d_param_ms_total",
        "decode_wait_group_ready_ms_total",
        "ssd_to_cpu_bytes_total", "h2d_param_bytes_total",
        "ssd_to_cpu_effective_GBps", "h2d_param_effective_GBps",
        # WSM 总 IO + phase breakdown
        "ssd_to_cpu_ms_total", "h2d_param_ms_total",
        "prefill_ssd_to_cpu_ms", "prefill_h2d_param_ms",
        "prefill_io_ms_from_wsm",
        "decode_ssd_to_cpu_ms", "decode_h2d_param_ms",
        "decode_io_ms_from_wsm",
        # 全局 compute breakdown
        "decoder_global_attn_ms", "decoder_global_ffn_ms",
        "decoder_global_kv_fetch_ms", "decoder_global_total_forward_ms",
        "decoder_global_attn_share", "decoder_global_ffn_share",
        "decoder_global_kv_fetch_share",
        # by-category
        "setup_ms", "prompt_ms", "io_ms",
        "wsm_ms", "inference_ms", "non_inference_ms",
    ]

    out_exists = out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not out_exists:
            writer.writeheader()
        row = {k: summary.get(k, "") for k in field_order}
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="输入的 JSON 或 CSV 文件路径")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可选：把汇总结果追加写入这个 CSV（适合直接丢进 Excel / LaTeX）",
    )
    args = parser.parse_args()

    in_path = Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None

    if in_path.suffix.lower() == ".json":
        json_path = in_path
        candidate_csv = in_path.with_suffix(".csv")
        if candidate_csv.exists():
            csv_path = candidate_csv
    elif in_path.suffix.lower() == ".csv":
        csv_path = in_path
        candidate_json = in_path.with_suffix(".json")
        if candidate_json.exists():
            json_path = candidate_json
    else:
        # 非常规后缀，就都尝试一下
        if in_path.with_suffix(".json").exists():
            json_path = in_path.with_suffix(".json")
        if in_path.with_suffix(".csv").exists():
            csv_path = in_path.with_suffix(".csv")

    if json_path is None:
        print("警告：找不到同名 JSON，只能从 CSV 中做有限统计。")
    else:
        print(f"Using JSON: {json_path}")
    if csv_path is not None:
        print(f"Using CSV : {csv_path}")

    summary = summarize_one_run(json_path, csv_path)
    print_human_readable(summary)

    if args.out:
        out_path = Path(args.out)
        write_one_row_to_csv(summary, out_path)
        print(f"\n已将该 run 的关键指标写入: {out_path}")


if __name__ == "__main__":
    main()
