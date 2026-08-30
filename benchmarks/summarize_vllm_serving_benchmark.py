#!/usr/bin/env python3
"""Validate and summarize paired Qwen online-serving trials."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


BACKENDS = ("vllm_triton_attn", "flashdec")
REQUIRED_CASE = "qwen_c8_i4096_o128"
MIN_PAIRED_TRIALS = 3
TPOT_RATIO_LIMIT = 0.998
OUTPUT_THROUGHPUT_RATIO_MIN = 1.002
P90_TPOT_RATIO_LIMIT = 1.02
TTFT_RATIO_LIMIT = 1.05
MAX_TPOT_RATIO_SPREAD = 0.02


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")
    required = {
        "schema_version",
        "git_commit",
        "git_worktree_clean",
        "device",
        "torch_version",
        "torch_cuda",
        "triton_version",
        "vllm_version",
        "model_id",
        "model_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "compilation_mode",
        "flashdec_num_splits",
        "num_prompts",
        "num_warmups",
        "input_len",
        "output_len",
        "max_concurrency",
        "request_rate",
        "prefix_caching",
        "case",
        "backend",
        "trial",
        "completed",
        "failed",
        "median_ttft_ms",
        "p90_ttft_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "median_itl_ms",
        "p90_itl_ms",
        "median_e2el_ms",
        "p90_e2el_ms",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def summarize(input_path: Path, output_path: Path) -> str:
    rows = _read_rows(input_path)
    invariant_fields = (
        "schema_version",
        "git_commit",
        "git_worktree_clean",
        "device",
        "torch_version",
        "torch_cuda",
        "triton_version",
        "vllm_version",
        "model_id",
        "model_path",
        "model_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "compilation_mode",
        "flashdec_num_splits",
        "num_prompts",
        "num_warmups",
        "input_len",
        "output_len",
        "max_concurrency",
        "request_rate",
        "prefix_caching",
        "case",
    )
    first = rows[0]
    for row in rows:
        if any(row.get(field) != first.get(field) for field in invariant_fields):
            raise ValueError("environment/model/protocol invariants differ across rows")
        if row["schema_version"] != "1":
            raise ValueError("unsupported schema_version")
        if row["git_worktree_clean"] != "True":
            raise ValueError("formal serving evidence requires a clean worktree")
        if row["backend"] not in BACKENDS:
            raise ValueError(f"unknown backend: {row['backend']}")
        if row["case"] != REQUIRED_CASE:
            raise ValueError(f"formal serving case must be {REQUIRED_CASE}")
        if row["num_prompts"] != "128":
            raise ValueError("formal serving evidence requires 128 prompts")
        if int(row["completed"]) != 128 or int(row["failed"]) != 0:
            raise ValueError("every serving run must complete 128/128 requests")
        numeric_metrics = (
            "median_ttft_ms",
            "p90_ttft_ms",
            "median_tpot_ms",
            "p90_tpot_ms",
            "median_itl_ms",
            "p90_itl_ms",
            "median_e2el_ms",
            "p90_e2el_ms",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
        )
        if any(float(row[name]) <= 0 for name in numeric_metrics):
            raise ValueError("serving metrics must be positive")

    paired: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        trial = int(row["trial"])
        if row["backend"] in paired[trial]:
            raise ValueError(f"duplicate backend row for trial {trial}")
        paired[trial][row["backend"]] = row
    if any(set(pair) != set(BACKENDS) for pair in paired.values()):
        raise ValueError("every trial must contain both paired backends")
    if len(paired) < MIN_PAIRED_TRIALS:
        raise ValueError(
            f"formal serving evidence requires at least {MIN_PAIRED_TRIALS} "
            "paired process trials"
        )

    pairs = list(paired.values())

    def med(backend: str, metric: str) -> float:
        return statistics.median(float(pair[backend][metric]) for pair in pairs)

    def ratios(metric: str) -> list[float]:
        values = []
        for pair in pairs:
            native = float(pair["vllm_triton_attn"][metric])
            flashdec = float(pair["flashdec"][metric])
            values.append(flashdec / native)
        return values

    tpot_ratios = ratios("median_tpot_ms")
    p90_tpot_ratios = ratios("p90_tpot_ms")
    ttft_ratios = ratios("median_ttft_ms")
    throughput_ratios = ratios("output_throughput")
    tpot_ratio = statistics.median(tpot_ratios)
    p90_tpot_ratio = statistics.median(p90_tpot_ratios)
    ttft_ratio = statistics.median(ttft_ratios)
    throughput_ratio = statistics.median(throughput_ratios)

    tpot_pass = tpot_ratio <= TPOT_RATIO_LIMIT
    throughput_pass = throughput_ratio >= OUTPUT_THROUGHPUT_RATIO_MIN
    tail_pass = p90_tpot_ratio <= P90_TPOT_RATIO_LIMIT
    ttft_pass = ttft_ratio <= TTFT_RATIO_LIMIT
    stability_pass = (
        max(tpot_ratios) - min(tpot_ratios) <= MAX_TPOT_RATIO_SPREAD
    )
    gate_pass = all(
        (tpot_pass, throughput_pass, tail_pass, ttft_pass, stability_pass)
    )

    native_tpot = med("vllm_triton_attn", "median_tpot_ms")
    flashdec_tpot = med("flashdec", "median_tpot_ms")
    native_p90_tpot = med("vllm_triton_attn", "p90_tpot_ms")
    flashdec_p90_tpot = med("flashdec", "p90_tpot_ms")
    native_ttft = med("vllm_triton_attn", "median_ttft_ms")
    flashdec_ttft = med("flashdec", "median_ttft_ms")
    native_throughput = med("vllm_triton_attn", "output_throughput")
    flashdec_throughput = med("flashdec", "output_throughput")

    lines = [
        "# R7 Qwen2.5-3B vLLM Online Serving Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; paired server trials: {len(paired)}.",
        f"- Device: {first['device']}.",
        f"- Model: {first['model_id']} / {first['dtype']}.",
        f"- Model config SHA-256: `{first['model_config_sha256']}`.",
        (
            "- PyTorch / Triton / vLLM / PyTorch CUDA: "
            f"{first['torch_version']} / {first['triton_version']} / "
            f"{first['vllm_version']} / {first['torch_cuda']}."
        ),
        (
            f"- Workload: {first['num_prompts']} prompts, concurrency "
            f"{first['max_concurrency']}, input/output "
            f"{first['input_len']}/{first['output_len']}, request rate inf."
        ),
        (
            f"- Warmups: {first['num_warmups']}; prefix caching: "
            f"{first['prefix_caching']}; compilation: "
            f"`{first['compilation_mode']}`."
        ),
        f"- Git commit: `{first['git_commit']}`; clean at start: True.",
        "- Every run completed 128/128 requests with zero failures.",
        "",
        "## Paired Results",
        "",
        (
            "Latency ratios are `FlashDec/vLLM Triton`; throughput ratio is "
            "`FlashDec/vLLM Triton`. Values below 1 favor FlashDec latency, "
            "while values above 1 favor FlashDec throughput."
        ),
        "",
        "| metric | vLLM Triton | FlashDec | paired ratio [min,max] |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| median TPOT ms | {native_tpot:.4f} | {flashdec_tpot:.4f} | "
            f"{tpot_ratio:.4f}x [{min(tpot_ratios):.4f},{max(tpot_ratios):.4f}] |"
        ),
        (
            f"| p90 TPOT ms | {native_p90_tpot:.4f} | {flashdec_p90_tpot:.4f} | "
            f"{p90_tpot_ratio:.4f}x "
            f"[{min(p90_tpot_ratios):.4f},{max(p90_tpot_ratios):.4f}] |"
        ),
        (
            f"| median TTFT ms | {native_ttft:.3f} | {flashdec_ttft:.3f} | "
            f"{ttft_ratio:.4f}x [{min(ttft_ratios):.4f},{max(ttft_ratios):.4f}] |"
        ),
        (
            f"| output throughput tok/s | {native_throughput:.3f} | "
            f"{flashdec_throughput:.3f} | {throughput_ratio:.4f}x "
            f"[{min(throughput_ratios):.4f},{max(throughput_ratios):.4f}] |"
        ),
        "",
        "## Frozen Confirmatory Performance Gate",
        "",
        (
            "These pilot-informed thresholds were frozen before the "
            "confirmatory three-trial run."
        ),
        (
            f"- Median TPOT ratio <= {TPOT_RATIO_LIMIT:.3f}x: "
            f"{'PASS' if tpot_pass else 'FAIL'}."
        ),
        (
            f"- Output-throughput ratio >= {OUTPUT_THROUGHPUT_RATIO_MIN:.3f}x: "
            f"{'PASS' if throughput_pass else 'FAIL'}."
        ),
        (
            f"- p90 TPOT ratio <= {P90_TPOT_RATIO_LIMIT:.2f}x: "
            f"{'PASS' if tail_pass else 'FAIL'}."
        ),
        (
            f"- Median TTFT ratio <= {TTFT_RATIO_LIMIT:.2f}x: "
            f"{'PASS' if ttft_pass else 'FAIL'}."
        ),
        (
            f"- TPOT paired-ratio spread <= {MAX_TPOT_RATIO_SPREAD:.2f}: "
            f"{'PASS' if stability_pass else 'FAIL'}."
        ),
        f"- Overall external-serving gate: **{'PASS' if gate_pass else 'FAIL'}**.",
        "",
        "## Boundary",
        "",
        (
            "This is a saturated local HTTP serving comparison on one RTX 5070. "
            "It measures the same vLLM scheduler, API server, model, cache policy, "
            "and request stream; only the eligible single-token decode attention "
            "backend differs. It is not a multi-GPU or distributed-serving claim."
        ),
        "",
    ]
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if not gate_pass:
        raise ValueError("preregistered external-serving performance gate failed")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.input, args.output))


if __name__ == "__main__":
    main()
