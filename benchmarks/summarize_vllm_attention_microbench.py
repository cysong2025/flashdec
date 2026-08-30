#!/usr/bin/env python3
"""Validate and summarize paired FlashDec/vLLM attention microbenchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


BACKENDS = ("vllm_triton_attn", "flashdec")
REQUIRED_WIN_CASES = (
    "qwen_b1_ctx1024",
    "qwen_b4_ctx1024",
    "qwen_b8_ctx1024",
)
WIN_RATIO_LIMIT = 0.95
REGRESSION_RATIO_LIMIT = 1.25


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
        "dtype",
        "case",
        "backend",
        "trial",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "correctness",
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
        "dtype",
    )
    first = rows[0]
    for row in rows:
        if any(row[field] != first[field] for field in invariant_fields):
            raise ValueError("environment/model invariants differ across rows")
        if row["schema_version"] != "1":
            raise ValueError("unsupported schema_version")
        if row["backend"] not in BACKENDS:
            raise ValueError(f"unknown backend: {row['backend']}")
        if row["correctness"] != "PASS":
            raise ValueError("correctness must PASS for every row")
        if float(row["p50_ms"]) <= 0:
            raise ValueError("p50_ms must be positive")

    paired: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["case"], int(row["trial"]))
        if row["backend"] in paired[key]:
            raise ValueError(f"duplicate backend row for {key}")
        paired[key][row["backend"]] = row
    if any(set(pair) != set(BACKENDS) for pair in paired.values()):
        raise ValueError("every case/trial must contain both paired backends")

    by_case: dict[str, list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (case, _trial), pair in paired.items():
        by_case[case].append(pair)
    missing_win_cases = set(REQUIRED_WIN_CASES) - set(by_case)
    if missing_win_cases:
        raise ValueError(f"missing required Qwen cases: {sorted(missing_win_cases)}")

    results = []
    for case in sorted(by_case):
        pairs = by_case[case]
        native = statistics.median(
            float(pair["vllm_triton_attn"]["p50_ms"]) for pair in pairs
        )
        flashdec = statistics.median(
            float(pair["flashdec"]["p50_ms"]) for pair in pairs
        )
        ratios = [
            float(pair["flashdec"]["p50_ms"])
            / float(pair["vllm_triton_attn"]["p50_ms"])
            for pair in pairs
        ]
        results.append(
            {
                "case": case,
                "trials": len(pairs),
                "native": native,
                "flashdec": flashdec,
                "ratio": statistics.median(ratios),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
            }
        )

    by_name = {result["case"]: result for result in results}
    wins_pass = all(
        by_name[case]["ratio"] <= WIN_RATIO_LIMIT for case in REQUIRED_WIN_CASES
    )
    guardrail_pass = all(
        result["ratio"] <= REGRESSION_RATIO_LIMIT for result in results
    )
    gate_pass = wins_pass and guardrail_pass
    geo_ratio = math.exp(
        statistics.mean(math.log(result["ratio"]) for result in results)
    )

    lines = [
        "# R7 vLLM Qwen Attention Microbenchmark Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; paired trials: {len(paired)}.",
        f"- Device: {first['device']}.",
        f"- Model shape contract: {first['model_id']} / {first['dtype']}.",
        (
            "- PyTorch / Triton / vLLM / PyTorch CUDA: "
            f"{first['torch_version']} / {first['triton_version']} / "
            f"{first['vllm_version']} / {first['torch_cuda']}."
        ),
        f"- Git commit: `{first['git_commit']}`; clean at start: {first['git_worktree_clean']}.",
        "- Every pair passed full-output cross-backend correctness.",
        "",
        "## Paired Results",
        "",
        (
            "Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. "
            "Each cell is the median across paired trials."
        ),
        "",
        "| case | trials | vLLM Triton p50 ms | FlashDec p50 ms | ratio [min,max] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | {result['trials']} | {result['native']:.6f} | "
            f"{result['flashdec']:.6f} | {result['ratio']:.4f}x "
            f"[{result['ratio_min']:.4f},{result['ratio_max']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Preregistered Performance Gate",
            "",
            (
                f"- Required ctx1024 Qwen cases <= {WIN_RATIO_LIMIT:.2f}x: "
                f"{'PASS' if wins_pass else 'FAIL'}."
            ),
            (
                f"- Every measured case <= {REGRESSION_RATIO_LIMIT:.2f}x guardrail: "
                f"{'PASS' if guardrail_pass else 'FAIL'}."
            ),
            f"- Geometric-mean p50 ratio across cases: {geo_ratio:.4f}x.",
            f"- Overall external-kernel gate: **{'PASS' if gate_pass else 'FAIL'}**.",
            "",
            "## Boundary",
            "",
            (
                "This gate compares only single-token decode attention inside the same "
                "vLLM KV layout and metadata contract. It is necessary but not sufficient "
                "for a model- or serving-level performance claim; those are measured "
                "separately with Qwen2.5-3B."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if not gate_pass:
        raise ValueError("preregistered external-kernel performance gate failed")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = summarize(args.input, args.output)
    print(text)


if __name__ == "__main__":
    main()
