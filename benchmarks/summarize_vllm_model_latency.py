#!/usr/bin/env python3
"""Validate and summarize paired Qwen vLLM model-latency trials."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from flashdec.benchmark import validate_vllm_cache_root


BACKENDS = ("vllm_triton_attn", "flashdec")
TARGET_CASE = "qwen_b8_i2048_o128"
GUARDRAIL_CASE = "qwen_b8_i128_o128"
TARGET_RATIO_LIMIT = 0.995
GUARDRAIL_RATIO_LIMIT = 1.02
MAX_RATIO_SPREAD = 0.03
MIN_PAIRED_TRIALS = 3
DATASET_GENERATION_PROTOCOL = (
    "sha256-indexed-u64be-mod-model-tokenizer-nonspecial-v2"
)
TIMING_SCOPE = (
    "wall-clock blocking LLM.generate call after full-length warmup; "
    "model load, engine startup, JIT/graph capture, and result hashing excluded"
)


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
        "tokenizer_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "compilation_mode",
        "vllm_cache_root",
        "flashdec_num_splits",
        "warmup_iters",
        "num_iters",
        "dataset_seed",
        "dataset_generation_protocol",
        "sampling_seed",
        "prompt_format",
        "skip_tokenizer_init",
        "sampling_n",
        "sampling_temperature",
        "sampling_min_tokens",
        "sampling_max_tokens",
        "sampling_ignore_eos",
        "sampling_detokenize",
        "timing_scope",
        "vllm_engine_multiprocessing",
        "case",
        "batch_size",
        "input_len",
        "output_len",
        "backend",
        "trial",
        "dataset_path",
        "dataset_sha256",
        "output_token_ids_sha256",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "output_tokens_per_s",
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
        "tokenizer_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "compilation_mode",
        "vllm_cache_root",
        "flashdec_num_splits",
        "warmup_iters",
        "num_iters",
        "dataset_seed",
        "dataset_generation_protocol",
        "sampling_seed",
        "prompt_format",
        "skip_tokenizer_init",
        "sampling_n",
        "sampling_temperature",
        "sampling_ignore_eos",
        "sampling_detokenize",
        "timing_scope",
        "vllm_engine_multiprocessing",
    )
    first = rows[0]
    case_datasets: dict[str, tuple[str, ...]] = {}
    case_outputs: dict[str, str] = {}
    for row in rows:
        if any(row.get(field) != first.get(field) for field in invariant_fields):
            raise ValueError("environment/model/protocol invariants differ across rows")
        if row["schema_version"] != "3":
            raise ValueError("unsupported schema_version")
        validate_vllm_cache_root(row["vllm_cache_root"], row["git_commit"])
        if row["git_worktree_clean"] != "True":
            raise ValueError("formal model evidence requires a clean worktree")
        if row["backend"] not in BACKENDS:
            raise ValueError(f"unknown backend: {row['backend']}")
        if row["dataset_generation_protocol"] != DATASET_GENERATION_PROTOCOL:
            raise ValueError("unsupported deterministic dataset protocol")
        if row["prompt_format"] != "token_ids":
            raise ValueError("model latency evidence must use token-ID prompts")
        if row["skip_tokenizer_init"] != "True":
            raise ValueError("model latency evidence must skip tokenizer initialization")
        if (
            row["sampling_n"] != "1"
            or float(row["sampling_temperature"]) != 0.0
            or row["sampling_min_tokens"] != row["output_len"]
            or row["sampling_max_tokens"] != row["output_len"]
            or row["sampling_ignore_eos"] != "True"
            or row["sampling_detokenize"] != "False"
        ):
            raise ValueError("model latency evidence must use fixed greedy decoding")
        if row["timing_scope"] != TIMING_SCOPE:
            raise ValueError("unsupported model latency timing scope")
        if row["vllm_engine_multiprocessing"] != "True":
            raise ValueError("formal model latency requires vLLM engine multiprocessing")
        if not re.fullmatch(r"[0-9a-f]{64}", row["dataset_sha256"]):
            raise ValueError("invalid dataset SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", row["output_token_ids_sha256"]):
            raise ValueError("invalid generated-token SHA-256")
        if not Path(row["dataset_path"]).is_absolute():
            raise ValueError("dataset path must be absolute")
        dataset_invariants = (
            row["batch_size"],
            row["input_len"],
            row["output_len"],
            row["dataset_path"],
            row["dataset_sha256"],
        )
        previous = case_datasets.setdefault(row["case"], dataset_invariants)
        if previous != dataset_invariants:
            raise ValueError(
                "every backend/trial for a case must use the exact same dataset"
            )
        previous_output = case_outputs.setdefault(
            row["case"], row["output_token_ids_sha256"]
        )
        if previous_output != row["output_token_ids_sha256"]:
            raise ValueError(
                "every backend/trial for a case must generate the exact same tokens"
            )
        if min(
            float(row["avg_latency_ms"]),
            float(row["p50_latency_ms"]),
            float(row["p90_latency_ms"]),
        ) <= 0:
            raise ValueError("latencies must be positive")

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
    required_cases = {TARGET_CASE, GUARDRAIL_CASE}
    missing_cases = required_cases - set(by_case)
    if missing_cases:
        raise ValueError(f"missing required model cases: {sorted(missing_cases)}")
    if any(len(pairs) < MIN_PAIRED_TRIALS for pairs in by_case.values()):
        raise ValueError(
            f"formal model evidence requires at least {MIN_PAIRED_TRIALS} "
            "paired process trials per case"
        )

    results = []
    for case in sorted(by_case):
        pairs = by_case[case]
        native = statistics.median(
            float(pair["vllm_triton_attn"]["p50_latency_ms"])
            for pair in pairs
        )
        flashdec = statistics.median(
            float(pair["flashdec"]["p50_latency_ms"]) for pair in pairs
        )
        ratios = [
            float(pair["flashdec"]["p50_latency_ms"])
            / float(pair["vllm_triton_attn"]["p50_latency_ms"])
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
                "reduction_pct": (1.0 - flashdec / native) * 100.0,
                "throughput_uplift_pct": (native / flashdec - 1.0) * 100.0,
            }
        )

    by_name = {result["case"]: result for result in results}
    target_pass = by_name[TARGET_CASE]["ratio"] <= TARGET_RATIO_LIMIT
    guardrail_pass = (
        by_name[GUARDRAIL_CASE]["ratio"] <= GUARDRAIL_RATIO_LIMIT
    )
    stability_pass = all(
        result["ratio_max"] - result["ratio_min"] <= MAX_RATIO_SPREAD
        for result in results
    )
    gate_pass = target_pass and guardrail_pass and stability_pass
    geo_ratio = math.exp(
        statistics.mean(math.log(result["ratio"]) for result in results)
    )

    lines = [
        "# R7 Qwen2.5-3B vLLM Model Latency Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; paired process trials: {len(paired)}.",
        f"- Device: {first['device']}.",
        f"- Model: {first['model_id']} / {first['dtype']}.",
        f"- Model config SHA-256: `{first['model_config_sha256']}`.",
        (
            f"- Prompt dataset: fixed token IDs; seed `{first['dataset_seed']}`; "
            f"protocol `{first['dataset_generation_protocol']}`."
        ),
        (
            "- Decoding: greedy (`temperature=0`, `n=1`), fixed output length "
            "(`ignore_eos=True`), and detokenization disabled."
        ),
        (
            "- PyTorch / Triton / vLLM / PyTorch CUDA: "
            f"{first['torch_version']} / {first['triton_version']} / "
            f"{first['vllm_version']} / {first['torch_cuda']}."
        ),
        (
            f"- Compilation: `{first['compilation_mode']}`; FlashDec split policy: "
            f"`{first['flashdec_num_splits']}`."
        ),
        f"- Commit-scoped vLLM cache: `{first['vllm_cache_root']}`.",
        (
            f"- Per process: {first['warmup_iters']} warmup iterations and "
            f"{first['num_iters']} measured iterations."
        ),
        f"- Git commit: `{first['git_commit']}`; clean at start: True.",
        "- Per-case prompt dataset identities:",
        *(
            f"  - `{case}`: `{values[4]}` (`{values[3]}`)."
            for case, values in sorted(case_datasets.items())
        ),
        "- Per-case generated-token identities:",
        *(
            f"  - `{case}`: `{digest}`."
            for case, digest in sorted(case_outputs.items())
        ),
        "",
        "## Paired Results",
        "",
        (
            "Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. "
            "Latency is fixed-batch, end-to-end `LLM.generate` time with model "
            "loading and compilation excluded."
        ),
        "",
        (
            "| case | trials | vLLM p50 ms | FlashDec p50 ms | ratio "
            "[min,max] | latency reduction | output TPS uplift |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | {result['trials']} | {result['native']:.3f} | "
            f"{result['flashdec']:.3f} | {result['ratio']:.4f}x "
            f"[{result['ratio_min']:.4f},{result['ratio_max']:.4f}] | "
            f"{result['reduction_pct']:.2f}% | "
            f"{result['throughput_uplift_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen Confirmatory Performance Gate",
            "",
            (
                "These pilot-informed thresholds were frozen before the "
                "confirmatory three-trial run."
            ),
            (
                f"- B8 input2048/output128 target <= {TARGET_RATIO_LIMIT:.3f}x: "
                f"{'PASS' if target_pass else 'FAIL'}."
            ),
            (
                "- B8 input128/output128 guardrail <= "
                f"{GUARDRAIL_RATIO_LIMIT:.2f}x: "
                f"{'PASS' if guardrail_pass else 'FAIL'}."
            ),
            (
                f"- Every case paired-ratio spread <= {MAX_RATIO_SPREAD:.2f}: "
                f"{'PASS' if stability_pass else 'FAIL'}."
            ),
            f"- Geometric-mean p50 ratio: {geo_ratio:.4f}x.",
            f"- Overall external-model gate: **{'PASS' if gate_pass else 'FAIL'}**.",
            "",
            "## Boundary",
            "",
            (
                "This is an offline, fixed-batch vLLM model-latency comparison. "
                "It includes Qwen transformer execution, scheduling, KV-cache access, "
                "sampling, and Python API overhead, but excludes model startup/JIT and "
                "does not claim online TTFT/TPOT behavior."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if not gate_pass:
        raise ValueError("preregistered external-model performance gate failed")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.input, args.output))


if __name__ == "__main__":
    main()
