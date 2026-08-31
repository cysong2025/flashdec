#!/usr/bin/env python3
"""Validate and summarize paired Qwen vLLM model-latency trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from flashdec.benchmark import validate_vllm_cache_root
from flashdec.vllm_attestation import (
    SPLIT_ATTESTATION_CSV_FIELDS,
    VALID_SPLIT_COUNTS,
    canonical_attestation_bytes,
    validate_split_attestation,
)


BACKENDS = ("vllm_triton_attn", "flashdec")
BACKEND_ARGS = {
    "vllm_triton_attn": "TRITON_ATTN",
    "flashdec": "CUSTOM",
}
TARGET_CASE = "qwen_b8_i8192_o4096"
GUARDRAIL_CASE = "qwen_b8_i512_o2"
FORMAL_CASE_SHAPES = {
    GUARDRAIL_CASE: (8, 512, 2),
    TARGET_CASE: (8, 8192, 4096),
}
FORMAL_TRIALS = {1, 2, 3, 4}
FORMAL_PRIME_ITERS = 1
FORMAL_WARMUP_ITERS = 1
FORMAL_NUM_ITERS = 1
ACCURACY_PREFIX_LEN = 2
TARGET_RATIO_LIMIT = 0.970
GUARDRAIL_RATIO_LIMIT = 1.05
MAX_RATIO_SPREAD = 0.03
DATASET_GENERATION_PROTOCOL = (
    "sha256-indexed-u64be-mod-model-tokenizer-nonspecial-v2"
)
TIMING_SCOPE = (
    "wall-clock blocking LLM.generate call only; full-length JIT-prime and "
    "warmup calls, model load, engine startup, JIT/graph capture, and result "
    "hashing excluded"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")
    required = {
        "schema_version",
        "started_at",
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
        "prime_iters",
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
        "accuracy_prefix_len",
        "case",
        "batch_size",
        "input_len",
        "output_len",
        "backend",
        "backend_arg",
        "trial",
        "run_order",
        "dataset_path",
        "dataset_sha256",
        "output_token_ids_sha256",
        "cross_backend_exact_sequences",
        "cross_backend_common_prefix_tokens",
        "cross_backend_min_common_prefix_tokens",
        "cross_backend_generated_tokens",
        "cross_backend_full_hash_equal",
        "cross_backend_accuracy_prefix_pass",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "output_tokens_per_s",
        "raw_json",
        "log",
        "command",
    } | set(SPLIT_ATTESTATION_CSV_FIELDS)
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def _validate_row_attestation(
    row: dict[str, str],
    *,
    seen_paths: set[str],
    seen_nonces: set[str],
) -> bool | None:
    if row["backend"] == "vllm_triton_attn":
        if any(row[field] != "" for field in SPLIT_ATTESTATION_CSV_FIELDS):
            raise ValueError("native row must not contain split attestation")
        return None

    if any(row[field] == "" for field in SPLIT_ATTESTATION_CSV_FIELDS):
        raise ValueError("FlashDec row is missing split attestation")
    marker_path = row["split_attestation_path"]
    nonce = row["split_attestation_nonce"]
    if not Path(marker_path).is_absolute():
        raise ValueError("split-attestation marker path must be absolute")
    if marker_path in seen_paths:
        raise ValueError("split-attestation marker paths must be unique")
    if nonce in seen_nonces:
        raise ValueError("split-attestation nonces must be unique")
    seen_paths.add(marker_path)
    seen_nonces.add(nonce)

    try:
        payload = json.loads(row["split_attestation_json"])
    except json.JSONDecodeError as error:
        raise ValueError("split-attestation CSV JSON is invalid") from error
    validate_split_attestation(
        payload,
        expected_nonce=nonce,
        expected_case=row["case"],
        expected_trial=int(row["trial"]),
        expected_dataset_sha256=row["dataset_sha256"],
        expected_git_commit=row["git_commit"],
        expected_min_seq_len=int(row["input_len"]),
        expected_num_reqs=8,
    )
    canonical = canonical_attestation_bytes(payload)
    canonical_text = canonical.decode("utf-8").removesuffix("\n")
    if row["split_attestation_json"] != canonical_text:
        raise ValueError("split-attestation CSV JSON is not canonical")
    if hashlib.sha256(canonical).hexdigest() != row["split_attestation_sha256"]:
        raise ValueError("split-attestation SHA-256 differs from payload")

    flattened = {
        "schema_version",
        "nonce",
        "engine_pid",
        "backend",
        "case",
        "trial",
        "dataset_sha256",
        "git_commit",
        "max_seq_len",
        "logical_blocks",
        "num_reqs",
        "num_splits",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "query_dtype",
        "kv_cache_dtype",
        "cuda_graph_capture",
    }
    for field in flattened:
        if row[f"split_attestation_{field}"] != str(payload[field]):
            raise ValueError(
                f"split-attestation flattened {field} differs from payload"
            )
    if (
        payload["num_splits"] not in VALID_SPLIT_COUNTS
        or payload["num_reqs"] != 8
        or payload["num_q_heads"] != 16
        or payload["num_kv_heads"] != 2
        or payload["head_dim"] != 128
        or payload["block_size"] not in (16, 32)
        or payload["query_dtype"] != "bfloat16"
        or payload["kv_cache_dtype"] != "bfloat16"
    ):
        raise ValueError(
            "split-attestation observed shape differs from formal protocol"
        )
    return payload["cuda_graph_capture"]


def summarize(input_path: Path, output_path: Path) -> str:
    rows = _read_rows(input_path)
    invariant_fields = (
        "schema_version",
        "started_at",
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
        "prime_iters",
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
        "accuracy_prefix_len",
    )
    first = rows[0]
    case_datasets: dict[str, tuple[str, ...]] = {}
    case_output_hashes: dict[str, set[str]] = defaultdict(set)
    backend_output_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_attestation_paths: set[str] = set()
    split_attestation_nonces: set[str] = set()
    split_capture_states: list[bool] = []
    for row in rows:
        if any(row.get(field) != first.get(field) for field in invariant_fields):
            raise ValueError("environment/model/protocol invariants differ across rows")
        if row["schema_version"] != "5":
            raise ValueError("unsupported schema_version")
        validate_vllm_cache_root(row["vllm_cache_root"], row["git_commit"])
        if row["git_worktree_clean"] != "True":
            raise ValueError("formal model evidence requires a clean worktree")
        if row["backend"] not in BACKENDS:
            raise ValueError(f"unknown backend: {row['backend']}")
        if row["backend_arg"] != BACKEND_ARGS[row["backend"]]:
            raise ValueError("backend/backend_arg mapping differs from formal protocol")
        if row["model_id"] != "Qwen2.5-3B-Instruct":
            raise ValueError("formal model latency requires Qwen2.5-3B-Instruct")
        if row["vllm_version"] != "0.25.1":
            raise ValueError("formal model latency requires vLLM 0.25.1")
        if row["dtype"] != "bfloat16" or row["kv_cache_dtype"] != "bfloat16":
            raise ValueError("formal model latency requires BF16 model and KV cache")
        if (
            row["max_model_len"] != "12288"
            or row["max_num_seqs"] != "8"
            or row["max_num_batched_tokens"] != "2048"
            or not math.isclose(float(row["gpu_memory_utilization"]), 0.85)
            or row["compilation_mode"] != "default_inductor_cudagraph"
            or row["flashdec_num_splits"] != "auto"
        ):
            raise ValueError("capacity/compilation settings differ from formal protocol")
        if (
            row["prime_iters"] != str(FORMAL_PRIME_ITERS)
            or row["warmup_iters"] != str(FORMAL_WARMUP_ITERS)
            or row["num_iters"] != str(FORMAL_NUM_ITERS)
            or row["dataset_seed"] != "20260830"
            or row["sampling_seed"] != "20260830"
        ):
            raise ValueError("trial strength or seeds differ from formal protocol")
        for hash_field in (
            "model_config_sha256",
            "tokenizer_config_sha256",
            "model_manifest_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", row[hash_field]):
                raise ValueError(f"invalid {hash_field}")
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
        if row["accuracy_prefix_len"] != str(ACCURACY_PREFIX_LEN):
            raise ValueError("unsupported cross-backend accuracy prefix length")
        if not re.fullmatch(r"[0-9a-f]{64}", row["dataset_sha256"]):
            raise ValueError("invalid dataset SHA-256")
        if not re.fullmatch(r"[0-9a-f]{64}", row["output_token_ids_sha256"]):
            raise ValueError("invalid generated-token SHA-256")
        if not Path(row["dataset_path"]).is_absolute():
            raise ValueError("dataset path must be absolute")
        if not Path(row["raw_json"]).is_absolute() or not Path(row["log"]).is_absolute():
            raise ValueError("raw JSON and log paths must be absolute")
        if not row["command"].strip():
            raise ValueError("worker command must be recorded")
        expected_shape = FORMAL_CASE_SHAPES.get(row["case"])
        actual_shape = tuple(
            int(row[field]) for field in ("batch_size", "input_len", "output_len")
        )
        if expected_shape is None or actual_shape != expected_shape:
            raise ValueError("case name/shape differs from the frozen formal matrix")
        capture_state = _validate_row_attestation(
            row,
            seen_paths=split_attestation_paths,
            seen_nonces=split_attestation_nonces,
        )
        if capture_state is not None:
            split_capture_states.append(capture_state)
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
        case_output_hashes[row["case"]].add(row["output_token_ids_sha256"])
        backend_output_hashes[(row["case"], row["backend"])].add(
            row["output_token_ids_sha256"]
        )
        latency_values = tuple(
            float(row[field])
            for field in ("avg_latency_ms", "p50_latency_ms", "p90_latency_ms")
        )
        output_tps = float(row["output_tokens_per_s"])
        if any(not math.isfinite(value) or value <= 0 for value in latency_values):
            raise ValueError("latencies must be positive")
        if not math.isfinite(output_tps) or output_tps <= 0:
            raise ValueError("output throughput must be finite and positive")
        expected_tps = actual_shape[0] * actual_shape[2] * 1000.0 / latency_values[1]
        if not math.isclose(output_tps, expected_tps, rel_tol=1e-5, abs_tol=1e-3):
            raise ValueError("output throughput does not match p50 latency")

    paired: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    if any(len(hashes) != 1 for hashes in backend_output_hashes.values()):
        raise ValueError(
            "each case/backend must produce one deterministic full-output hash "
            "across independent processes"
        )
    for row in rows:
        trial = int(row["trial"])
        if trial not in FORMAL_TRIALS:
            raise ValueError("trial IDs differ from the frozen formal protocol")
        expected_order = (
            ("vllm_triton_attn", "flashdec")
            if trial % 2
            else ("flashdec", "vllm_triton_attn")
        )
        if int(row["run_order"]) != expected_order.index(row["backend"]) + 1:
            raise ValueError("backend run order differs from the frozen AB/BA protocol")
        key = (row["case"], trial)
        if row["backend"] in paired[key]:
            raise ValueError(f"duplicate backend row for {key}")
        paired[key][row["backend"]] = row
    if any(set(pair) != set(BACKENDS) for pair in paired.values()):
        raise ValueError("every case/trial must contain both paired backends")
    if len(split_capture_states) != len(paired):
        raise ValueError("every FlashDec row must carry one split attestation")

    parity_by_case: dict[str, list[dict[str, int | bool]]] = defaultdict(list)
    for (case, _trial), pair in paired.items():
        native_row = pair["vllm_triton_attn"]
        flashdec_row = pair["flashdec"]
        parity_fields = (
            "cross_backend_exact_sequences",
            "cross_backend_common_prefix_tokens",
            "cross_backend_min_common_prefix_tokens",
            "cross_backend_generated_tokens",
            "cross_backend_full_hash_equal",
            "cross_backend_accuracy_prefix_pass",
        )
        if any(native_row[field] != flashdec_row[field] for field in parity_fields):
            raise ValueError("paired rows disagree on cross-backend parity metrics")
        batch_size = int(native_row["batch_size"])
        output_len = int(native_row["output_len"])
        exact_sequences = int(native_row["cross_backend_exact_sequences"])
        common_prefix_tokens = int(
            native_row["cross_backend_common_prefix_tokens"]
        )
        min_common_prefix = int(
            native_row["cross_backend_min_common_prefix_tokens"]
        )
        generated_tokens = int(native_row["cross_backend_generated_tokens"])
        if native_row["cross_backend_full_hash_equal"] not in ("True", "False"):
            raise ValueError("invalid full-rollout hash parity flag")
        if native_row["cross_backend_accuracy_prefix_pass"] not in (
            "True",
            "False",
        ):
            raise ValueError("invalid accuracy-prefix parity flag")
        full_hash_equal = (
            native_row["output_token_ids_sha256"]
            == flashdec_row["output_token_ids_sha256"]
        )
        accuracy_prefix_pass = min_common_prefix >= ACCURACY_PREFIX_LEN
        if (
            generated_tokens != batch_size * output_len
            or not 0 <= exact_sequences <= batch_size
            or not 0 <= min_common_prefix <= output_len
            or not 0 <= common_prefix_tokens <= generated_tokens
            or (native_row["cross_backend_full_hash_equal"] == "True")
            != full_hash_equal
            or (native_row["cross_backend_accuracy_prefix_pass"] == "True")
            != accuracy_prefix_pass
            or (exact_sequences == batch_size) != full_hash_equal
        ):
            raise ValueError("cross-backend parity metrics are internally inconsistent")
        parity_by_case[case].append(
            {
                "exact_sequences": exact_sequences,
                "common_prefix_tokens": common_prefix_tokens,
                "min_common_prefix": min_common_prefix,
                "generated_tokens": generated_tokens,
                "accuracy_prefix_pass": accuracy_prefix_pass,
            }
        )

    by_case: dict[str, list[dict[str, dict[str, str]]]] = defaultdict(list)
    for (case, _trial), pair in paired.items():
        by_case[case].append(pair)
    required_cases = set(FORMAL_CASE_SHAPES)
    if set(by_case) != required_cases:
        raise ValueError(
            "formal summary requires exactly the frozen model cases: "
            f"{sorted(required_cases)}"
        )
    if any(
        {trial for (case_name, trial) in paired if case_name == case}
        != FORMAL_TRIALS
        for case in required_cases
    ):
        raise ValueError(
            "formal model evidence requires exactly the frozen paired trials"
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
        paired_median_ratio = statistics.median(ratios)
        results.append(
            {
                "case": case,
                "trials": len(pairs),
                "native": native,
                "flashdec": flashdec,
                "ratio": paired_median_ratio,
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "reduction_pct": (1.0 - paired_median_ratio) * 100.0,
                "throughput_uplift_pct": (
                    1.0 / paired_median_ratio - 1.0
                ) * 100.0,
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
    accuracy_pass = all(
        parity["accuracy_prefix_pass"]
        for values in parity_by_case.values()
        for parity in values
    )
    gate_pass = target_pass and guardrail_pass and stability_pass and accuracy_pass
    geo_ratio = math.exp(
        statistics.mean(math.log(result["ratio"]) for result in results)
    )

    lines = [
        "# R8 Qwen2.5-3B vLLM Model Latency Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        (
            f"- Rows: {len(rows)}; paired backend process pairs: {len(paired)} "
            f"({len(FORMAL_TRIALS)} trials per case)."
        ),
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
            "- Per-process iteration counts: full-length JIT-prime "
            f"`{first['prime_iters']}`; full-length warmup "
            f"`{first['warmup_iters']}`; measured `{first['num_iters']}`."
        ),
        (
            "- JIT-prime output hashes are retained in raw worker JSON for "
            "audit only; warmup and measured hashes remain the determinism gate."
        ),
        (
            f"- Integration guardrail: `{GUARDRAIL_CASE}` uses the 512-token "
            "prompt boundary and generates exactly two tokens; the second "
            "generated token covers the first eligible FlashDec split decode."
        ),
        (
            "- Split activation: every CUSTOM worker supplied a unique, "
            "canonical engine-process marker proving a successful multi-split "
            "FlashDec launch before warmup/timing."
        ),
        (
            "- First observed split occurred during CUDA Graph capture in "
            f"{sum(split_capture_states)}/{len(split_capture_states)} CUSTOM "
            "workers (recorded, not a pass/fail condition)."
        ),
        f"- Git commit: `{first['git_commit']}`; clean at start: True.",
        "- Per-case prompt dataset identities:",
        *(
            f"  - `{case}`: `{values[4]}` (`{values[3]}`)."
            for case, values in sorted(case_datasets.items())
        ),
        "- Per-case generated-token identities:",
        *(
            f"  - `{case}`: minimum cross-backend common prefix "
            f"`{min(value['min_common_prefix'] for value in parity_by_case[case])}` "
            f"tokens/request; {len(case_output_hashes[case])} unique full-rollout "
            "SHA-256 (descriptive only)."
            for case in sorted(parity_by_case)
        ),
        "",
        "## Paired Results",
        "",
        (
            "Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. "
            "Latency is fixed-batch, end-to-end `LLM.generate` time with model "
            "loading and compilation excluded. Reduction and TPS uplift are "
            "derived from the same paired-median ratio used by the gate."
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
                "confirmatory four-trial balanced AB/BA run."
            ),
            (
                f"- B8 input8192/output4096 target <= {TARGET_RATIO_LIMIT:.3f}x "
                "(at least 3% end-to-end latency reduction): "
                f"{'PASS' if target_pass else 'FAIL'}."
            ),
            (
                "- B8 input512/output2 two-token split-decode guardrail <= "
                f"{GUARDRAIL_RATIO_LIMIT:.2f}x: "
                f"{'PASS' if guardrail_pass else 'FAIL'}."
            ),
            (
                f"- Every case paired-ratio spread <= {MAX_RATIO_SPREAD:.2f}: "
                f"{'PASS' if stability_pass else 'FAIL'}."
            ),
            (
                f"- Every request shares at least {ACCURACY_PREFIX_LEN} output "
                "tokens across backends (the second generated token covers the "
                "first eligible FlashDec split decode): "
                f"{'PASS' if accuracy_pass else 'FAIL'}."
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
            (
                "The first two greedy tokens must match from identical prompt state; "
                "at the 512-token prompt boundary, the second generated token "
                "covers the first eligible FlashDec split-decode decision. Full "
                "autoregressive rollout hashes are "
                "descriptive because one "
                "near-tied floating-point choice can change all later inputs."
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
