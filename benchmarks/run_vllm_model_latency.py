#!/usr/bin/env python3
"""Run paired vLLM latency processes on frozen Qwen2.5-3B workloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flashdec.benchmark import (
    vllm_cache_root_for_commit,
    write_vllm_cache_log_metadata,
)
from flashdec.vllm_attestation import (
    SPLIT_ATTESTATION_CSV_FIELDS,
    SPLIT_ATTESTATION_ENV,
    canonical_attestation_bytes,
    validate_split_attestation,
)


SCHEMA_VERSION = 5
WORKER_RESULT_SCHEMA_VERSION = 3
DATASET_SCHEMA_VERSION = 2
DATASET_GENERATION_PROTOCOL = (
    "sha256-indexed-u64be-mod-model-tokenizer-nonspecial-v2"
)
DATASET_GENERATION_DESCRIPTION = (
    "For every request/position, SHA-256 hashes the UTF-8 string "
    "'flashdec-vllm-model-latency-v1|seed=<seed>|case=<case>|request=<index>"
    "|position=<index>'. The first unsigned big-endian 64-bit word is reduced "
    "modulo the count of vocabulary IDs that are not marked special by either "
    "the model or tokenizer configuration, then maps to that ranked ID."
)
TIMING_SCOPE = (
    "wall-clock blocking LLM.generate call only; full-length JIT-prime and "
    "warmup calls, model load, engine startup, JIT/graph capture, and result "
    "hashing excluded"
)
MODEL_ID = "Qwen2.5-3B-Instruct"
ACCURACY_PREFIX_LEN = 2
BACKENDS = (
    ("vllm_triton_attn", "TRITON_ATTN"),
    ("flashdec", "CUSTOM"),
)


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    input_len: int
    output_len: int
    run_by_default: bool = True


CASES = (
    Case("qwen_b8_i128_o128", 8, 128, 128),
    Case("qwen_b8_i2048_o128", 8, 2048, 128),
    # Formal/pilot additions are explicit opt-ins so historical defaults remain
    # unchanged and callers deliberately select the intended evidence workload.
    Case("qwen_b8_i128_o2", 8, 128, 2, run_by_default=False),
    Case("qwen_b8_i512_o2", 8, 512, 2, run_by_default=False),
    Case("qwen_b8_i2048_o2048", 8, 2048, 2048, run_by_default=False),
    Case("qwen_b8_i8192_o4096", 8, 8192, 4096, run_by_default=False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _special_token_ids(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    vocab_size: int,
) -> list[int]:
    values: set[int] = set()
    for config in (model_config, tokenizer_config):
        for key, value in config.items():
            if not key.endswith("_token_id"):
                continue
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if (
                    isinstance(candidate, int)
                    and not isinstance(candidate, bool)
                    and 0 <= candidate < vocab_size
                ):
                    values.add(candidate)

    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    if not isinstance(added_tokens, dict):
        raise ValueError("tokenizer added_tokens_decoder must be a JSON object")
    for raw_token_id, metadata in added_tokens.items():
        if not isinstance(metadata, dict) or metadata.get("special") is not True:
            continue
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as error:
            raise ValueError("special added-token ID must be an integer") from error
        if 0 <= token_id < vocab_size:
            values.add(token_id)
    return sorted(values)


def _allowed_token_id(rank: int, excluded_token_ids: list[int]) -> int:
    """Map an allowed-vocabulary rank without materializing the vocabulary."""

    token_id = rank
    for excluded in excluded_token_ids:
        if excluded <= token_id:
            token_id += 1
        else:
            break
    return token_id


def _generate_dataset(
    case: Case,
    *,
    seed: int,
    vocab_size: int,
    excluded_token_ids: list[int],
) -> dict[str, Any]:
    if seed < 0 or seed >= 2**64:
        raise ValueError("dataset seed must fit an unsigned 64-bit integer")
    if vocab_size <= 0:
        raise ValueError("model vocab_size must be positive")
    excluded = sorted(set(excluded_token_ids))
    if any(value < 0 or value >= vocab_size for value in excluded):
        raise ValueError("excluded token ID lies outside the vocabulary")
    allowed_count = vocab_size - len(excluded)
    if allowed_count <= 0:
        raise ValueError("vocabulary contains no usable prompt token IDs")

    prompts: list[list[int]] = []
    for request_index in range(case.batch_size):
        prompt: list[int] = []
        for position in range(case.input_len):
            material = (
                "flashdec-vllm-model-latency-v1"
                f"|seed={seed}|case={case.name}"
                f"|request={request_index}|position={position}"
            ).encode("utf-8")
            value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            prompt.append(_allowed_token_id(value % allowed_count, excluded))
        prompts.append(prompt)
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generation_protocol": DATASET_GENERATION_PROTOCOL,
        "generation_description": DATASET_GENERATION_DESCRIPTION,
        "seed": seed,
        "case": case.name,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "vocab_size": vocab_size,
        "excluded_token_ids": excluded,
        "prompt_token_ids": prompts,
    }


def _write_dataset(path: Path, dataset: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    encoded = _canonical_json_bytes(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _parse_case_spec(spec: str) -> Case:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError("case-spec must be NAME:BATCH:INPUT_LEN:OUTPUT_LEN")
    name = parts[0]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("case name may contain only letters, digits, '.', '_', '-'")
    try:
        batch_size, input_len, output_len = (int(value) for value in parts[1:])
    except ValueError as error:
        raise ValueError("case-spec shape values must be integers") from error
    if min(batch_size, input_len, output_len) <= 0:
        raise ValueError("case-spec shape values must be positive")
    return Case(name, batch_size, input_len, output_len, run_by_default=False)


def _resolve_cases(
    selected_names: list[str] | None,
    case_specs: list[str] | None,
) -> list[Case]:
    by_name = {case.name: case for case in CASES}
    custom_names: list[str] = []
    for spec in case_specs or []:
        case = _parse_case_spec(spec)
        if case.name in by_name:
            raise ValueError(f"duplicate case name: {case.name}")
        by_name[case.name] = case
        custom_names.append(case.name)

    if selected_names:
        unknown = set(selected_names) - set(by_name)
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        names = list(dict.fromkeys(selected_names))
    elif custom_names:
        names = custom_names
    else:
        names = [case.name for case in CASES if case.run_by_default]
    return [by_name[name] for name in names]


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prime-iters", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--dataset-seed", type=int, default=20260830)
    parser.add_argument("--sampling-seed", type=int, default=20260830)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--vllm-cache-base",
        type=Path,
        help=(
            "Base directory for commit-scoped vLLM caches; defaults to "
            "FLASHDEC_VLLM_CACHE_BASE, then VLLM_CACHE_ROOT, then "
            "~/.cache/vllm-flashdec"
        ),
    )
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "Run only a named built-in or --case-spec case; may be repeated. "
            "The qwen_b8_i128_o2, qwen_b8_i512_o2, "
            "qwen_b8_i2048_o2048, and qwen_b8_i8192_o4096 built-ins are "
            "opt-in."
        ),
    )
    parser.add_argument(
        "--case-spec",
        action="append",
        metavar="NAME:BATCH:INPUT_LEN:OUTPUT_LEN",
        help=(
            "Define a deterministic token-ID case. If --case is omitted, only "
            "the custom cases are run; may be repeated."
        ),
    )
    return parser.parse_args()


def _validate_environment(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.trials <= 0
        or args.prime_iters < 0
        or args.warmup_iters < 0
        or args.num_iters <= 0
    ):
        raise ValueError(
            "trials/num-iters must be positive and prime/warmup non-negative"
        )
    if min(
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
    ) <= 0:
        raise ValueError("vLLM capacity limits must be positive")
    if any(
        seed < 0 or seed >= 2**64
        for seed in (args.dataset_seed, args.sampling_seed)
    ):
        raise ValueError("dataset/sampling seeds must fit unsigned 64-bit integers")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("set VLLM_USE_FLASHINFER_SAMPLER=0")
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1":
        raise RuntimeError("set VLLM_WSL2_ENABLE_PIN_MEMORY=1")
    if os.environ.get("VLLM_PLUGINS") != "flashdec":
        raise RuntimeError("set VLLM_PLUGINS=flashdec")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "1":
        raise RuntimeError("set VLLM_ENABLE_V1_MULTIPROCESSING=1")
    if args.require_clean and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean needs a clean Git worktree")

    model_config = args.model / "config.json"
    tokenizer_config = args.model / "tokenizer_config.json"
    model_manifest = args.model / "SHA256SUMS"
    if (
        not model_config.is_file()
        or not tokenizer_config.is_file()
        or not model_manifest.is_file()
    ):
        raise FileNotFoundError(
            "model config.json, tokenizer_config.json, and SHA256SUMS are required"
        )
    worker = Path(__file__).with_name("run_vllm_model_latency_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(f"model-latency worker not found: {worker}")
    return model_config, tokenizer_config, model_manifest, worker


def _command(
    *,
    worker: Path,
    model: Path,
    dataset: Path,
    dataset_sha256: str,
    backend: str,
    prime_iters: int,
    warmup_iters: int,
    num_iters: int,
    sampling_seed: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    output_json: Path,
    split_attestation_path: Path | None = None,
    split_attestation_nonce: str | None = None,
    split_attestation_trial: int | None = None,
    split_attestation_git_commit: str | None = None,
) -> list[str]:
    attestation_values = (
        split_attestation_path,
        split_attestation_nonce,
        split_attestation_trial,
        split_attestation_git_commit,
    )
    if backend == "CUSTOM" and any(value is None for value in attestation_values):
        raise ValueError("CUSTOM command requires complete split attestation")
    if backend != "CUSTOM" and any(
        value is not None for value in attestation_values
    ):
        raise ValueError("native command must not receive split attestation")
    command = [
        sys.executable,
        str(worker),
        "--backend",
        backend,
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--dataset-sha256",
        dataset_sha256,
        "--prime-iters",
        str(prime_iters),
        "--warmup-iters",
        str(warmup_iters),
        "--num-iters",
        str(num_iters),
        "--sampling-seed",
        str(sampling_seed),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--output",
        str(output_json),
    ]
    if backend == "CUSTOM":
        command.extend(
            [
                "--split-attestation-path",
                str(split_attestation_path),
                "--split-attestation-nonce",
                str(split_attestation_nonce),
                "--split-attestation-trial",
                str(split_attestation_trial),
                "--split-attestation-git-commit",
                str(split_attestation_git_commit),
            ]
        )
    return command


def _validate_worker_result(
    result: dict[str, Any],
    *,
    case: Case,
    backend_arg: str,
    dataset_path: Path,
    dataset_sha256: str,
    dataset_seed: int,
    sampling_seed: int,
    prime_iters: int,
    warmup_iters: int,
    num_iters: int,
    expected_attestation_path: Path | None = None,
    expected_attestation_nonce: str | None = None,
    expected_attestation_trial: int | None = None,
    expected_attestation_git_commit: str | None = None,
    expected_environment: dict[str, str] | None = None,
) -> tuple[
    list[float],
    float,
    float,
    float,
    str,
    tuple[tuple[int, ...], ...],
    dict[str, Any] | None,
]:
    expected = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "backend_arg": backend_arg,
        "case": case.name,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_seed": dataset_seed,
        "dataset_generation_protocol": DATASET_GENERATION_PROTOCOL,
        "prompt_format": "token_ids",
        "skip_tokenizer_init": True,
        "sampling_seed": sampling_seed,
        "sampling_n": 1,
        "sampling_temperature": 0.0,
        "sampling_min_tokens": case.output_len,
        "sampling_max_tokens": case.output_len,
        "sampling_ignore_eos": True,
        "sampling_detokenize": False,
        "prime_iters": prime_iters,
        "warmup_iters": warmup_iters,
        "num_iters": num_iters,
        "timing_scope": TIMING_SCOPE,
        "vllm_engine_multiprocessing": True,
        "accuracy_prefix_len": ACCURACY_PREFIX_LEN,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"worker result {field!r} differs: expected {value!r}, "
                f"got {result.get(field)!r}"
            )
    for field, value in (expected_environment or {}).items():
        if result.get(field) != value:
            raise ValueError(
                f"worker environment {field!r} differs: expected {value!r}, "
                f"got {result.get(field)!r}"
            )

    attestation_payload = result.get("split_activation_attestation")
    attestation_path = result.get("split_activation_attestation_path")
    attestation_sha256 = result.get("split_activation_attestation_sha256")
    if backend_arg == "TRITON_ATTN":
        if any(
            value is not None
            for value in (
                attestation_payload,
                attestation_path,
                attestation_sha256,
                expected_attestation_path,
                expected_attestation_nonce,
                expected_attestation_trial,
                expected_attestation_git_commit,
            )
        ):
            raise ValueError("native worker must not contain split attestation")
        attestation_record = None
    else:
        if any(
            value is None
            for value in (
                expected_attestation_path,
                expected_attestation_nonce,
                expected_attestation_trial,
                expected_attestation_git_commit,
            )
        ):
            raise ValueError("CUSTOM validation requires split attestation binding")
        expected_path = expected_attestation_path
        if not isinstance(attestation_path, str) or not Path(
            attestation_path
        ).is_absolute():
            raise ValueError("worker returned invalid split-attestation path")
        if Path(attestation_path) != expected_path:
            raise ValueError("worker split-attestation path binding differs")
        if not isinstance(attestation_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", attestation_sha256
        ):
            raise ValueError("worker returned invalid split-attestation SHA-256")
        validate_split_attestation(
            attestation_payload,
            expected_nonce=expected_attestation_nonce,
            expected_case=case.name,
            expected_trial=expected_attestation_trial,
            expected_dataset_sha256=dataset_sha256,
            expected_git_commit=expected_attestation_git_commit,
            expected_min_seq_len=case.input_len,
            expected_num_reqs=case.batch_size,
        )
        encoded_attestation = canonical_attestation_bytes(attestation_payload)
        if hashlib.sha256(encoded_attestation).hexdigest() != attestation_sha256:
            raise ValueError("worker split-attestation SHA-256 differs from payload")
        try:
            on_disk_attestation = expected_path.read_bytes()
        except FileNotFoundError as error:
            raise ValueError("worker split-attestation marker is missing") from error
        if on_disk_attestation != encoded_attestation:
            raise ValueError("worker split-attestation marker differs from payload")
        attestation_record = {
            "path": attestation_path,
            "sha256": attestation_sha256,
            "payload": attestation_payload,
        }

    latencies = [float(value) for value in result.get("latencies_s", [])]
    if len(latencies) != num_iters or any(
        not math.isfinite(value) or value <= 0 for value in latencies
    ):
        raise ValueError("worker returned invalid latency samples")
    prime_hashes = result.get("prime_output_sha256")
    warmup_hashes = result.get("warmup_output_sha256")
    measured_hashes = result.get("measured_output_sha256")
    if not isinstance(prime_hashes, list) or len(prime_hashes) != prime_iters:
        raise ValueError("worker returned invalid JIT-prime output hashes")
    if not isinstance(warmup_hashes, list) or len(warmup_hashes) != warmup_iters:
        raise ValueError("worker returned invalid warmup output hashes")
    if not isinstance(measured_hashes, list) or len(measured_hashes) != num_iters:
        raise ValueError("worker returned invalid measured output hashes")
    for digest in [*prime_hashes, *warmup_hashes, *measured_hashes]:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("worker returned an invalid output SHA-256")
    output_sha256 = result.get("output_token_ids_sha256")
    if not isinstance(output_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ):
        raise ValueError("worker returned an invalid canonical output SHA-256")
    if set([*warmup_hashes, *measured_hashes]) != {output_sha256}:
        raise ValueError(
            "worker fixed-greedy outputs differ within one process"
        )
    output_token_ids = result.get("output_token_ids")
    if (
        not isinstance(output_token_ids, list)
        or len(output_token_ids) != case.batch_size
        or any(
            not isinstance(tokens, list)
            or len(tokens) != case.output_len
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in tokens
            )
            for tokens in output_token_ids
        )
    ):
        raise ValueError("worker returned invalid output token IDs")
    canonical_output = json.dumps(
        output_token_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_output).hexdigest() != output_sha256:
        raise ValueError("worker output token IDs do not match their SHA-256")

    avg_s = float(result["avg_latency_s"])
    p50_s = float(result["percentiles_s"]["50"])
    p90_s = float(result["percentiles_s"]["90"])
    if any(
        not math.isfinite(value) or value <= 0
        for value in (avg_s, p50_s, p90_s)
    ):
        raise ValueError("worker returned non-positive latency aggregates")
    ordered = sorted(latencies)

    def percentile(percent: float) -> float:
        rank = (len(ordered) - 1) * percent / 100.0
        lower = math.floor(rank)
        upper = math.ceil(rank)
        fraction = rank - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    expected_aggregates = (
        statistics.fmean(latencies),
        percentile(50.0),
        percentile(90.0),
    )
    for name, actual, expected_value in zip(
        ("average", "p50", "p90"),
        (avg_s, p50_s, p90_s),
        expected_aggregates,
    ):
        if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"worker {name} latency aggregate does not match raw samples"
            )
    return (
        latencies,
        avg_s,
        p50_s,
        p90_s,
        output_sha256,
        tuple(tuple(tokens) for tokens in output_token_ids),
        attestation_record,
    )


def _cross_backend_parity(
    native_tokens: tuple[tuple[int, ...], ...],
    flashdec_tokens: tuple[tuple[int, ...], ...],
    *,
    accuracy_prefix_len: int = ACCURACY_PREFIX_LEN,
) -> dict[str, int | bool]:
    if len(native_tokens) != len(flashdec_tokens) or not native_tokens:
        raise ValueError("cross-backend output batches must be equal and non-empty")
    common_prefixes: list[int] = []
    exact_sequences = 0
    generated_tokens = 0
    for native_sequence, flashdec_sequence in zip(
        native_tokens, flashdec_tokens, strict=True
    ):
        if len(native_sequence) != len(flashdec_sequence) or not native_sequence:
            raise ValueError(
                "cross-backend output sequences must be equal-length and non-empty"
            )
        common_prefix = next(
            (
                index
                for index, (native_token, flashdec_token) in enumerate(
                    zip(native_sequence, flashdec_sequence, strict=True)
                )
                if native_token != flashdec_token
            ),
            len(native_sequence),
        )
        common_prefixes.append(common_prefix)
        exact_sequences += native_sequence == flashdec_sequence
        generated_tokens += len(native_sequence)
    full_equal = exact_sequences == len(native_tokens)
    min_common_prefix = min(common_prefixes)
    return {
        "cross_backend_exact_sequences": exact_sequences,
        "cross_backend_common_prefix_tokens": sum(common_prefixes),
        "cross_backend_min_common_prefix_tokens": min_common_prefix,
        "cross_backend_generated_tokens": generated_tokens,
        "cross_backend_full_hash_equal": full_equal,
        "cross_backend_accuracy_prefix_pass": (
            min_common_prefix >= accuracy_prefix_len
        ),
    }


def _attestation_csv_values(
    record: dict[str, Any] | None,
) -> dict[str, object]:
    if record is None:
        return {field: "" for field in SPLIT_ATTESTATION_CSV_FIELDS}
    payload = record["payload"]
    values: dict[str, object] = {
        "split_attestation_json": canonical_attestation_bytes(payload)
        .decode("utf-8")
        .removesuffix("\n"),
        "split_attestation_path": record["path"],
        "split_attestation_sha256": record["sha256"],
    }
    for field in (
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
    ):
        values[f"split_attestation_{field}"] = payload[field]
    if set(values) != set(SPLIT_ATTESTATION_CSV_FIELDS):
        raise AssertionError("split-attestation CSV field mapping drifted")
    return values


def main() -> None:
    args = _parse_args()
    (
        model_config,
        tokenizer_config,
        model_manifest,
        worker,
    ) = _validate_environment(args)
    selected = _resolve_cases(args.case, args.case_spec)
    for case in selected:
        if case.batch_size > args.max_num_seqs:
            raise ValueError(f"{case.name} batch_size exceeds max-num-seqs")
        if case.input_len + case.output_len > args.max_model_len:
            raise ValueError(f"{case.name} sequence length exceeds max-model-len")

    config = json.loads(model_config.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise ValueError("model config must contain an integer vocab_size")
    excluded_token_ids = _special_token_ids(config, tokenizer, vocab_size)

    # Heavyweight imports remain inside the GPU-only entry point so pure
    # dataset/command tests do not require a local vLLM installation.
    import torch
    import triton
    import vllm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    raw_dir = args.output.parent / f"{args.output.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    attestation_dir = raw_dir / "split_attestations"
    attestation_dir.mkdir()
    dataset_dir = raw_dir / "datasets"
    datasets: dict[str, tuple[Path, str]] = {}
    for case in selected:
        dataset_path = dataset_dir / f"{case.name}.json"
        dataset = _generate_dataset(
            case,
            seed=args.dataset_seed,
            vocab_size=vocab_size,
            excluded_token_ids=excluded_token_ids,
        )
        dataset_sha256 = _write_dataset(dataset_path, dataset)
        datasets[case.name] = (dataset_path, dataset_sha256)
        print(
            f"DATASET case={case.name} seed={args.dataset_seed} "
            f"protocol={DATASET_GENERATION_PROTOCOL} sha256={dataset_sha256} "
            f"path={dataset_path.resolve()}",
            flush=True,
        )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = _git_value("rev-parse", "HEAD")
    worktree_clean = _git_value("status", "--porcelain") == ""
    cache_root = vllm_cache_root_for_commit(
        commit,
        cache_base=args.vllm_cache_base,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    model_path = args.model.resolve()
    common = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "git_commit": commit,
        "git_worktree_clean": worktree_clean,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_version": triton.__version__,
        "vllm_version": vllm.__version__,
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_config),
        "tokenizer_config_sha256": _sha256(tokenizer_config),
        "model_manifest_sha256": _sha256(model_manifest),
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "compilation_mode": "default_inductor_cudagraph",
        "vllm_cache_root": str(cache_root),
        "flashdec_num_splits": os.environ.get(
            "FLASHDEC_VLLM_NUM_SPLITS", "auto"
        ),
        "prime_iters": args.prime_iters,
        "warmup_iters": args.warmup_iters,
        "num_iters": args.num_iters,
        "dataset_seed": args.dataset_seed,
        "dataset_generation_protocol": DATASET_GENERATION_PROTOCOL,
        "sampling_seed": args.sampling_seed,
        "prompt_format": "token_ids",
        "skip_tokenizer_init": True,
        "sampling_n": 1,
        "sampling_temperature": 0.0,
        "sampling_ignore_eos": True,
        "sampling_detokenize": False,
        "timing_scope": TIMING_SCOPE,
        "vllm_engine_multiprocessing": True,
        "accuracy_prefix_len": ACCURACY_PREFIX_LEN,
    }
    child_env = os.environ.copy()
    for env_name in SPLIT_ATTESTATION_ENV.values():
        child_env.pop(env_name, None)
    child_env["PYTHONHASHSEED"] = "0"
    child_env["VLLM_CACHE_ROOT"] = str(cache_root)
    rows: list[dict[str, object]] = []
    paired_outputs: dict[
        tuple[str, int],
        dict[str, tuple[dict[str, object], tuple[tuple[int, ...], ...]]],
    ] = {}
    print(f"VLLM_CACHE_ROOT={cache_root}", flush=True)

    for case in selected:
        dataset_path, dataset_sha256 = datasets[case.name]
        for trial in range(1, args.trials + 1):
            ordered = BACKENDS if trial % 2 else tuple(reversed(BACKENDS))
            for run_order, (backend_name, backend_arg) in enumerate(ordered, 1):
                stem = f"{case.name}_trial{trial}_{backend_name}"
                output_json = raw_dir / f"{stem}.json"
                log_path = raw_dir / f"{stem}.log"
                if backend_arg == "CUSTOM":
                    split_attestation_path = (
                        attestation_dir / f"{stem}.split.json"
                    ).resolve()
                    split_attestation_nonce = secrets.token_hex(32)
                    split_attestation_trial = trial
                    split_attestation_git_commit = commit
                else:
                    split_attestation_path = None
                    split_attestation_nonce = None
                    split_attestation_trial = None
                    split_attestation_git_commit = None
                command = _command(
                    worker=worker,
                    model=model_path,
                    dataset=dataset_path.resolve(),
                    dataset_sha256=dataset_sha256,
                    backend=backend_arg,
                    prime_iters=args.prime_iters,
                    warmup_iters=args.warmup_iters,
                    num_iters=args.num_iters,
                    sampling_seed=args.sampling_seed,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    output_json=output_json,
                    split_attestation_path=split_attestation_path,
                    split_attestation_nonce=split_attestation_nonce,
                    split_attestation_trial=split_attestation_trial,
                    split_attestation_git_commit=split_attestation_git_commit,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    write_vllm_cache_log_metadata(
                        log,
                        commit=commit,
                        cache_root=cache_root,
                        command=shlex.join(command),
                    )
                    completed = subprocess.run(
                        command,
                        env=child_env,
                        text=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"vLLM benchmark failed ({completed.returncode}); "
                        f"see {log_path}"
                    )
                result = json.loads(output_json.read_text(encoding="utf-8"))
                (
                    _latencies,
                    avg_s,
                    p50_s,
                    p90_s,
                    output_sha256,
                    output_token_ids,
                    attestation_record,
                ) = _validate_worker_result(
                    result,
                    case=case,
                    backend_arg=backend_arg,
                    dataset_path=dataset_path,
                    dataset_sha256=dataset_sha256,
                    dataset_seed=args.dataset_seed,
                    sampling_seed=args.sampling_seed,
                    prime_iters=args.prime_iters,
                    warmup_iters=args.warmup_iters,
                    num_iters=args.num_iters,
                    expected_attestation_path=split_attestation_path,
                    expected_attestation_nonce=split_attestation_nonce,
                    expected_attestation_trial=split_attestation_trial,
                    expected_attestation_git_commit=(
                        split_attestation_git_commit
                    ),
                    expected_environment={
                        "device": common["device"],
                        "torch_version": common["torch_version"],
                        "torch_cuda": common["torch_cuda"],
                        "vllm_version": common["vllm_version"],
                    },
                )
                row = {
                    **common,
                    "case": case.name,
                    "batch_size": case.batch_size,
                    "input_len": case.input_len,
                    "output_len": case.output_len,
                    "backend": backend_name,
                    "backend_arg": backend_arg,
                    "trial": trial,
                    "run_order": run_order,
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": dataset_sha256,
                    "output_token_ids_sha256": output_sha256,
                    "sampling_min_tokens": case.output_len,
                    "sampling_max_tokens": case.output_len,
                    "avg_latency_ms": f"{avg_s * 1000.0:.6f}",
                    "p50_latency_ms": f"{p50_s * 1000.0:.6f}",
                    "p90_latency_ms": f"{p90_s * 1000.0:.6f}",
                    "output_tokens_per_s": (
                        f"{case.batch_size * case.output_len / p50_s:.3f}"
                    ),
                    "raw_json": str(output_json),
                    "log": str(log_path),
                    "command": shlex.join(command),
                    **_attestation_csv_values(attestation_record),
                }
                rows.append(row)
                pair_key = (case.name, trial)
                pair_outputs = paired_outputs.setdefault(pair_key, {})
                pair_outputs[backend_name] = (row, output_token_ids)
                if len(pair_outputs) == len(BACKENDS):
                    native_row, native_tokens = pair_outputs["vllm_triton_attn"]
                    flashdec_row, flashdec_tokens = pair_outputs["flashdec"]
                    parity = _cross_backend_parity(
                        native_tokens,
                        flashdec_tokens,
                    )
                    if parity["cross_backend_full_hash_equal"] != (
                        native_row["output_token_ids_sha256"]
                        == flashdec_row["output_token_ids_sha256"]
                    ):
                        raise ValueError(
                            "cross-backend token equality disagrees with output hashes"
                        )
                    native_row.update(parity)
                    flashdec_row.update(parity)
                    print(
                        f"PARITY case={case.name} trial={trial} "
                        "min_common_prefix="
                        f"{parity['cross_backend_min_common_prefix_tokens']} "
                        "common_prefix="
                        f"{parity['cross_backend_common_prefix_tokens']}/"
                        f"{parity['cross_backend_generated_tokens']} "
                        "exact_sequences="
                        f"{parity['cross_backend_exact_sequences']}/"
                        f"{case.batch_size}",
                        flush=True,
                    )
                print(
                    case.name,
                    backend_name,
                    trial,
                    row["p50_latency_ms"],
                    row["output_tokens_per_s"],
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
