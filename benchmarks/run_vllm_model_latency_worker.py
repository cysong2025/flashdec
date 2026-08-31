#!/usr/bin/env python3
"""Measure one vLLM backend with a pre-generated token-ID dataset.

This worker intentionally owns model construction, JIT priming, and warmup so the
parent runner can launch every backend in a fresh process while keeping model
load and compilation out of the measured ``LLM.generate`` samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from flashdec.vllm_attestation import (
    SPLIT_ATTESTATION_ENV,
    canonical_attestation_bytes,
    validate_split_attestation,
)


DATASET_SCHEMA_VERSION = 2
DATASET_GENERATION_PROTOCOL = (
    "sha256-indexed-u64be-mod-model-tokenizer-nonspecial-v2"
)
RESULT_SCHEMA_VERSION = 3
ACCURACY_PREFIX_LEN = 2
TIMING_SCOPE = (
    "wall-clock blocking LLM.generate call only; full-length JIT-prime and "
    "warmup calls, model load, engine startup, JIT/graph capture, and result "
    "hashing excluded"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one latency sample is required")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_dataset(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    dataset = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "generation_protocol",
        "seed",
        "case",
        "batch_size",
        "input_len",
        "output_len",
        "vocab_size",
        "excluded_token_ids",
        "prompt_token_ids",
    }
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")
    if dataset["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema_version")
    if dataset["generation_protocol"] != DATASET_GENERATION_PROTOCOL:
        raise ValueError("unsupported dataset generation protocol")

    integer_fields = ("seed", "batch_size", "input_len", "output_len", "vocab_size")
    if any(
        not isinstance(dataset[field], int) or isinstance(dataset[field], bool)
        for field in integer_fields
    ):
        raise ValueError("dataset integer metadata must contain JSON integers")
    if dataset["seed"] < 0 or dataset["seed"] >= 2**64:
        raise ValueError("dataset seed must fit an unsigned 64-bit integer")
    if min(
        dataset["batch_size"],
        dataset["input_len"],
        dataset["output_len"],
        dataset["vocab_size"],
    ) <= 0:
        raise ValueError("dataset shape and vocabulary values must be positive")
    if not isinstance(dataset["case"], str) or not dataset["case"]:
        raise ValueError("dataset case must be a non-empty string")

    excluded = dataset["excluded_token_ids"]
    if not isinstance(excluded, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in excluded
    ):
        raise ValueError("excluded_token_ids must be a JSON integer list")
    if excluded != sorted(set(excluded)):
        raise ValueError("excluded_token_ids must be sorted and unique")
    vocab_size = dataset["vocab_size"]
    if any(value < 0 or value >= vocab_size for value in excluded):
        raise ValueError("excluded token ID lies outside the vocabulary")

    prompts = dataset["prompt_token_ids"]
    if not isinstance(prompts, list) or len(prompts) != dataset["batch_size"]:
        raise ValueError("prompt count does not match batch_size")
    excluded_set = set(excluded)
    for prompt in prompts:
        if not isinstance(prompt, list) or len(prompt) != dataset["input_len"]:
            raise ValueError("prompt length does not match input_len")
        for token_id in prompt:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("prompt token IDs must be JSON integers")
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError("prompt token ID lies outside the vocabulary")
            if token_id in excluded_set:
                raise ValueError("prompt contains an excluded special token ID")
    return dataset


def _output_token_ids(
    outputs: Any,
    expected_count: int,
    output_len: int,
) -> list[list[int]]:
    if len(outputs) != expected_count:
        raise ValueError("vLLM returned an unexpected request count")
    token_ids: list[list[int]] = []
    for request_output in outputs:
        candidates = request_output.outputs
        if len(candidates) != 1:
            raise ValueError("vLLM must return exactly one sequence per request")
        generated = [int(value) for value in candidates[0].token_ids]
        if len(generated) != output_len:
            raise ValueError(
                f"expected exactly {output_len} generated tokens, got {len(generated)}"
            )
        token_ids.append(generated)
    return token_ids


def _token_ids_sha256(token_ids: list[list[int]]) -> str:
    encoded = json.dumps(
        token_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _install_split_attestation_environment(
    args: argparse.Namespace,
    dataset: dict[str, Any],
) -> tuple[Path, str, int, str, str] | None:
    marker_values = (
        args.split_attestation_path,
        args.split_attestation_nonce,
        args.split_attestation_trial,
        args.split_attestation_git_commit,
    )
    inherited = {
        name: os.environ[name]
        for name in SPLIT_ATTESTATION_ENV.values()
        if name in os.environ
    }
    if inherited:
        raise RuntimeError(
            "split-attestation environment must be owned by the worker CLI"
        )
    if args.backend == "TRITON_ATTN":
        if any(value is not None for value in marker_values):
            raise ValueError("native worker must not receive split attestation")
        return None
    if any(value is None for value in marker_values):
        raise ValueError("CUSTOM worker requires complete split attestation arguments")

    path = args.split_attestation_path
    if not path.is_absolute():
        raise ValueError("split-attestation path must be absolute")
    if path.exists():
        raise FileExistsError(f"refusing pre-existing split marker {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"split-attestation parent directory not found: {path.parent}"
        )
    nonce = args.split_attestation_nonce
    if not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise ValueError("split-attestation nonce must be 64 lowercase hex digits")
    trial = args.split_attestation_trial
    if trial <= 0:
        raise ValueError("split-attestation trial must be positive")
    git_commit = args.split_attestation_git_commit
    if not re.fullmatch(r"[0-9a-f]{7,64}", git_commit):
        raise ValueError("split-attestation Git commit is invalid")
    values = {
        "path": str(path),
        "nonce": nonce,
        "case": dataset["case"],
        "trial": str(trial),
        "dataset_sha256": args.dataset_sha256,
        "git_commit": git_commit,
    }
    for field, env_name in SPLIT_ATTESTATION_ENV.items():
        os.environ[env_name] = values[field]
    return path, nonce, trial, git_commit, args.dataset_sha256


def _read_marker_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("split-attestation marker must be a regular file")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("split-attestation marker permissions exceed 0600")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > 65536:
                raise ValueError("split-attestation marker is unexpectedly large")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _verify_split_attestation(
    binding: tuple[Path, str, int, str, str] | None,
    *,
    backend: str,
    dataset: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if backend == "TRITON_ATTN":
        if binding is not None:
            raise ValueError("native worker cannot validate split attestation")
        return None, None, None
    if binding is None:
        raise ValueError("CUSTOM worker is missing split attestation binding")
    path, nonce, trial, git_commit, dataset_sha256 = binding
    try:
        encoded = _read_marker_bytes(path)
    except FileNotFoundError as error:
        raise RuntimeError(
            "CUSTOM JIT-prime completed without an observed FlashDec split launch"
        ) from error
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("split-attestation marker is not valid JSON") from error
    validate_split_attestation(
        payload,
        expected_nonce=nonce,
        expected_case=dataset["case"],
        expected_trial=trial,
        expected_dataset_sha256=dataset_sha256,
        expected_git_commit=git_commit,
        expected_min_seq_len=dataset["input_len"],
        expected_num_reqs=dataset["batch_size"],
    )
    canonical = canonical_attestation_bytes(payload)
    if encoded != canonical:
        raise ValueError("split-attestation marker is not canonical JSON")
    return payload, hashlib.sha256(encoded).hexdigest(), str(path)


def _run_generation_phases(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
    *,
    batch_size: int,
    output_len: int,
    prime_iters: int,
    warmup_iters: int,
    num_iters: int,
    after_prime: Callable[
        [], tuple[dict[str, Any] | None, str | None, str | None]
    ],
) -> tuple[
    list[str],
    list[str],
    list[float],
    list[str],
    list[list[int]],
    tuple[dict[str, Any] | None, str | None, str | None],
]:
    prime_output_sha256 = []
    for _ in range(prime_iters):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        token_ids = _output_token_ids(outputs, batch_size, output_len)
        prime_output_sha256.append(_token_ids_sha256(token_ids))

    # This callback is deliberately between the prime and warmup loops. It is
    # the fail-closed boundary that prevents unproved CUSTOM work from reaching
    # either warmup or a timed sample.
    attestation = after_prime()

    warmup_output_sha256 = []
    for _ in range(warmup_iters):
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        token_ids = _output_token_ids(outputs, batch_size, output_len)
        warmup_output_sha256.append(_token_ids_sha256(token_ids))
    latencies_s: list[float] = []
    measured_output_sha256: list[str] = []
    for _ in range(num_iters):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        latencies_s.append(time.perf_counter() - started)
        token_ids = _output_token_ids(outputs, batch_size, output_len)
        measured_output_sha256.append(_token_ids_sha256(token_ids))
    return (
        prime_output_sha256,
        warmup_output_sha256,
        latencies_s,
        measured_output_sha256,
        token_ids,
        attestation,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("TRITON_ATTN", "CUSTOM"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prime-iters", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--num-iters", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--split-attestation-path", type=Path)
    parser.add_argument("--split-attestation-nonce")
    parser.add_argument("--split-attestation-trial", type=int)
    parser.add_argument("--split-attestation-git-commit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.prime_iters < 0 or args.warmup_iters < 0 or args.num_iters <= 0:
        raise ValueError(
            "num-iters must be positive and prime/warmup-iters non-negative"
        )
    if args.sampling_seed < 0 or args.sampling_seed >= 2**64:
        raise ValueError("sampling-seed must fit an unsigned 64-bit integer")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("set VLLM_USE_FLASHINFER_SAMPLER=0")
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1":
        raise RuntimeError("set VLLM_WSL2_ENABLE_PIN_MEMORY=1")
    if os.environ.get("VLLM_PLUGINS") != "flashdec":
        raise RuntimeError("set VLLM_PLUGINS=flashdec")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "1":
        raise RuntimeError("set VLLM_ENABLE_V1_MULTIPROCESSING=1")

    dataset = _load_dataset(args.dataset, args.dataset_sha256)
    if dataset["batch_size"] > args.max_num_seqs:
        raise ValueError("dataset batch_size exceeds max-num-seqs")
    if dataset["input_len"] + dataset["output_len"] > args.max_model_len:
        raise ValueError("dataset sequence length exceeds max-model-len")
    split_attestation_binding = _install_split_attestation_environment(
        args, dataset
    )

    # Keep heavyweight GPU-only imports out of module import so dataset and
    # protocol validation remain unit-testable on machines without vLLM.
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        seed=args.sampling_seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=False,
        attention_config={"backend": args.backend},
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        min_tokens=dataset["output_len"],
        max_tokens=dataset["output_len"],
        ignore_eos=True,
        detokenize=False,
        seed=args.sampling_seed,
    )
    prompts = [
        {"prompt_token_ids": list(prompt)}
        for prompt in dataset["prompt_token_ids"]
    ]

    # These full-length calls are an explicit JIT-prime phase. Some vLLM
    # kernels are compiled lazily only after the graph reaches its longest
    # decode shape, so a shorter prefill-time compile cannot replace them.
    # Their outputs are retained for audit but are deliberately not part of
    # the within-process determinism gate below: compilation may change prime
    # calls, while every subsequent warmup and measured call must be identical.
    (
        prime_output_sha256,
        warmup_output_sha256,
        latencies_s,
        measured_output_sha256,
        token_ids,
        split_attestation,
    ) = _run_generation_phases(
        llm,
        prompts,
        sampling_params,
        batch_size=dataset["batch_size"],
        output_len=dataset["output_len"],
        prime_iters=args.prime_iters,
        warmup_iters=args.warmup_iters,
        num_iters=args.num_iters,
        after_prime=lambda: _verify_split_attestation(
            split_attestation_binding,
            backend=args.backend,
            dataset=dataset,
        ),
    )
    (
        split_activation_attestation,
        split_activation_attestation_sha256,
        split_activation_attestation_path,
    ) = split_attestation

    output_hashes = [*warmup_output_sha256, *measured_output_sha256]
    if len(set(output_hashes)) != 1:
        raise RuntimeError(
            "fixed greedy decoding produced different token sequences within "
            "one worker process; "
            f"prime_sha256={prime_output_sha256!r}; "
            f"warmup_sha256={warmup_output_sha256!r}; "
            f"measured_sha256={measured_output_sha256!r}"
        )
    output_token_ids_sha256 = measured_output_sha256[0]

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend_arg": args.backend,
        "case": dataset["case"],
        "batch_size": dataset["batch_size"],
        "input_len": dataset["input_len"],
        "output_len": dataset["output_len"],
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": args.dataset_sha256,
        "dataset_seed": dataset["seed"],
        "dataset_generation_protocol": dataset["generation_protocol"],
        "prompt_format": "token_ids",
        "skip_tokenizer_init": True,
        "sampling_seed": args.sampling_seed,
        "sampling_n": 1,
        "sampling_temperature": 0.0,
        "sampling_min_tokens": dataset["output_len"],
        "sampling_max_tokens": dataset["output_len"],
        "sampling_ignore_eos": True,
        "sampling_detokenize": False,
        "prime_iters": args.prime_iters,
        "warmup_iters": args.warmup_iters,
        "num_iters": args.num_iters,
        "timing_scope": TIMING_SCOPE,
        "vllm_engine_multiprocessing": True,
        "accuracy_prefix_len": ACCURACY_PREFIX_LEN,
        "split_activation_attestation": split_activation_attestation,
        "split_activation_attestation_path": split_activation_attestation_path,
        "split_activation_attestation_sha256": (
            split_activation_attestation_sha256
        ),
        "latencies_s": latencies_s,
        "avg_latency_s": statistics.fmean(latencies_s),
        "percentiles_s": {
            "50": _percentile(latencies_s, 50.0),
            "90": _percentile(latencies_s, 90.0),
        },
        "prime_output_sha256": prime_output_sha256,
        "warmup_output_sha256": warmup_output_sha256,
        "measured_output_sha256": measured_output_sha256,
        "output_token_ids_sha256": output_token_ids_sha256,
        "output_token_ids": token_ids,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_version": vllm.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{dataset['case']} {args.backend} p50_ms="
        f"{payload['percentiles_s']['50'] * 1000.0:.6f} "
        f"dataset_sha256={args.dataset_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
