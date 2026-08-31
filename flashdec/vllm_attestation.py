"""Fail-closed evidence for FlashDec split activation inside vLLM.

This module is intentionally free of torch and vLLM imports so benchmark
workers and summarizers can validate engine-process evidence without loading a
GPU runtime.
"""

from __future__ import annotations

import json
import re
from typing import Any


SPLIT_ATTESTATION_SCHEMA_VERSION = 1
SPLIT_ATTESTATION_BACKEND = "CUSTOM"
SPLIT_ATTESTATION_ENV = {
    "path": "FLASHDEC_VLLM_SPLIT_ATTESTATION_PATH",
    "nonce": "FLASHDEC_VLLM_SPLIT_ATTESTATION_NONCE",
    "case": "FLASHDEC_VLLM_SPLIT_ATTESTATION_CASE",
    "trial": "FLASHDEC_VLLM_SPLIT_ATTESTATION_TRIAL",
    "dataset_sha256": "FLASHDEC_VLLM_SPLIT_ATTESTATION_DATASET_SHA256",
    "git_commit": "FLASHDEC_VLLM_SPLIT_ATTESTATION_GIT_COMMIT",
}
SPLIT_ATTESTATION_FIELDS = frozenset(
    {
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
)
VALID_SPLIT_COUNTS = frozenset({2, 4, 8, 16})
VALID_BLOCK_SIZES = frozenset({16, 32})
VALID_DTYPES = frozenset({"float16", "bfloat16"})
SPLIT_ATTESTATION_CSV_FIELDS = (
    "split_attestation_json",
    "split_attestation_path",
    "split_attestation_sha256",
    "split_attestation_schema_version",
    "split_attestation_nonce",
    "split_attestation_engine_pid",
    "split_attestation_backend",
    "split_attestation_case",
    "split_attestation_trial",
    "split_attestation_dataset_sha256",
    "split_attestation_git_commit",
    "split_attestation_max_seq_len",
    "split_attestation_logical_blocks",
    "split_attestation_num_reqs",
    "split_attestation_num_splits",
    "split_attestation_num_q_heads",
    "split_attestation_num_kv_heads",
    "split_attestation_head_dim",
    "split_attestation_block_size",
    "split_attestation_query_dtype",
    "split_attestation_kv_cache_dtype",
    "split_attestation_cuda_graph_capture",
)


def canonical_attestation_bytes(payload: dict[str, Any]) -> bytes:
    """Return the only accepted on-disk encoding for an attestation."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _json_int(payload: dict[str, Any], field: str, *, positive: bool = True) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"split attestation {field} must be a JSON integer")
    if positive and value <= 0:
        raise ValueError(f"split attestation {field} must be positive")
    return value


def validate_split_attestation(
    payload: Any,
    *,
    expected_nonce: str | None = None,
    expected_case: str | None = None,
    expected_trial: int | None = None,
    expected_dataset_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_min_seq_len: int | None = None,
    expected_num_reqs: int | None = None,
) -> dict[str, Any]:
    """Validate marker structure and optional worker/runner bindings."""

    if not isinstance(payload, dict):
        raise ValueError("split attestation must be a JSON object")
    if set(payload) != SPLIT_ATTESTATION_FIELDS:
        missing = sorted(SPLIT_ATTESTATION_FIELDS - set(payload))
        extra = sorted(set(payload) - SPLIT_ATTESTATION_FIELDS)
        raise ValueError(
            "split attestation fields differ from schema: "
            f"missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != SPLIT_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported split attestation schema_version")
    if payload["backend"] != SPLIT_ATTESTATION_BACKEND:
        raise ValueError("split attestation backend must be CUSTOM")

    nonce = payload["nonce"]
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise ValueError("split attestation nonce must be 64 lowercase hex digits")
    case = payload["case"]
    if not isinstance(case, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case):
        raise ValueError("split attestation case is invalid")
    dataset_sha256 = payload["dataset_sha256"]
    if not isinstance(dataset_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", dataset_sha256
    ):
        raise ValueError("split attestation dataset_sha256 is invalid")
    git_commit = payload["git_commit"]
    if not isinstance(git_commit, str) or not re.fullmatch(
        r"[0-9a-f]{7,64}", git_commit
    ):
        raise ValueError("split attestation git_commit is invalid")

    _json_int(payload, "trial")
    _json_int(payload, "engine_pid")
    max_seq_len = _json_int(payload, "max_seq_len")
    logical_blocks = _json_int(payload, "logical_blocks")
    num_reqs = _json_int(payload, "num_reqs")
    num_splits = _json_int(payload, "num_splits")
    num_q_heads = _json_int(payload, "num_q_heads")
    num_kv_heads = _json_int(payload, "num_kv_heads")
    head_dim = _json_int(payload, "head_dim")
    block_size = _json_int(payload, "block_size")
    if num_splits not in VALID_SPLIT_COUNTS:
        raise ValueError("split attestation did not observe a multi-split launch")
    if block_size not in VALID_BLOCK_SIZES:
        raise ValueError("split attestation block_size is unsupported")
    if logical_blocks != (max_seq_len + block_size - 1) // block_size:
        raise ValueError("split attestation logical_blocks is inconsistent")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("split attestation head grouping is inconsistent")
    if head_dim != 128:
        raise ValueError("split attestation head_dim is unsupported")
    for field in ("query_dtype", "kv_cache_dtype"):
        if payload[field] not in VALID_DTYPES:
            raise ValueError(f"split attestation {field} is unsupported")
    if payload["query_dtype"] != payload["kv_cache_dtype"]:
        raise ValueError("split attestation query/cache dtypes differ")
    if not isinstance(payload["cuda_graph_capture"], bool):
        raise ValueError("split attestation cuda_graph_capture must be boolean")

    expected_values = {
        "nonce": expected_nonce,
        "case": expected_case,
        "trial": expected_trial,
        "dataset_sha256": expected_dataset_sha256,
        "git_commit": expected_git_commit,
        "num_reqs": expected_num_reqs,
    }
    for field, expected in expected_values.items():
        if expected is not None and payload[field] != expected:
            raise ValueError(
                f"split attestation {field} binding differs: "
                f"expected {expected!r}, got {payload[field]!r}"
            )
    if expected_min_seq_len is not None and max_seq_len < expected_min_seq_len:
        raise ValueError(
            "split attestation max_seq_len is below the benchmark input length"
        )
    return payload
