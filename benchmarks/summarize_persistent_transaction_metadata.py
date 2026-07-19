"""Strictly validate and summarize R4-B persistent metadata paired trials."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_persistent_transaction_metadata import (
    BLOCK_SIZE,
    CASES,
    HEAD_DIM,
    MAX_PROFILE_ATTEMPTS,
    METADATA_PATHS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_WARPS,
    PROFILE_TIMING_SCOPE,
    WALL_TIMING_SCOPE,
)


REQUIRED_FIELDS = (
    "name",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "min_ms",
    "max_ms",
    "repeats",
    "date",
    "run_id",
    "op",
    "case",
    "metadata_path",
    "append_backend",
    "raw_dispatch",
    "decode_backend",
    "dtype",
    "device",
    "torch",
    "cuda",
    "git_commit",
    "num_layers",
    "batch_size",
    "context_tokens",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "warmup",
    "trial",
    "trial_count",
    "path_order",
    "seed",
    "p99_ms",
    "begin_host_p50_ms",
    "commit_host_p50_ms",
    "decode_tokens_per_second",
    "layer_steps_per_second",
    "kv_write_bytes_per_token",
    "cache_capacity_bytes",
    "final_seq_len",
    "final_used_blocks",
    "final_free_blocks",
    "final_request_blocks",
    "max_blocks",
    "allocation_count",
    "fresh_allocation_count",
    "reuse_count",
    "capacity_failure_count",
    "transaction_begin_count",
    "transaction_commit_count",
    "transaction_abort_count",
    "transaction_layer_write_count",
    "engine_completed_step_count",
    "engine_appended_token_count",
    "validated_invariants",
    "timing_scope",
    "wall_timer_cuda_events",
    "profile_timing_scope",
    "metadata_build_delta",
    "metadata_materialization_delta",
    "metadata_reuse_delta",
    "metadata_release_delta",
    "metadata_resident_before",
    "metadata_resident_after",
    "metadata_builds_per_token",
    "metadata_materializations_per_token",
    "metadata_reuses_per_token",
    "metadata_releases_per_token",
    "profile_steps",
    "profile_token_count",
    "profile_append_count",
    "profile_decode_count",
    "profile_append_cpu_ms_per_layer",
    "profile_item_count",
    "profile_local_scalar_dense_count",
    "profile_attempt_count",
    "rollback_repeats",
    "rollback_p50_ms",
    "rollback_blocks",
    "rollback_metadata_releases",
    "rollback_metadata_resident_after",
    "rollback_validated",
    "parity_steps",
    "parity_output_equal",
    "parity_cache_equal",
    "parity_state_equal",
    "parity_validated",
    "speedup_vs_materialized_p50",
)

_INT_FIELDS = {
    "repeats",
    "num_layers",
    "batch_size",
    "context_tokens",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "warmup",
    "trial",
    "trial_count",
    "seed",
    "kv_write_bytes_per_token",
    "cache_capacity_bytes",
    "final_seq_len",
    "final_used_blocks",
    "final_free_blocks",
    "final_request_blocks",
    "max_blocks",
    "allocation_count",
    "fresh_allocation_count",
    "reuse_count",
    "capacity_failure_count",
    "transaction_begin_count",
    "transaction_commit_count",
    "transaction_abort_count",
    "transaction_layer_write_count",
    "engine_completed_step_count",
    "engine_appended_token_count",
    "metadata_build_delta",
    "metadata_materialization_delta",
    "metadata_reuse_delta",
    "metadata_release_delta",
    "metadata_resident_before",
    "metadata_resident_after",
    "profile_steps",
    "profile_token_count",
    "profile_append_count",
    "profile_decode_count",
    "profile_item_count",
    "profile_local_scalar_dense_count",
    "profile_attempt_count",
    "rollback_repeats",
    "rollback_blocks",
    "rollback_metadata_releases",
    "rollback_metadata_resident_after",
    "parity_steps",
}

_FLOAT_FIELDS = {
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "begin_host_p50_ms",
    "commit_host_p50_ms",
    "decode_tokens_per_second",
    "layer_steps_per_second",
    "metadata_builds_per_token",
    "metadata_materializations_per_token",
    "metadata_reuses_per_token",
    "metadata_releases_per_token",
    "profile_append_cpu_ms_per_layer",
    "rollback_p50_ms",
    "speedup_vs_materialized_p50",
}

_TRUE_FIELDS = (
    "validated_invariants",
    "rollback_validated",
    "parity_output_equal",
    "parity_cache_equal",
    "parity_state_equal",
    "parity_validated",
)
_CASE_PATTERN = re.compile(r"^l(?P<layers>\d+)_b(?P<batch>\d+)_c(?P<context>\d+)$")


class MetadataValidationError(ValueError):
    """Raised when benchmark evidence violates the frozen R4-B contract."""


@dataclass(frozen=True)
class TrialPair:
    dtype: str
    case: str
    trial: int
    materialized: dict
    persistent: dict


def _integer(row, field):
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MetadataValidationError(f"{field} must be an integer") from exc
    return value


def _number(row, field, *, positive=False, nonnegative=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MetadataValidationError(f"{field} must be finite") from exc
    if not math.isfinite(value):
        raise MetadataValidationError(f"{field} must be finite")
    if positive and value <= 0.0:
        raise MetadataValidationError(f"{field} must be positive and finite")
    if nonnegative and value < 0.0:
        raise MetadataValidationError(f"{field} must be non-negative and finite")
    return value


def _close(actual, expected, *, rel=2e-3, abs_=1e-6):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)


def _validate_schema(rows):
    expected = set(REQUIRED_FIELDS)
    for index, row in enumerate(rows):
        fields = set(row)
        missing = sorted(expected - fields)
        unexpected = sorted(fields - expected)
        if missing:
            raise MetadataValidationError(
                f"row {index} missing columns: {', '.join(missing)}"
            )
        if unexpected:
            raise MetadataValidationError(
                f"row {index} has unexpected columns: {', '.join(unexpected)}"
            )


def _validate_row(row):
    for field in _INT_FIELDS:
        _integer(row, field)
    for field in _FLOAT_FIELDS:
        _number(row, field)
    for field in _TRUE_FIELDS:
        if row[field] != "True":
            raise MetadataValidationError(f"{field} must be True")
    if row["wall_timer_cuda_events"] != "False":
        raise MetadataValidationError("wall_timer_cuda_events must be False")
    if row["name"] != "persistent_transaction_metadata":
        raise MetadataValidationError("unexpected benchmark name")
    if row["op"] != "persistent_transaction_metadata":
        raise MetadataValidationError("unexpected op")
    if row["metadata_path"] not in METADATA_PATHS:
        raise MetadataValidationError("unsupported metadata_path")
    if row["append_backend"] != "fused_cuda":
        raise MetadataValidationError("append backend must remain fused CUDA")
    if row["raw_dispatch"] != "trusted":
        raise MetadataValidationError("raw fused dispatch must remain trusted")
    if row["decode_backend"] != "triton":
        raise MetadataValidationError("decode backend must be triton")
    if row["timing_scope"] != WALL_TIMING_SCOPE:
        raise MetadataValidationError("wall timing scope drifted")
    if row["profile_timing_scope"] != PROFILE_TIMING_SCOPE:
        raise MetadataValidationError("profile timing scope drifted")
    match = _CASE_PATTERN.match(row["case"])
    if match is None:
        raise MetadataValidationError("case name is malformed")
    expected_shape = tuple(int(match.group(key)) for key in ("layers", "batch", "context"))
    actual_shape = (
        _integer(row, "num_layers"),
        _integer(row, "batch_size"),
        _integer(row, "context_tokens"),
    )
    if actual_shape != expected_shape:
        raise MetadataValidationError("case name and shape disagree")
    if (
        _integer(row, "num_q_heads") != NUM_Q_HEADS
        or _integer(row, "num_kv_heads") != NUM_KV_HEADS
        or _integer(row, "head_dim") != HEAD_DIM
        or _integer(row, "block_size") != BLOCK_SIZE
        or _integer(row, "num_warps") != NUM_WARPS
    ):
        raise MetadataValidationError("frozen model/kernel shape drifted")

    minimum = _number(row, "min_ms", positive=True)
    p50 = _number(row, "p50_ms", positive=True)
    p90 = _number(row, "p90_ms", positive=True)
    p99 = _number(row, "p99_ms", positive=True)
    maximum = _number(row, "max_ms", positive=True)
    mean = _number(row, "mean_ms", positive=True)
    if not minimum <= p50 <= p90 <= p99 <= maximum:
        raise MetadataValidationError("latency percentile range is invalid")
    if not minimum <= mean <= maximum:
        raise MetadataValidationError("mean latency is outside min/max range")
    _number(row, "begin_host_p50_ms", positive=True)
    _number(row, "commit_host_p50_ms", positive=True)
    _number(row, "profile_append_cpu_ms_per_layer", positive=True)
    _number(row, "rollback_p50_ms", positive=True)

    repeats = _integer(row, "repeats")
    layers, batch, context = actual_shape
    if repeats <= 0:
        raise MetadataValidationError("repeats must be positive")
    expected_seq_len = context + repeats
    if _integer(row, "final_seq_len") != expected_seq_len:
        raise MetadataValidationError("final sequence trajectory mismatch")
    expected_max_blocks = batch * math.ceil((expected_seq_len + 1) / BLOCK_SIZE)
    expected_used = batch * math.ceil(expected_seq_len / BLOCK_SIZE)
    if _integer(row, "max_blocks") != expected_max_blocks:
        raise MetadataValidationError("max block accounting mismatch")
    if _integer(row, "final_used_blocks") != expected_used:
        raise MetadataValidationError("used block accounting mismatch")
    if _integer(row, "final_request_blocks") != expected_used:
        raise MetadataValidationError("request block accounting mismatch")
    if _integer(row, "final_free_blocks") != expected_max_blocks - expected_used:
        raise MetadataValidationError("free block accounting mismatch")
    if _integer(row, "allocation_count") != expected_used:
        raise MetadataValidationError("allocation count mismatch")
    if _integer(row, "fresh_allocation_count") != expected_used:
        raise MetadataValidationError("fresh allocation count mismatch")
    if _integer(row, "reuse_count") != 0:
        raise MetadataValidationError("unexpected allocator reuse")
    if _integer(row, "capacity_failure_count") != 0:
        raise MetadataValidationError("unexpected capacity failure")
    if _integer(row, "transaction_begin_count") != expected_seq_len:
        raise MetadataValidationError("transaction begin trajectory mismatch")
    if _integer(row, "transaction_commit_count") != expected_seq_len:
        raise MetadataValidationError("transaction commit trajectory mismatch")
    if _integer(row, "transaction_abort_count") != 0:
        raise MetadataValidationError("measured transaction unexpectedly aborted")
    if _integer(row, "transaction_layer_write_count") != expected_seq_len * layers:
        raise MetadataValidationError("transaction layer write trajectory mismatch")
    if _integer(row, "engine_completed_step_count") != repeats:
        raise MetadataValidationError("Engine completed-step trajectory mismatch")
    if _integer(row, "engine_appended_token_count") != repeats * batch:
        raise MetadataValidationError("Engine appended-token trajectory mismatch")

    expected_tps = batch * 1_000.0 / mean
    if not _close(_number(row, "decode_tokens_per_second", positive=True), expected_tps):
        raise MetadataValidationError("decode TPS derivation mismatch")
    expected_layer_tps = expected_tps * layers
    if not _close(_number(row, "layer_steps_per_second", positive=True), expected_layer_tps):
        raise MetadataValidationError("layer TPS derivation mismatch")
    dtype_bytes = 2
    expected_write_bytes = (
        batch * layers * 2 * NUM_KV_HEADS * HEAD_DIM * dtype_bytes
    )
    if _integer(row, "kv_write_bytes_per_token") != expected_write_bytes:
        raise MetadataValidationError("KV write byte derivation mismatch")
    expected_capacity = (
        layers
        * expected_max_blocks
        * 2
        * NUM_KV_HEADS
        * BLOCK_SIZE
        * HEAD_DIM
        * dtype_bytes
    )
    if _integer(row, "cache_capacity_bytes") != expected_capacity:
        raise MetadataValidationError("cache capacity byte derivation mismatch")

    path = row["metadata_path"]
    expected_views = 2 * layers + 2 if path == "materialized" else 1
    expected_reuses = 0 if path == "materialized" else layers
    expected_totals = {
        "metadata_build_delta": repeats,
        "metadata_materialization_delta": repeats * expected_views,
        "metadata_reuse_delta": repeats * expected_reuses,
        "metadata_release_delta": repeats,
    }
    for field, expected in expected_totals.items():
        if _integer(row, field) != expected:
            raise MetadataValidationError(
                f"{field} mismatch for {path}: expected {expected}"
            )
    expected_per_token = {
        "metadata_builds_per_token": 1.0,
        "metadata_materializations_per_token": float(expected_views),
        "metadata_reuses_per_token": float(expected_reuses),
        "metadata_releases_per_token": 1.0,
    }
    for field, expected in expected_per_token.items():
        if not _close(_number(row, field, nonnegative=True), expected, rel=0.0):
            raise MetadataValidationError(
                f"{field} mismatch for {path}: expected {expected}"
            )
    if _integer(row, "metadata_resident_before") != 0:
        raise MetadataValidationError("metadata resident-before must be zero")
    if _integer(row, "metadata_resident_after") != 0:
        raise MetadataValidationError("metadata resident-after must be zero")

    profile_steps = _integer(row, "profile_steps")
    if profile_steps <= 0 or _integer(row, "profile_token_count") != profile_steps:
        raise MetadataValidationError("profile token count mismatch")
    if _integer(row, "profile_append_count") != profile_steps * layers:
        raise MetadataValidationError("profile append count mismatch")
    if _integer(row, "profile_decode_count") != profile_steps * layers:
        raise MetadataValidationError("profile decode count mismatch")
    if _integer(row, "profile_item_count") != 0:
        raise MetadataValidationError("profile_item_count must be zero")
    if _integer(row, "profile_local_scalar_dense_count") != 0:
        raise MetadataValidationError(
            "profile_local_scalar_dense_count must be zero"
        )
    attempts = _integer(row, "profile_attempt_count")
    if not 1 <= attempts <= MAX_PROFILE_ATTEMPTS:
        raise MetadataValidationError("profile attempt count is out of bounds")

    rollback_repeats = _integer(row, "rollback_repeats")
    if rollback_repeats <= 0:
        raise MetadataValidationError("rollback repeats must be positive")
    if _integer(row, "rollback_blocks") != rollback_repeats * batch:
        raise MetadataValidationError("rollback block count mismatch")
    if _integer(row, "rollback_metadata_releases") != rollback_repeats:
        raise MetadataValidationError("rollback metadata release mismatch")
    if _integer(row, "rollback_metadata_resident_after") != 0:
        raise MetadataValidationError("rollback retained transaction metadata")
    if _integer(row, "parity_steps") <= 0:
        raise MetadataValidationError("parity steps must be positive")


def _pair_fields(row):
    excluded = {
        "date",
        "metadata_path",
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "begin_host_p50_ms",
        "commit_host_p50_ms",
        "decode_tokens_per_second",
        "layer_steps_per_second",
        "metadata_build_delta",
        "metadata_materialization_delta",
        "metadata_reuse_delta",
        "metadata_release_delta",
        "metadata_builds_per_token",
        "metadata_materializations_per_token",
        "metadata_reuses_per_token",
        "metadata_releases_per_token",
        "profile_append_cpu_ms_per_layer",
        "profile_attempt_count",
        "rollback_p50_ms",
        "speedup_vs_materialized_p50",
    }
    return {field: row[field] for field in REQUIRED_FIELDS if field not in excluded}


def validate_rows(
    rows,
    *,
    expected_trials,
    expected_cases=None,
    expected_dtypes=("float16", "bfloat16"),
):
    rows = [dict(row) for row in rows]
    if not rows:
        raise MetadataValidationError("CSV contains no rows")
    _validate_schema(rows)
    for row in rows:
        _validate_row(row)
    expected_trials = int(expected_trials)
    if expected_trials <= 0:
        raise MetadataValidationError("expected_trials must be positive")
    if expected_cases is None:
        expected_cases = tuple(CASES)
    expected_cases = tuple(expected_cases)
    expected_dtypes = tuple(expected_dtypes)
    if not expected_cases or not expected_dtypes:
        raise MetadataValidationError("expected matrix must be non-empty")

    expected_keys = {
        (dtype, case, trial, path)
        for dtype in expected_dtypes
        for case in expected_cases
        for trial in range(1, expected_trials + 1)
        for path in METADATA_PATHS
    }
    by_key = {}
    for row in rows:
        key = (
            row["dtype"],
            row["case"],
            _integer(row, "trial"),
            row["metadata_path"],
        )
        if key in by_key:
            raise MetadataValidationError(f"duplicate row: {key}")
        by_key[key] = row
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise MetadataValidationError(
            f"matrix incomplete; missing={missing[:4]}, extra={extra[:4]}"
        )

    run_ids = {row["run_id"] for row in rows}
    commits = {row["git_commit"] for row in rows}
    devices = {row["device"] for row in rows}
    runtimes = {(row["torch"], row["cuda"]) for row in rows}
    if len(run_ids) != 1 or len(commits) != 1 or len(devices) != 1 or len(runtimes) != 1:
        raise MetadataValidationError("run/device/runtime/commit metadata drifted")
    if any(_integer(row, "trial_count") != expected_trials for row in rows):
        raise MetadataValidationError("trial_count disagrees with expected trials")
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        if len({_integer(row, "seed") for row in trial_rows}) != 1:
            raise MetadataValidationError(
                "all dtype/case pairs must share the same seed per trial"
            )
        expected_order = "->".join(
            METADATA_PATHS if trial % 2 else reversed(METADATA_PATHS)
        )
        if {row["path_order"] for row in trial_rows} != {expected_order}:
            raise MetadataValidationError(
                "all dtype/case pairs must share the rotated path order"
            )

    pairs = []
    for dtype in expected_dtypes:
        for case in expected_cases:
            seeds = []
            for trial in range(1, expected_trials + 1):
                materialized = by_key[(dtype, case, trial, "materialized")]
                persistent = by_key[(dtype, case, trial, "persistent")]
                if _pair_fields(materialized) != _pair_fields(persistent):
                    raise MetadataValidationError(
                        f"paired inputs/trajectory drifted for {(dtype, case, trial)}"
                    )
                expected_order = "->".join(
                    METADATA_PATHS if trial % 2 else reversed(METADATA_PATHS)
                )
                if materialized["path_order"] != expected_order:
                    raise MetadataValidationError("trial path order did not rotate")
                seeds.append(_integer(materialized, "seed"))
                ratio = _number(materialized, "p50_ms", positive=True) / _number(
                    persistent, "p50_ms", positive=True
                )
                if not _close(
                    _number(
                        persistent,
                        "speedup_vs_materialized_p50",
                        positive=True,
                    ),
                    ratio,
                    rel=1e-3,
                ):
                    raise MetadataValidationError("persistent p50 speedup mismatch")
                if not _close(
                    _number(
                        materialized,
                        "speedup_vs_materialized_p50",
                        positive=True,
                    ),
                    1.0,
                    rel=1e-3,
                ):
                    raise MetadataValidationError("materialized speedup must be 1")
                pairs.append(
                    TrialPair(dtype, case, trial, materialized, persistent)
                )
            if any(right != left + 1 for left, right in zip(seeds, seeds[1:])):
                raise MetadataValidationError("trial seeds must increase by one")
    return pairs


def _median(values):
    return statistics.median(values)


def _geomean(values):
    if not values or any(value <= 0.0 for value in values):
        raise MetadataValidationError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _ratio(pair, numerator, denominator, field):
    return _number(getattr(pair, numerator), field, positive=True) / _number(
        getattr(pair, denominator), field, positive=True
    )


def aggregate(pairs):
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair.dtype, pair.case), []).append(pair)
    aggregates = []
    all_ratios = {
        "p50": [],
        "p90": [],
        "p99": [],
        "mean": [],
        "decode_tokens_per_second": [],
        "profile_append_cpu": [],
    }
    for (dtype, case), group in sorted(grouped.items()):
        ratios = {
            "p50": [_ratio(pair, "materialized", "persistent", "p50_ms") for pair in group],
            "p90": [_ratio(pair, "materialized", "persistent", "p90_ms") for pair in group],
            "p99": [_ratio(pair, "materialized", "persistent", "p99_ms") for pair in group],
            "mean": [_ratio(pair, "materialized", "persistent", "mean_ms") for pair in group],
            "decode_tokens_per_second": [
                _ratio(pair, "persistent", "materialized", "decode_tokens_per_second")
                for pair in group
            ],
            "profile_append_cpu": [
                _ratio(
                    pair,
                    "materialized",
                    "persistent",
                    "profile_append_cpu_ms_per_layer",
                )
                for pair in group
            ],
        }
        for key, values in ratios.items():
            all_ratios[key].extend(values)
        p50 = ratios["p50"]
        direction = (
            "persistent_faster"
            if min(p50) > 1.0
            else "persistent_slower"
            if max(p50) < 1.0
            else "unstable_crosses_1"
        )
        first = group[0]
        aggregates.append(
            {
                "dtype": dtype,
                "case": case,
                "ratios": {
                    key: {
                        "median": _median(values),
                        "min": min(values),
                        "max": max(values),
                    }
                    for key, values in ratios.items()
                },
                "direction": direction,
                "absolute": {
                    path: {
                        field: _median(
                            [_number(getattr(pair, path), field) for pair in group]
                        )
                        for field in (
                            "p50_ms",
                            "profile_append_cpu_ms_per_layer",
                            "metadata_builds_per_token",
                            "metadata_materializations_per_token",
                            "metadata_reuses_per_token",
                            "metadata_releases_per_token",
                            "metadata_resident_after",
                        )
                    }
                    for path in METADATA_PATHS
                },
                "num_layers": _integer(first.materialized, "num_layers"),
            }
        )
    overall = {key: _geomean(values) for key, values in all_ratios.items()}
    overall["groups_passing_min"] = sum(
        item["ratios"]["p50"]["min"] > 1.0 for item in aggregates
    )
    overall["group_count"] = len(aggregates)
    overall["keep_gate_passed"] = (
        overall["p50"] >= 1.05
        and overall["groups_passing_min"] == overall["group_count"] == 16
    )
    return aggregates, overall


def _range(cell):
    return f"{cell['median']:.4f}x [{cell['min']:.4f},{cell['max']:.4f}]"


def render_markdown(input_path, pairs, aggregates, overall):
    first = pairs[0].materialized
    attempts = [
        _integer(row, "profile_attempt_count")
        for pair in pairs
        for row in (pair.materialized, pair.persistent)
    ]
    lines = [
        "# Persistent Transaction Metadata Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(pairs) * 2}; paired trials: {len(pairs)}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Materialized and persistent paths used identical R4-A trusted fused CUDA/Triton math; only Cache-owned transaction metadata lifetime differed.",
        "- Matrix, rotating path order, seeds/inputs, exact parity, block/transaction/Engine trajectory, rollback, metadata build/reuse/release, CPU profiler ranges, and invariants were validated.",
        "- `views/token` counts Cache transaction-view materializations only; it does not count unrelated public result tensors elsewhere in the system.",
        f"- Profiler captures: {len(attempts)}; extra retries: {sum(value - 1 for value in attempts)}; maximum attempts per row: {max(attempts)}.",
        "- Complete-token latency is pure synchronized wall time with no CUDA events; the separate profiler is CPU-only.",
        "",
        "Ratios above 1 favor persistent metadata. Latency and append-CPU ratios are materialized/persistent; TPS is persistent/materialized.",
        "",
        "## Cross-trial Cases",
        "",
        "| dtype | case | p50 median [min,max] | p90 [min,max] | p99 [min,max] | TPS [min,max] | append CPU [min,max] | views mat/pers | reuses mat/pers | direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in aggregates:
        ratios = item["ratios"]
        absolute = item["absolute"]
        lines.append(
            "| {dtype} | {case} | {p50} | {p90} | {p99} | {tps} | {append} | {mat_views:.1f}/{pers_views:.1f} | {mat_reuse:.1f}/{pers_reuse:.1f} | {direction} |".format(
                dtype=item["dtype"],
                case=item["case"],
                p50=_range(ratios["p50"]),
                p90=_range(ratios["p90"]),
                p99=_range(ratios["p99"]),
                tps=_range(ratios["decode_tokens_per_second"]),
                append=_range(ratios["profile_append_cpu"]),
                mat_views=absolute["materialized"]["metadata_materializations_per_token"],
                pers_views=absolute["persistent"]["metadata_materializations_per_token"],
                mat_reuse=absolute["materialized"]["metadata_reuses_per_token"],
                pers_reuse=absolute["persistent"]["metadata_reuses_per_token"],
                direction=item["direction"],
            )
        )
    lines.extend(
        [
            "",
            "## Absolute Metadata Medians",
            "",
            "| dtype | case | path | token p50 ms | append CPU ms/layer | builds/token | views/token | reuses/token | releases/token | resident after |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in aggregates:
        for path in METADATA_PATHS:
            absolute = item["absolute"][path]
            lines.append(
                "| {dtype} | {case} | {path} | {p50:.6f} | {append:.6f} | {build:.1f} | {views:.1f} | {reuse:.1f} | {release:.1f} | {resident:.1f} |".format(
                    dtype=item["dtype"],
                    case=item["case"],
                    path=path,
                    p50=absolute["p50_ms"],
                    append=absolute["profile_append_cpu_ms_per_layer"],
                    build=absolute["metadata_builds_per_token"],
                    views=absolute["metadata_materializations_per_token"],
                    reuse=absolute["metadata_reuses_per_token"],
                    release=absolute["metadata_releases_per_token"],
                    resident=absolute["metadata_resident_after"],
                )
            )
    lines.extend(
        [
            "",
            "## Overall Geometric Mean",
            "",
            "| metric | persistent vs materialized |",
            "| --- | ---: |",
            f"| complete-token p50 | {overall['p50']:.4f}x |",
            f"| complete-token p90 | {overall['p90']:.4f}x |",
            f"| complete-token p99 | {overall['p99']:.4f}x |",
            f"| complete-token mean | {overall['mean']:.4f}x |",
            f"| decode tokens/s | {overall['decode_tokens_per_second']:.4f}x |",
            f"| profiler append CPU/layer | {overall['profile_append_cpu']:.4f}x |",
            "",
            "## Manual Keep Gate",
            "",
            "The validator verifies evidence integrity but does not turn performance noise into a release decision.",
            "",
            "- Required: overall p50 >= 1.05x and all 16 dtype/case groups have paired p50 min > 1.0x.",
            f"- Observed: overall p50 {overall['p50']:.4f}x; groups above 1 in every trial {overall['groups_passing_min']}/{overall['group_count']}.",
            f"- Gate status: `{'pass' if overall['keep_gate_passed'] else 'fail'}`.",
            "",
            "## Interpretation",
            "",
            "- `unstable_crosses_1` means the paired p50 direction changes across trials; do not claim a stable win.",
            "- Persistent metadata must build exactly once, reuse once per layer, release once, and leave zero resident bundles per token. Materialized recreates the legacy `2L+2` view boundary only inside this benchmark.",
            "- Cache transaction-view counts are not presented as a count of every Engine result tensor allocation.",
            "- Both paths must report zero `aten::item` and `_local_scalar_dense`; CPU profiler totals explain host overhead but are not release latency.",
            "- p50, p90, and p99 are reported with their full paired ranges; ratios below 1 are negative results.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument(
        "--expected-cases", nargs="+", default=list(CASES)
    )
    parser.add_argument(
        "--expected-dtypes",
        nargs="+",
        choices=["float16", "bfloat16"],
        default=["float16", "bfloat16"],
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        rows = read_csv(args.input)
        pairs = validate_rows(
            rows,
            expected_trials=args.expected_trials,
            expected_cases=args.expected_cases,
            expected_dtypes=args.expected_dtypes,
        )
        aggregates, overall = aggregate(pairs)
        markdown = render_markdown(args.input, pairs, aggregates, overall)
    except (OSError, MetadataValidationError) as exc:
        raise SystemExit(f"invalid persistent transaction metadata CSV: {exc}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(rows)} rows and wrote {output}")


if __name__ == "__main__":
    main()
