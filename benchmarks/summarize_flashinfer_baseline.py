"""Strictly validate and summarize the FlashInfer kernel baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_flashinfer_baseline import (
    BACKENDS,
    BLOCK_SIZE,
    DEFAULT_CASES,
    DTYPES,
    EXPECTED_CUDA_BINDINGS_VERSION,
    EXPECTED_CUDA_HOME_BASENAME,
    EXPECTED_CUDA_PATHFINDER_VERSION,
    EXPECTED_CUDA_PYTHON_VERSION,
    EXPECTED_CUDA_TOOLKIT_VERSION,
    EXPECTED_FLASHINFER_VERSION,
    EXPECTED_FLASHINFER_CUDA_ARCH_LIST,
    EXPECTED_NINJA_VERSION,
    EXPECTED_NVCC_RELEASE,
    EXPECTED_NVCC_VERSION,
    EXPECTED_PYTHON_MAJOR_MINOR,
    EXPECTED_TORCH_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRITON_VERSION,
    FLASHDEC_KV_LAYOUT,
    FLASHINFER_BASELINE_NAME,
    FLASHINFER_BACKEND,
    FLASHINFER_KV_LAYOUT,
    FORMAL_REPEATS,
    FORMAL_TRIALS,
    FORMAL_WARMUP,
    HEAD_DIM,
    LEGACY_FLASHINFER_BASELINE_NAMES,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    TIMING_SCOPE,
    _logical_workload_bytes,
)


CASE_SHAPES = {
    "small_b1_ctx128": (1, 32, 8, 128, 128),
    "medium_b16_ctx1024": (16, 32, 8, 128, 1024),
    "large_b16_ctx8192": (16, 32, 8, 128, 8192),
    "large_batch_b64_ctx4096": (64, 32, 8, 128, 4096),
}
FLASHDEC_BACKEND = "flashdec_triton"
EXTERNAL_BACKENDS = tuple(
    backend for backend in BACKENDS if backend != FLASHDEC_BACKEND
)
GLOBAL_IDENTITY_FIELDS = (
    "name",
    "op",
    "date",
    "device",
    "python",
    "torch",
    "triton",
    "cuda",
    "cuda_toolkit",
    "cuda_python",
    "cuda_bindings",
    "cuda_pathfinder",
    "ninja",
    "cuda_home",
    "cuda_home_realpath",
    "nvcc_release",
    "nvcc_version",
    "flashinfer_cuda_arch_list",
    "git_commit",
    "git_worktree_clean",
    "command",
    "flashinfer_version",
    "expected_flashinfer_version",
    "flashinfer_workspace_mib",
    "flashinfer_backend",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "flashdec_kv_layout",
    "flashinfer_kv_layout",
    "pos_encoding_mode",
    "sm_scale",
    "trial_count",
    "base_seed",
    "warmup",
    "repeats",
    "timing_scope",
)
PAIRED_INPUT_FIELDS = (
    "case",
    "dtype",
    "num_seqs",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "context_tokens",
    "min_seq_len",
    "max_seq_len",
    "block_size",
    "pages_per_seq",
    "num_pages",
    "flashdec_kv_layout",
    "flashinfer_kv_layout",
    "pos_encoding_mode",
    "sm_scale",
    "trial",
    "trial_count",
    "backend_order",
    "case_order",
    "dtype_order",
    "base_seed",
    "seed",
    "warmup",
    "repeats",
    "timing_scope",
    "page_table_digest",
    "reference_sample_size",
    "reference_validated",
    "cross_backend_validated",
    "rtol",
    "atol",
    "logical_workload_bytes",
    "validated_invariants",
)
REQUIRED_FIELDS = set(GLOBAL_IDENTITY_FIELDS) | set(PAIRED_INPUT_FIELDS) | {
    "backend",
    "flashinfer_use_tensor_cores",
    "max_abs_error_vs_reference",
    "max_abs_error_vs_flashdec",
    "max_tolerance_ratio_vs_reference",
    "max_tolerance_ratio_vs_flashdec",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "decode_tokens_per_second",
    "logical_workload_gbps_p50",
}


class FlashInferBaselineValidationError(ValueError):
    """Raised when baseline rows cannot support a paired public comparison."""


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FlashInferBaselineValidationError(
            f"{field} must be an integer"
        ) from exc


def _number(row, field, *, allow_zero=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FlashInferBaselineValidationError(
            f"{field} must be numeric"
        ) from exc
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise FlashInferBaselineValidationError(
            f"{field} must be {qualifier} and finite"
        )
    return value


def _positive_float(row, field):
    return _number(row, field, allow_zero=False)


def _rotate(values, offset):
    values = tuple(values)
    offset %= len(values)
    return values[offset:] + values[:offset]


def _validate_expected_values(name, values, *, known=None):
    values = tuple(values)
    if not values or len(values) != len(set(values)):
        raise FlashInferBaselineValidationError(
            f"{name} must be non-empty and unique"
        )
    if known is not None:
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise FlashInferBaselineValidationError(
                f"{name} contains unsupported values: {unknown}"
            )
    return values


def read_csv(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise FlashInferBaselineValidationError(
            "FlashInfer baseline CSV must contain rows"
        )
    return rows


def _validate_schema(rows):
    for index, row in enumerate(rows, start=1):
        columns = set(row)
        missing = sorted(REQUIRED_FIELDS - columns)
        unexpected = sorted(columns - REQUIRED_FIELDS)
        if missing or unexpected:
            raise FlashInferBaselineValidationError(
                f"CSV columns differ at row {index}; "
                f"missing={missing}, unexpected={unexpected}"
            )


def _validate_global_identity(
    rows,
    expected_trials,
    expected_warmup,
    expected_repeats,
):
    for field in GLOBAL_IDENTITY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise FlashInferBaselineValidationError(
                f"global field {field} is inconsistent: {sorted(values)}"
            )
    first = rows[0]
    accepted_names = (
        FLASHINFER_BASELINE_NAME,
        *LEGACY_FLASHINFER_BASELINE_NAMES,
    )
    if first["name"] not in accepted_names:
        raise FlashInferBaselineValidationError(
            f"name must be one of {accepted_names!r}, got {first['name']!r}"
        )
    expected_strings = {
        "op": "paged_decode_attention",
        "torch": EXPECTED_TORCH_VERSION,
        "triton": EXPECTED_TRITON_VERSION,
        "cuda": EXPECTED_TORCH_CUDA_VERSION,
        "cuda_toolkit": EXPECTED_CUDA_TOOLKIT_VERSION,
        "cuda_python": EXPECTED_CUDA_PYTHON_VERSION,
        "cuda_bindings": EXPECTED_CUDA_BINDINGS_VERSION,
        "cuda_pathfinder": EXPECTED_CUDA_PATHFINDER_VERSION,
        "ninja": EXPECTED_NINJA_VERSION,
        "nvcc_release": EXPECTED_NVCC_RELEASE,
        "nvcc_version": EXPECTED_NVCC_VERSION,
        "flashinfer_cuda_arch_list": EXPECTED_FLASHINFER_CUDA_ARCH_LIST,
        "flashinfer_version": EXPECTED_FLASHINFER_VERSION,
        "expected_flashinfer_version": EXPECTED_FLASHINFER_VERSION,
        "flashinfer_backend": FLASHINFER_BACKEND,
        "git_worktree_clean": "True",
        "flashdec_kv_layout": FLASHDEC_KV_LAYOUT,
        "flashinfer_kv_layout": FLASHINFER_KV_LAYOUT,
        "pos_encoding_mode": "NONE",
        "timing_scope": TIMING_SCOPE,
    }
    for field, expected in expected_strings.items():
        if first[field] != expected:
            raise FlashInferBaselineValidationError(
                f"{field} must be {expected!r}, got {first[field]!r}"
            )
    for field in (
        "date",
        "device",
        "python",
        "cuda_home",
        "cuda_home_realpath",
        "git_commit",
        "command",
    ):
        if not first[field].strip():
            raise FlashInferBaselineValidationError(
                f"global field {field} must be non-empty"
            )
    try:
        run_started_at = datetime.fromisoformat(first["date"])
    except ValueError as exc:
        raise FlashInferBaselineValidationError(
            "date must be an ISO-8601 timestamp"
        ) from exc
    if run_started_at.utcoffset() is None:
        raise FlashInferBaselineValidationError(
            "date must include a timezone offset"
        )
    if "run_flashinfer_baseline.py" not in first["command"]:
        raise FlashInferBaselineValidationError(
            "command must identify the FlashInfer baseline runner"
        )
    if "--require-clean" not in first["command"]:
        raise FlashInferBaselineValidationError(
            "command must include --require-clean"
        )
    if not first["python"].startswith(f"{EXPECTED_PYTHON_MAJOR_MINOR}."):
        raise FlashInferBaselineValidationError(
            f"python must be {EXPECTED_PYTHON_MAJOR_MINOR}.x"
        )
    for field in ("cuda_home", "cuda_home_realpath"):
        value = Path(first[field])
        if not value.is_absolute() or value.name != EXPECTED_CUDA_HOME_BASENAME:
            raise FlashInferBaselineValidationError(
                f"{field} must identify the frozen CUDA 12.8 toolkit"
            )
    expected_integers = {
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "trial_count": expected_trials,
        "flashinfer_workspace_mib": 128,
    }
    for field, expected in expected_integers.items():
        if _integer(first, field) != expected:
            raise FlashInferBaselineValidationError(
                f"global {field} mismatch: expected {expected}"
            )
    if _integer(first, "warmup") != expected_warmup:
        raise FlashInferBaselineValidationError(
            f"warmup mismatch: expected {expected_warmup}"
        )
    if _integer(first, "repeats") != expected_repeats:
        raise FlashInferBaselineValidationError(
            f"repeats mismatch: expected {expected_repeats}"
        )
    expected_scale = HEAD_DIM**-0.5
    if not math.isclose(
        _positive_float(first, "sm_scale"),
        expected_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FlashInferBaselineValidationError("sm_scale mismatch")


def _validate_shape(row, key):
    case = row["case"]
    try:
        num_seqs, num_q_heads, num_kv_heads, head_dim, context_tokens = (
            CASE_SHAPES[case]
        )
    except KeyError as exc:
        raise FlashInferBaselineValidationError(
            f"unknown case geometry: {case}"
        ) from exc
    expected_values = {
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context_tokens": context_tokens,
        "min_seq_len": context_tokens,
        "max_seq_len": context_tokens,
        "block_size": BLOCK_SIZE,
        "pages_per_seq": math.ceil(context_tokens / BLOCK_SIZE),
        "num_pages": num_seqs * math.ceil(context_tokens / BLOCK_SIZE),
        "reference_sample_size": min(2, num_seqs),
    }
    for field, expected in expected_values.items():
        if _integer(row, field) != expected:
            raise FlashInferBaselineValidationError(
                f"{field} shape mismatch for {key}: expected {expected}"
            )
    expected_bytes = _logical_workload_bytes(
        num_seqs=num_seqs,
        context_tokens=context_tokens,
        dtype_name=row["dtype"],
    )
    if _integer(row, "logical_workload_bytes") != expected_bytes:
        raise FlashInferBaselineValidationError(
            f"logical_workload_bytes mismatch for {key}"
        )


def _validate_backend(row, key):
    expected_tensor_core = {
        "flashdec_triton": "not_applicable",
        "flashinfer_fa2_cuda_core": "False",
        "flashinfer_fa2_tensor_core": "True",
    }
    if row["flashinfer_use_tensor_cores"] != expected_tensor_core[row["backend"]]:
        raise FlashInferBaselineValidationError(
            f"tensor-core backend identity mismatch for {key}"
        )


def _validate_correctness(row, key):
    for field in (
        "reference_validated",
        "cross_backend_validated",
        "validated_invariants",
    ):
        if row[field] != "True":
            raise FlashInferBaselineValidationError(
                f"{field} failed for {key}"
            )
    for field in (
        "max_abs_error_vs_reference",
        "max_abs_error_vs_flashdec",
    ):
        _number(row, field, allow_zero=True)
    for field in (
        "max_tolerance_ratio_vs_reference",
        "max_tolerance_ratio_vs_flashdec",
    ):
        ratio = _number(row, field, allow_zero=True)
        if ratio > 1.0 + 1e-7:
            raise FlashInferBaselineValidationError(
                f"{field} exceeds the recorded tolerance for {key}"
            )
    if row["backend"] == FLASHDEC_BACKEND and not math.isclose(
        _number(row, "max_abs_error_vs_flashdec", allow_zero=True),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FlashInferBaselineValidationError(
            f"FlashDec self-comparison error must be zero: {key}"
        )
    if row["backend"] == FLASHDEC_BACKEND and not math.isclose(
        _number(
            row,
            "max_tolerance_ratio_vs_flashdec",
            allow_zero=True,
        ),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FlashInferBaselineValidationError(
            f"FlashDec self-comparison tolerance ratio must be zero: {key}"
        )
    expected_tolerance = 3e-2 if row["dtype"] == "bfloat16" else 2e-2
    for field in ("rtol", "atol"):
        if not math.isclose(
            _positive_float(row, field),
            expected_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FlashInferBaselineValidationError(
                f"{field} mismatch for {key}"
            )


def _validate_metrics(row, key):
    latency = {
        field: _positive_float(row, field)
        for field in ("mean_ms", "p50_ms", "p90_ms", "p99_ms", "min_ms", "max_ms")
    }
    if not (
        latency["min_ms"]
        <= latency["p50_ms"]
        <= latency["p90_ms"]
        <= latency["p99_ms"]
        <= latency["max_ms"]
    ):
        raise FlashInferBaselineValidationError(
            f"latency percentile order is invalid: {key}"
        )
    if not latency["min_ms"] <= latency["mean_ms"] <= latency["max_ms"]:
        raise FlashInferBaselineValidationError(
            f"mean latency is outside min/max: {key}"
        )
    tps = _positive_float(row, "decode_tokens_per_second")
    workload_gbps = _positive_float(row, "logical_workload_gbps_p50")
    expected_tps = _integer(row, "num_seqs") * 1_000.0 / latency["p50_ms"]
    if not math.isclose(tps, expected_tps, rel_tol=5e-4, abs_tol=1e-3):
        raise FlashInferBaselineValidationError(f"decode TPS mismatch: {key}")
    expected_workload_gbps = (
        _integer(row, "logical_workload_bytes")
        / (latency["p50_ms"] * 1_000_000.0)
    )
    if not math.isclose(
        workload_gbps,
        expected_workload_gbps,
        rel_tol=5e-4,
        abs_tol=1e-4,
    ):
        raise FlashInferBaselineValidationError(
            f"logical workload GB/s mismatch: {key}"
        )


def _validate_trial_sequence(
    rows,
    *,
    expected_trials,
    expected_cases,
    expected_dtypes,
    expected_backends,
):
    base_seed = _integer(rows[0], "base_seed")
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        backend_order = _rotate(expected_backends, trial - 1)
        case_order = _rotate(expected_cases, trial - 1)
        dtype_order = _rotate(expected_dtypes, trial - 1)
        expected_metadata = {
            "backend_order": "->".join(backend_order),
            "case_order": "->".join(case_order),
            "dtype_order": "->".join(dtype_order),
        }
        for field, expected in expected_metadata.items():
            values = {row[field] for row in trial_rows}
            if values != {expected}:
                raise FlashInferBaselineValidationError(
                    f"trial {trial} {field} is invalid: {sorted(values)}"
                )
        seeds = {_integer(row, "seed") for row in trial_rows}
        expected_seed = base_seed + trial - 1
        if seeds != {expected_seed}:
            raise FlashInferBaselineValidationError(
                f"trial seeds must increase by one; trial {trial} has {sorted(seeds)}"
            )
        expected_sequence = [
            (dtype, case, backend)
            for dtype in dtype_order
            for case in case_order
            for backend in backend_order
        ]
        actual_sequence = [
            (row["dtype"], row["case"], row["backend"])
            for row in trial_rows
        ]
        if actual_sequence != expected_sequence:
            raise FlashInferBaselineValidationError(
                f"trial {trial} execution order does not match rotated metadata"
            )


def validate_rows(
    rows,
    *,
    expected_trials=FORMAL_TRIALS,
    expected_warmup=FORMAL_WARMUP,
    expected_repeats=FORMAL_REPEATS,
    expected_cases=DEFAULT_CASES,
    expected_dtypes=DTYPES,
    expected_backends=BACKENDS,
):
    """Validate the complete common-shape matrix and return its rows."""

    rows = list(rows)
    expected_trials = int(expected_trials)
    if expected_trials <= 0:
        raise FlashInferBaselineValidationError("expected_trials must be positive")
    expected_warmup = int(expected_warmup)
    expected_repeats = int(expected_repeats)
    if expected_warmup < 0:
        raise FlashInferBaselineValidationError(
            "expected_warmup must be non-negative"
        )
    if expected_repeats <= 0:
        raise FlashInferBaselineValidationError(
            "expected_repeats must be positive"
        )
    expected_cases = _validate_expected_values(
        "expected_cases", expected_cases, known=CASE_SHAPES
    )
    expected_dtypes = _validate_expected_values(
        "expected_dtypes", expected_dtypes, known=DTYPES
    )
    expected_backends = _validate_expected_values(
        "expected_backends", expected_backends, known=BACKENDS
    )
    if expected_backends != tuple(BACKENDS):
        raise FlashInferBaselineValidationError(
            "baseline comparisons require FlashDec and both fixed FlashInfer backends"
        )
    if not rows:
        raise FlashInferBaselineValidationError("rows must be non-empty")
    _validate_schema(rows)
    _validate_global_identity(
        rows,
        expected_trials,
        expected_warmup,
        expected_repeats,
    )

    expected_keys = {
        (dtype, case, trial, backend)
        for trial in range(1, expected_trials + 1)
        for dtype in expected_dtypes
        for case in expected_cases
        for backend in expected_backends
    }
    indexed = {}
    for row in rows:
        trial = _integer(row, "trial")
        key = (row["dtype"], row["case"], trial, row["backend"])
        if row["backend"] not in expected_backends:
            raise FlashInferBaselineValidationError(
                f"unsupported backend: {row['backend']}"
            )
        if key in indexed:
            raise FlashInferBaselineValidationError(f"duplicate row: {key}")
        indexed[key] = row
        _validate_shape(row, key)
        _validate_backend(row, key)
        _validate_correctness(row, key)
        _validate_metrics(row, key)
        digest = row["page_table_digest"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise FlashInferBaselineValidationError(
                f"page_table_digest must be lowercase SHA-256: {key}"
            )

    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        unexpected = sorted(set(indexed) - expected_keys)
        raise FlashInferBaselineValidationError(
            f"matrix incomplete; missing={missing}, unexpected={unexpected}"
        )

    for trial in range(1, expected_trials + 1):
        for dtype in expected_dtypes:
            for case in expected_cases:
                group = [
                    indexed[(dtype, case, trial, backend)]
                    for backend in expected_backends
                ]
                for field in PAIRED_INPUT_FIELDS:
                    values = {row[field] for row in group}
                    if len(values) != 1:
                        raise FlashInferBaselineValidationError(
                            "paired input differs for "
                            f"{(dtype, case, trial)}: {field}"
                        )

    _validate_trial_sequence(
        rows,
        expected_trials=expected_trials,
        expected_cases=expected_cases,
        expected_dtypes=expected_dtypes,
        expected_backends=expected_backends,
    )
    return rows


def _statistics(values):
    values = list(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(rows):
    """Return descriptive per-case ratios without applying a win/keep gate."""

    indexed = {
        (row["dtype"], row["case"], _integer(row, "trial"), row["backend"]): row
        for row in rows
    }
    groups = sorted(
        {(row["dtype"], row["case"]) for row in rows},
        key=lambda item: (
            DTYPES.index(item[0]),
            DEFAULT_CASES.index(item[1]),
        ),
    )
    result = []
    for dtype, case in groups:
        trials = sorted(
            {
                _integer(row, "trial")
                for row in rows
                if row["dtype"] == dtype and row["case"] == case
            }
        )
        for backend in EXTERNAL_BACKENDS:
            p50_ratios = []
            tps_ratios = []
            flashdec_p50 = []
            external_p50 = []
            flashdec_p90 = []
            external_p90 = []
            flashdec_p99 = []
            external_p99 = []
            flashdec_tps = []
            external_tps = []
            external_workload_gbps = []
            for trial in trials:
                baseline = indexed[(dtype, case, trial, FLASHDEC_BACKEND)]
                external = indexed[(dtype, case, trial, backend)]
                baseline_latency = _positive_float(baseline, "p50_ms")
                external_latency = _positive_float(external, "p50_ms")
                baseline_p90 = _positive_float(baseline, "p90_ms")
                external_p90_value = _positive_float(external, "p90_ms")
                baseline_p99 = _positive_float(baseline, "p99_ms")
                external_p99_value = _positive_float(external, "p99_ms")
                baseline_throughput = _positive_float(
                    baseline, "decode_tokens_per_second"
                )
                external_throughput = _positive_float(
                    external, "decode_tokens_per_second"
                )
                p50_ratios.append(baseline_latency / external_latency)
                tps_ratios.append(external_throughput / baseline_throughput)
                flashdec_p50.append(baseline_latency)
                external_p50.append(external_latency)
                flashdec_p90.append(baseline_p90)
                external_p90.append(external_p90_value)
                flashdec_p99.append(baseline_p99)
                external_p99.append(external_p99_value)
                flashdec_tps.append(baseline_throughput)
                external_tps.append(external_throughput)
                external_workload_gbps.append(
                    _positive_float(external, "logical_workload_gbps_p50")
                )
            result.append(
                {
                    "dtype": dtype,
                    "case": case,
                    "backend": backend,
                    "trials": len(trials),
                    "p50_ratio": _statistics(p50_ratios),
                    "tps_ratio": _statistics(tps_ratios),
                    "absolute": {
                        "flashdec_p50_ms": _statistics(flashdec_p50),
                        "external_p50_ms": _statistics(external_p50),
                        "flashdec_p90_ms": _statistics(flashdec_p90),
                        "external_p90_ms": _statistics(external_p90),
                        "flashdec_p99_ms": _statistics(flashdec_p99),
                        "external_p99_ms": _statistics(external_p99),
                        "flashdec_tps": _statistics(flashdec_tps),
                        "external_tps": _statistics(external_tps),
                        "external_logical_workload_gbps_p50": _statistics(
                            external_workload_gbps
                        ),
                    },
                }
            )
    return result


def _ratio_text(stats):
    return (
        f"{stats['median']:.4f}x "
        f"[{stats['min']:.4f},{stats['max']:.4f}]"
    )


def _metric_text(stats, digits=6):
    return (
        f"{stats['median']:.{digits}f} "
        f"[{stats['min']:.{digits}f},{stats['max']:.{digits}f}]"
    )


def render_markdown(input_path, rows, aggregates):
    first = rows[0]
    lines = [
        "# FlashInfer Paged-decode Baseline Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; trials: {first['trial_count']}.",
        f"- Device: {first['device']}.",
        f"- Run started: {first['date']}.",
        f"- Python/PyTorch/Triton/PyTorch CUDA: {first['python']} / {first['torch']} / {first['triton']} / {first['cuda']}.",
        f"- CUDA packages (toolkit/python/bindings/pathfinder): {first['cuda_toolkit']} / {first['cuda_python']} / {first['cuda_bindings']} / {first['cuda_pathfinder']}; Ninja: {first['ninja']}.",
        f"- CUDA_HOME: `{first['cuda_home']}` (realpath `{first['cuda_home_realpath']}`); NVCC: release {first['nvcc_release']} / V{first['nvcc_version']}.",
        f"- FlashInfer CUDA arch list: `{first['flashinfer_cuda_arch_list']}`.",
        f"- FlashInfer: `{first['flashinfer_version']}` (fixed expected version `{EXPECTED_FLASHINFER_VERSION}`).",
        f"- FlashInfer workspace: {first['flashinfer_workspace_mib']} MiB per wrapper.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Git worktree was clean when the runner started.",
        f"- Runner command: `{first['command']}`.",
        "- Common operation: paged single-token decode with 32 query heads, 8 KV heads, head dimension 128, page size 32, and FP16/BF16.",
        "- FlashInfer consumes its documented `HND` paged layout; FlashDec consumes its physical `token_major` layout. Both views share the same logical pages and page table; no layout conversion is inside timing.",
        "- CUDA events cover `run` only. Input construction, reference validation, FlashInfer plan/JIT, and synchronization setup are excluded.",
        "- Every row passed the sampled PyTorch reference check, full cross-backend parity check, page-table pairing, and invariant validation.",
        "",
        "## Paired Cross-trial Results",
        "",
        "| dtype | case | external backend | FlashDec p50 ms | external p50 ms | p50 ratio FlashDec/external | FlashDec tokens/s | external tokens/s | TPS ratio external/FlashDec | external logical workload GB/s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregates:
        absolute = item["absolute"]
        lines.append(
            f"| {item['dtype']} | {item['case']} | {item['backend']} | "
            f"{absolute['flashdec_p50_ms']['median']:.6f} | "
            f"{absolute['external_p50_ms']['median']:.6f} | "
            f"{_ratio_text(item['p50_ratio'])} | "
            f"{absolute['flashdec_tps']['median']:.3f} | "
            f"{absolute['external_tps']['median']:.3f} | "
            f"{_ratio_text(item['tps_ratio'])} | "
            f"{absolute['external_logical_workload_gbps_p50']['median']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Absolute Tail Percentiles Across Trials",
            "",
            "Each cell is `median [min,max]` in milliseconds across the paired trials.",
            "",
            "| dtype | case | external backend | FlashDec p90 ms | external p90 ms | FlashDec p99 ms | external p99 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in aggregates:
        absolute = item["absolute"]
        lines.append(
            f"| {item['dtype']} | {item['case']} | {item['backend']} | "
            f"{_metric_text(absolute['flashdec_p90_ms'])} | "
            f"{_metric_text(absolute['external_p90_ms'])} | "
            f"{_metric_text(absolute['flashdec_p99_ms'])} | "
            f"{_metric_text(absolute['external_p99_ms'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Ratios above 1 favor the named FlashInfer backend. Latency uses `FlashDec/external`; throughput uses `external/FlashDec`.",
            "- These ratios are descriptive evidence only; the comparison has no pass/fail performance or winner gate.",
            "- Logical workload GB/s counts each Q/K/V/output element once. It excludes metadata, caching, and implementation-specific rereads, so it is a shape-normalized workload proxy rather than measured DRAM bandwidth.",
            "- This is a common-shape, kernel-only comparison. It does not compare scheduler, KV ownership, prefix caching, multi-layer transactions, or end-to-end serving behavior.",
            "- `fa2_cuda_core` and `fa2_tensor_core` are two execution choices from the same pinned FlashInfer installation, not separate library versions.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmarks/results/flashinfer_paged_decode_baseline_summary.md",
    )
    parser.add_argument(
        "--expected-trials", type=int, default=FORMAL_TRIALS
    )
    parser.add_argument(
        "--expected-warmup", type=int, default=FORMAL_WARMUP
    )
    parser.add_argument(
        "--expected-repeats", type=int, default=FORMAL_REPEATS
    )
    parser.add_argument(
        "--expected-cases", nargs="+", default=list(DEFAULT_CASES)
    )
    parser.add_argument(
        "--expected-dtypes", nargs="+", default=list(DTYPES)
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        rows = read_csv(args.input)
        validate_rows(
            rows,
            expected_trials=args.expected_trials,
            expected_warmup=args.expected_warmup,
            expected_repeats=args.expected_repeats,
            expected_cases=args.expected_cases,
            expected_dtypes=args.expected_dtypes,
        )
        markdown = render_markdown(args.input, rows, aggregate(rows))
    except (OSError, ValueError, FlashInferBaselineValidationError) as exc:
        raise SystemExit(f"invalid FlashInfer baseline CSV: {exc}") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(rows)} rows and wrote {output}")


if __name__ == "__main__":
    main()
