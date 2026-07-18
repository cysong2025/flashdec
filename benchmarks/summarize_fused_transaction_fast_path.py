"""Validate checked/trusted fused transaction trials and summarize stability."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_fused_transaction_fast_path import (
    MAX_PROFILE_ATTEMPTS,
    PROFILE_TIMING_SCOPE,
    TRANSACTION_PATHS,
    WALL_TIMING_SCOPE,
)


DEFAULT_CASES = tuple(
    f"l{layers}_b{batch}_c{context}"
    for layers in (2, 4)
    for batch in (4, 16)
    for context in (128, 1024)
)
DEFAULT_DTYPES = ("float16", "bfloat16")
RATIO_METRICS = (
    "p50",
    "p90",
    "p99",
    "mean",
    "decode_tokens_per_second",
    "begin_host_p50",
    "commit_host_p50",
    "profile_append_cpu",
)

PAIR_IDENTITY_FIELDS = (
    "name",
    "op",
    "run_id",
    "case",
    "append_backend",
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
    "repeats",
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
    "profile_steps",
    "profile_token_count",
    "profile_append_count",
    "profile_decode_count",
    "parity_steps",
    "parity_output_equal",
    "parity_cache_equal",
    "parity_state_equal",
    "parity_validated",
    "rollback_repeats",
    "rollback_blocks",
    "rollback_validated",
    "validated_invariants",
    "timing_scope",
    "wall_timer_cuda_events",
    "profile_timing_scope",
)

GLOBAL_IDENTITY_FIELDS = (
    "name",
    "op",
    "run_id",
    "append_backend",
    "decode_backend",
    "device",
    "torch",
    "cuda",
    "git_commit",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "warmup",
    "trial_count",
    "timing_scope",
    "wall_timer_cuda_events",
    "profile_timing_scope",
)

REQUIRED_FIELDS = set(PAIR_IDENTITY_FIELDS) | {
    "date",
    "transaction_path",
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
    "profile_append_cpu_ms_per_layer",
    "profile_item_count",
    "profile_local_scalar_dense_count",
    "profile_attempt_count",
    "rollback_p50_ms",
    "speedup_vs_checked_p50",
}


class FastPathValidationError(ValueError):
    """Raised when fast-path rows cannot support a paired conclusion."""


@dataclass(frozen=True)
class PairedTrial:
    dtype: str
    case: str
    trial: int
    checked_row: dict[str, str]
    trusted_row: dict[str, str]

    def ratios(self):
        checked = self.checked_row
        trusted = self.trusted_row
        return {
            "p50": _positive_float(checked, "p50_ms")
            / _positive_float(trusted, "p50_ms"),
            "p90": _positive_float(checked, "p90_ms")
            / _positive_float(trusted, "p90_ms"),
            "p99": _positive_float(checked, "p99_ms")
            / _positive_float(trusted, "p99_ms"),
            "mean": _positive_float(checked, "mean_ms")
            / _positive_float(trusted, "mean_ms"),
            "decode_tokens_per_second": _positive_float(
                trusted, "decode_tokens_per_second"
            )
            / _positive_float(checked, "decode_tokens_per_second"),
            "begin_host_p50": _positive_float(checked, "begin_host_p50_ms")
            / _positive_float(trusted, "begin_host_p50_ms"),
            "commit_host_p50": _positive_float(checked, "commit_host_p50_ms")
            / _positive_float(trusted, "commit_host_p50_ms"),
            "profile_append_cpu": _positive_float(
                checked, "profile_append_cpu_ms_per_layer"
            )
            / _positive_float(trusted, "profile_append_cpu_ms_per_layer"),
        }


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FastPathValidationError(f"{field} must be an integer") from exc


def _float(row, field, *, allow_zero=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise FastPathValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise FastPathValidationError(f"{field} must be {qualifier} and finite")
    return value


def _positive_float(row, field):
    return _float(row, field, allow_zero=False)


def read_csv(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise FastPathValidationError("fast-path CSV must contain rows")
    return rows


def _validate_numeric_row(row, key):
    latency = {
        field: _positive_float(row, field)
        for field in ("mean_ms", "p50_ms", "p90_ms", "p99_ms", "min_ms", "max_ms")
    }
    if not (
        latency["min_ms"] <= latency["p50_ms"] <= latency["p90_ms"]
        <= latency["p99_ms"] <= latency["max_ms"]
    ):
        raise FastPathValidationError(f"latency percentile order is invalid: {key}")
    if not latency["min_ms"] <= latency["mean_ms"] <= latency["max_ms"]:
        raise FastPathValidationError(f"mean latency is outside min/max: {key}")
    for field in (
        "begin_host_p50_ms",
        "commit_host_p50_ms",
        "decode_tokens_per_second",
        "layer_steps_per_second",
        "profile_append_cpu_ms_per_layer",
        "speedup_vs_checked_p50",
    ):
        _positive_float(row, field)
    for field in ("profile_item_count", "profile_local_scalar_dense_count"):
        if _integer(row, field) < 0:
            raise FastPathValidationError(f"{field} must be non-negative: {key}")
    profile_attempt_count = _integer(row, "profile_attempt_count")
    if not 1 <= profile_attempt_count <= MAX_PROFILE_ATTEMPTS:
        raise FastPathValidationError(
            f"profile_attempt_count must be in [1, {MAX_PROFILE_ATTEMPTS}]: {key}"
        )


def validate_rows(
    rows,
    *,
    expected_trials=3,
    expected_cases=DEFAULT_CASES,
    expected_dtypes=DEFAULT_DTYPES,
):
    rows = list(rows)
    expected_trials = int(expected_trials)
    expected_cases = tuple(expected_cases)
    expected_dtypes = tuple(expected_dtypes)
    if expected_trials <= 0:
        raise FastPathValidationError("expected_trials must be positive")
    for name, values in (
        ("expected_cases", expected_cases),
        ("expected_dtypes", expected_dtypes),
    ):
        if not values or len(values) != len(set(values)):
            raise FastPathValidationError(f"{name} must be non-empty and unique")
    if not rows:
        raise FastPathValidationError("rows must be non-empty")
    columns = set(rows[0])
    missing = sorted(REQUIRED_FIELDS - columns)
    unexpected_columns = sorted(columns - REQUIRED_FIELDS)
    if missing or unexpected_columns:
        raise FastPathValidationError(
            f"CSV columns differ; missing={missing}, unexpected={unexpected_columns}"
        )
    for field in GLOBAL_IDENTITY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise FastPathValidationError(
                f"global field {field} is inconsistent: {sorted(values)}"
            )
    if rows[0]["timing_scope"] != WALL_TIMING_SCOPE:
        raise FastPathValidationError("wall timing scope does not exclude CUDA events")
    if rows[0]["profile_timing_scope"] != PROFILE_TIMING_SCOPE:
        raise FastPathValidationError("profile timing scope is invalid")
    if rows[0]["wall_timer_cuda_events"] != "False":
        raise FastPathValidationError("wall timer must not record CUDA events")
    if rows[0]["name"] != "fused_transaction_fast_path":
        raise FastPathValidationError("benchmark name is invalid")
    if rows[0]["op"] != "fused_transaction_fast_path":
        raise FastPathValidationError("benchmark op is invalid")
    if rows[0]["append_backend"] != "fused_cuda":
        raise FastPathValidationError("append backend must be fused_cuda")
    if rows[0]["decode_backend"] != "triton":
        raise FastPathValidationError("decode backend must be triton")

    expected_keys = {
        (dtype, case, trial, transaction_path)
        for dtype in expected_dtypes
        for case in expected_cases
        for trial in range(1, expected_trials + 1)
        for transaction_path in TRANSACTION_PATHS
    }
    indexed = {}
    for row in rows:
        trial = _integer(row, "trial")
        key = (row["dtype"], row["case"], trial, row["transaction_path"])
        if row["transaction_path"] not in TRANSACTION_PATHS:
            raise FastPathValidationError(
                f"unsupported transaction_path: {row['transaction_path']}"
            )
        if key in indexed:
            raise FastPathValidationError(f"duplicate row: {key}")
        indexed[key] = row
        if row["validated_invariants"] != "True":
            raise FastPathValidationError(f"invariant failure: {key}")
        for field in (
            "parity_output_equal",
            "parity_cache_equal",
            "parity_state_equal",
            "parity_validated",
            "rollback_validated",
        ):
            if row[field] != "True":
                raise FastPathValidationError(f"{field} failed: {key}")
        if _integer(row, "trial_count") != expected_trials:
            raise FastPathValidationError(f"trial_count mismatch: {key}")
        if _integer(row, "repeats") <= 0 or _integer(row, "profile_steps") <= 0:
            raise FastPathValidationError(f"measurement counts must be positive: {key}")
        if (
            _integer(row, "final_used_blocks")
            + _integer(row, "final_free_blocks")
            != _integer(row, "max_blocks")
        ):
            raise FastPathValidationError(f"block accounting failed: {key}")
        if _integer(row, "final_request_blocks") != _integer(
            row, "final_used_blocks"
        ):
            raise FastPathValidationError(f"request block accounting failed: {key}")

        layers = _integer(row, "num_layers")
        batch = _integer(row, "batch_size")
        context = _integer(row, "context_tokens")
        repeats = _integer(row, "repeats")
        profile_steps = _integer(row, "profile_steps")
        parity_steps = _integer(row, "parity_steps")
        rollback_repeats = _integer(row, "rollback_repeats")
        if layers not in (2, 4):
            raise FastPathValidationError(f"only l2/l4 cases are allowed: {key}")
        if row["case"] != f"l{layers}_b{batch}_c{context}":
            raise FastPathValidationError(f"case/shape identity mismatch: {key}")
        expected_seq_len = context + repeats
        expected_used = batch * math.ceil(expected_seq_len / _integer(row, "block_size"))
        if _integer(row, "final_seq_len") != expected_seq_len:
            raise FastPathValidationError(f"final_seq_len mismatch: {key}")
        if _integer(row, "final_used_blocks") != expected_used:
            raise FastPathValidationError(f"final used blocks mismatch: {key}")
        expected_max_blocks = batch * math.ceil(
            (context + repeats + 1) / _integer(row, "block_size")
        )
        if _integer(row, "max_blocks") != expected_max_blocks:
            raise FastPathValidationError(f"max_blocks mismatch: {key}")
        if _integer(row, "transaction_begin_count") != expected_seq_len:
            raise FastPathValidationError(f"transaction begin mismatch: {key}")
        if _integer(row, "transaction_commit_count") != expected_seq_len:
            raise FastPathValidationError(f"transaction commit mismatch: {key}")
        if _integer(row, "transaction_abort_count") != 0:
            raise FastPathValidationError(f"normal path contains aborts: {key}")
        if _integer(row, "transaction_layer_write_count") != expected_seq_len * layers:
            raise FastPathValidationError(f"layer write count mismatch: {key}")
        if _integer(row, "engine_completed_step_count") != repeats:
            raise FastPathValidationError(f"Engine step count mismatch: {key}")
        if _integer(row, "engine_appended_token_count") != repeats * batch:
            raise FastPathValidationError(f"Engine appended-token count mismatch: {key}")
        if _integer(row, "profile_token_count") != profile_steps:
            raise FastPathValidationError(f"profile token count mismatch: {key}")
        if _integer(row, "profile_append_count") != profile_steps * layers:
            raise FastPathValidationError(f"profile append count mismatch: {key}")
        if _integer(row, "profile_decode_count") != profile_steps * layers:
            raise FastPathValidationError(f"profile decode count mismatch: {key}")
        expected_scalar_syncs = (
            5 * profile_steps * layers
            if row["transaction_path"] == "checked"
            else 0
        )
        for field in ("profile_item_count", "profile_local_scalar_dense_count"):
            if _integer(row, field) != expected_scalar_syncs:
                raise FastPathValidationError(
                    f"{field} mismatch for {row['transaction_path']} path: {key}; "
                    f"expected {expected_scalar_syncs}"
                )
        if parity_steps <= 0:
            raise FastPathValidationError(f"parity evidence missing: {key}")
        if rollback_repeats <= 0 or _positive_float(row, "rollback_p50_ms") <= 0:
            raise FastPathValidationError(f"rollback evidence missing: {key}")
        if _integer(row, "rollback_blocks") != rollback_repeats * batch:
            raise FastPathValidationError(f"rollback block count mismatch: {key}")
        dtype_bytes = 2
        expected_write_bytes = batch * layers * 2 * 8 * 128 * dtype_bytes
        if _integer(row, "kv_write_bytes_per_token") != expected_write_bytes:
            raise FastPathValidationError(f"KV write byte count mismatch: {key}")
        expected_capacity_bytes = (
            layers
            * expected_max_blocks
            * 2
            * 8
            * _integer(row, "block_size")
            * 128
            * dtype_bytes
        )
        if _integer(row, "cache_capacity_bytes") != expected_capacity_bytes:
            raise FastPathValidationError(f"cache capacity byte count mismatch: {key}")
        _validate_numeric_row(row, key)
        expected_tps = batch * 1_000.0 / _positive_float(row, "mean_ms")
        if not math.isclose(
            _positive_float(row, "decode_tokens_per_second"),
            expected_tps,
            rel_tol=5e-4,
            abs_tol=1e-3,
        ):
            raise FastPathValidationError(f"decode TPS mismatch: {key}")
        if not math.isclose(
            _positive_float(row, "layer_steps_per_second"),
            expected_tps * layers,
            rel_tol=5e-4,
            abs_tol=1e-3,
        ):
            raise FastPathValidationError(f"layer TPS mismatch: {key}")

    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        unexpected = sorted(set(indexed) - expected_keys)
        raise FastPathValidationError(
            f"matrix incomplete; missing={missing}, unexpected={unexpected}"
        )

    pairs = []
    for dtype in expected_dtypes:
        for case in expected_cases:
            for trial in range(1, expected_trials + 1):
                checked = indexed[(dtype, case, trial, "checked")]
                trusted = indexed[(dtype, case, trial, "trusted")]
                for field in PAIR_IDENTITY_FIELDS:
                    if checked[field] != trusted[field]:
                        raise FastPathValidationError(
                            f"paired trajectory differs for {(dtype, case, trial)}: {field}"
                        )
                pair = PairedTrial(dtype, case, trial, checked, trusted)
                reported = _positive_float(trusted, "speedup_vs_checked_p50")
                if not math.isclose(
                    reported,
                    pair.ratios()["p50"],
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    raise FastPathValidationError(
                        f"reported p50 speedup mismatch: {(dtype, case, trial)}"
                    )
                if not math.isclose(
                    _positive_float(checked, "speedup_vs_checked_p50"),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    raise FastPathValidationError(
                        "checked p50 speedup baseline must be 1"
                    )
                pairs.append(pair)
    _validate_trial_sequence(rows, expected_trials)
    return pairs


def _validate_trial_sequence(rows, expected_trials):
    previous_seed = None
    previous_order = None
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        seeds = {_integer(row, "seed") for row in trial_rows}
        orders = {row["path_order"] for row in trial_rows}
        if len(seeds) != 1 or len(orders) != 1:
            raise FastPathValidationError(f"trial {trial} has inconsistent seed/order")
        seed = next(iter(seeds))
        order = tuple(next(iter(orders)).split("->"))
        if set(order) != set(TRANSACTION_PATHS) or len(order) != 2:
            raise FastPathValidationError(f"trial {trial} path order is invalid")
        if previous_seed is not None and seed != previous_seed + 1:
            raise FastPathValidationError("trial seeds must increase by one")
        if previous_order is not None and order == previous_order:
            raise FastPathValidationError("adjacent trials must reverse path order")
        previous_seed = seed
        previous_order = order


def geometric_mean(values):
    values = tuple(float(value) for value in values)
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def aggregate(pairs):
    pairs = list(pairs)
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair.dtype, pair.case), []).append(pair)
    rows = []
    absolute_fields = (
        "p50_ms",
        "begin_host_p50_ms",
        "commit_host_p50_ms",
        "profile_append_cpu_ms_per_layer",
        "profile_item_count",
        "profile_local_scalar_dense_count",
        "profile_attempt_count",
    )
    for (dtype, case), group in sorted(grouped.items()):
        values = {
            metric: [pair.ratios()[metric] for pair in group]
            for metric in RATIO_METRICS
        }
        p50 = values["p50"]
        direction = (
            "trusted_faster"
            if all(value > 1.0 for value in p50)
            else "checked_faster"
            if all(value < 1.0 for value in p50)
            else "unstable_crosses_1"
        )
        rows.append(
            {
                "dtype": dtype,
                "case": case,
                "direction": direction,
                "metrics": {
                    metric: {
                        "median": statistics.median(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "geometric_mean": geometric_mean(metric_values),
                    }
                    for metric, metric_values in values.items()
                },
                "absolute": {
                    transaction_path: {
                        field: statistics.median(
                            _float(
                                pair.checked_row
                                if transaction_path == "checked"
                                else pair.trusted_row,
                                field,
                                allow_zero=field
                                in (
                                    "profile_item_count",
                                    "profile_local_scalar_dense_count",
                                ),
                            )
                            for pair in group
                        )
                        for field in absolute_fields
                    }
                    for transaction_path in TRANSACTION_PATHS
                },
            }
        )
    overall = {
        metric: geometric_mean(pair.ratios()[metric] for pair in pairs)
        for metric in RATIO_METRICS
    }
    return rows, overall


def render_markdown(input_path, pairs, aggregates, overall):
    pairs = list(pairs)
    first = pairs[0].checked_row
    profile_attempts = [
        _integer(row, "profile_attempt_count")
        for pair in pairs
        for row in (pair.checked_row, pair.trusted_row)
    ]
    lines = [
        "# Fused Transaction Fast-path Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(pairs) * 2}; paired trials: {len(pairs)}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Checked and trusted paths used identical fused CUDA/Triton math; only the Cache-owned validation boundary differed.",
        "- Matrix, seed/order, pure-wall timing scope, block accounting, Engine/transaction trajectory, CPU profiler ranges, bounded capture attempts, and invariants were validated.",
        f"- Profiler capture attempts: {sum(profile_attempts)} total; extra retries: {sum(value - 1 for value in profile_attempts)}; maximum per row: {max(profile_attempts)}.",
        "- Complete-token latency is pure synchronized wall time with no CUDA events in its interval; the separate profiler is CPU-only and excludes device attribution.",
        "",
        "Ratios above 1 favor trusted dispatch. Latency and append-CPU ratios are checked/trusted; throughput is trusted/checked.",
        "",
        "## Cross-trial Cases",
        "",
        "| dtype | case | p50 median [min,max] | p90 | p99 [min,max] | TPS | append CPU | direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in aggregates:
        metrics = row["metrics"]
        lines.append(
            f"| {row['dtype']} | {row['case']} | "
            f"{metrics['p50']['median']:.4f}x "
            f"[{metrics['p50']['min']:.4f},{metrics['p50']['max']:.4f}] | "
            f"{metrics['p90']['median']:.4f}x | "
            f"{metrics['p99']['median']:.4f}x "
            f"[{metrics['p99']['min']:.4f},{metrics['p99']['max']:.4f}] | "
            f"{metrics['decode_tokens_per_second']['median']:.4f}x | "
            f"{metrics['profile_append_cpu']['median']:.4f}x | "
            f"{row['direction']} |"
        )
    lines.extend(
        [
            "",
            "## Absolute Attribution Medians",
            "",
            "| dtype | case | path | token p50 ms | begin host ms | commit host ms | append CPU ms/layer | item | local scalar | capture attempts |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        for transaction_path in TRANSACTION_PATHS:
            absolute = row["absolute"][transaction_path]
            lines.append(
                f"| {row['dtype']} | {row['case']} | {transaction_path} | "
                f"{absolute['p50_ms']:.6f} | "
                f"{absolute['begin_host_p50_ms']:.6f} | "
                f"{absolute['commit_host_p50_ms']:.6f} | "
                f"{absolute['profile_append_cpu_ms_per_layer']:.6f} | "
                f"{absolute['profile_item_count']:.0f} | "
                f"{absolute['profile_local_scalar_dense_count']:.0f} | "
                f"{absolute['profile_attempt_count']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Overall Geometric Mean",
            "",
            "| metric | trusted vs checked |",
            "| --- | ---: |",
            f"| complete-token p50 | {overall['p50']:.4f}x |",
            f"| complete-token p90 | {overall['p90']:.4f}x |",
            f"| complete-token p99 | {overall['p99']:.4f}x |",
            f"| complete-token mean | {overall['mean']:.4f}x |",
            f"| decode tokens/s | {overall['decode_tokens_per_second']:.4f}x |",
            f"| begin host p50 | {overall['begin_host_p50']:.4f}x |",
            f"| commit host p50 | {overall['commit_host_p50']:.4f}x |",
            f"| profiler append CPU/layer | {overall['profile_append_cpu']:.4f}x |",
            "",
            "## Interpretation",
            "",
            "- `unstable_crosses_1` means the complete-token p50 direction changes across trials; do not claim a stable win.",
            "- The trusted path is accepted only for Cache-owned transaction metadata; public raw CUDA calls retain checked validation.",
            "- Profiler inclusive CPU totals, scalar-extraction counts, and bounded recapture attempts explain the removed synchronization boundary but are not release latency; device attribution is intentionally excluded.",
            "- p99 must be reported with its full range, and a ratio below 1 remains a negative result.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmarks/results/fused_transaction_fast_path_summary.md",
    )
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--expected-cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--expected-dtypes", nargs="+", default=list(DEFAULT_DTYPES))
    return parser.parse_args()


def main():
    args = parse_args()
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
    except (OSError, ValueError, FastPathValidationError) as exc:
        raise SystemExit(f"invalid fused transaction fast-path CSV: {exc}") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(pairs) * 2} rows and wrote {output}")


if __name__ == "__main__":
    main()
