"""Validate paired multi-layer Engine trials and write a stability summary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


DEFAULT_CASES = tuple(
    f"l{layers}_b{batch}_c{context}"
    for layers in (1, 2, 4)
    for batch in (4, 16)
    for context in (128, 1024)
)
DEFAULT_DTYPES = ("float16", "bfloat16")
REQUIRED_BACKENDS = ("torch", "fused_cuda")
RATIO_METRICS = (
    "p50",
    "p90",
    "p99",
    "mean",
    "device_p50",
    "layer_device_p50",
    "decode_tokens_per_second",
    "profile_append_device",
    "profile_decode_device",
    "profile_cuda_events",
)

PAIR_IDENTITY_FIELDS = (
    "name",
    "op",
    "case",
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
    "backend_order",
    "seed",
    "repeats",
    "kv_write_bytes_per_token",
    "cache_capacity_bytes",
    "final_seq_len",
    "final_used_blocks",
    "final_free_blocks",
    "max_blocks",
    "transaction_begin_count",
    "transaction_commit_count",
    "transaction_abort_count",
    "transaction_layer_write_count",
    "profile_steps",
    "profile_token_count",
    "profile_append_count",
    "profile_decode_count",
    "rollback_repeats",
    "rollback_blocks",
    "rollback_validated",
    "validated_invariants",
    "timing_scope",
    "profile_timing_scope",
)

GLOBAL_IDENTITY_FIELDS = (
    "op",
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
    "profile_timing_scope",
)

REQUIRED_FIELDS = set(PAIR_IDENTITY_FIELDS) | {
    "append_backend",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "device_p50_ms",
    "layer_device_p50_ms",
    "begin_host_mean_ms",
    "commit_host_mean_ms",
    "decode_tokens_per_second",
    "layer_steps_per_second",
    "profile_cuda_event_count",
    "profile_append_device_ms_per_layer",
    "profile_decode_device_ms_per_layer",
    "rollback_p50_ms",
    "speedup_vs_torch_p50",
}


class MultiLayerValidationError(ValueError):
    """Raised when multi-layer rows cannot support paired conclusions."""


@dataclass(frozen=True)
class PairedTrial:
    dtype: str
    case: str
    trial: int
    torch_row: dict[str, str]
    fused_row: dict[str, str]

    def ratios(self):
        torch = self.torch_row
        fused = self.fused_row
        return {
            "p50": _positive_float(torch, "p50_ms")
            / _positive_float(fused, "p50_ms"),
            "p90": _positive_float(torch, "p90_ms")
            / _positive_float(fused, "p90_ms"),
            "p99": _positive_float(torch, "p99_ms")
            / _positive_float(fused, "p99_ms"),
            "mean": _positive_float(torch, "mean_ms")
            / _positive_float(fused, "mean_ms"),
            "device_p50": _positive_float(torch, "device_p50_ms")
            / _positive_float(fused, "device_p50_ms"),
            "layer_device_p50": _positive_float(
                torch, "layer_device_p50_ms"
            )
            / _positive_float(fused, "layer_device_p50_ms"),
            "decode_tokens_per_second": _positive_float(
                fused, "decode_tokens_per_second"
            )
            / _positive_float(torch, "decode_tokens_per_second"),
            "profile_append_device": _positive_float(
                torch, "profile_append_device_ms_per_layer"
            )
            / _positive_float(fused, "profile_append_device_ms_per_layer"),
            "profile_decode_device": _positive_float(
                torch, "profile_decode_device_ms_per_layer"
            )
            / _positive_float(fused, "profile_decode_device_ms_per_layer"),
            "profile_cuda_events": _positive_float(
                torch, "profile_cuda_event_count"
            )
            / _positive_float(fused, "profile_cuda_event_count"),
        }


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MultiLayerValidationError(f"{field} must be an integer") from exc


def _float(row, field, *, allow_zero=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise MultiLayerValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MultiLayerValidationError(f"{field} must be {qualifier} and finite")
    return value


def _positive_float(row, field):
    return _float(row, field, allow_zero=False)


def read_csv(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise MultiLayerValidationError("multi-layer CSV must contain rows")
    return rows


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
        raise MultiLayerValidationError("expected_trials must be positive")
    for name, values in (
        ("expected_cases", expected_cases),
        ("expected_dtypes", expected_dtypes),
    ):
        if not values or len(set(values)) != len(values):
            raise MultiLayerValidationError(f"{name} must be non-empty and unique")
    if not rows:
        raise MultiLayerValidationError("rows must be non-empty")
    missing = sorted(REQUIRED_FIELDS - set(rows[0]))
    if missing:
        raise MultiLayerValidationError(
            f"missing required columns: {', '.join(missing)}"
        )
    for field in GLOBAL_IDENTITY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise MultiLayerValidationError(
                f"global field {field} is inconsistent: {sorted(values)}"
            )

    expected_keys = {
        (dtype, case, trial, backend)
        for dtype in expected_dtypes
        for case in expected_cases
        for trial in range(1, expected_trials + 1)
        for backend in REQUIRED_BACKENDS
    }
    indexed = {}
    for row in rows:
        trial = _integer(row, "trial")
        key = (row["dtype"], row["case"], trial, row["append_backend"])
        if key in indexed:
            raise MultiLayerValidationError(f"duplicate row: {key}")
        indexed[key] = row
        if row["validated_invariants"] != "True":
            raise MultiLayerValidationError(f"invariant failure: {key}")
        if row["rollback_validated"] != "True":
            raise MultiLayerValidationError(f"rollback validation failure: {key}")
        if _integer(row, "trial_count") != expected_trials:
            raise MultiLayerValidationError(f"trial_count mismatch: {key}")
        if (
            _integer(row, "final_used_blocks")
            + _integer(row, "final_free_blocks")
            != _integer(row, "max_blocks")
        ):
            raise MultiLayerValidationError(f"block accounting failed: {key}")

        layers = _integer(row, "num_layers")
        batch = _integer(row, "batch_size")
        context = _integer(row, "context_tokens")
        repeats = _integer(row, "repeats")
        profile_steps = _integer(row, "profile_steps")
        if row["case"] != f"l{layers}_b{batch}_c{context}":
            raise MultiLayerValidationError(f"case/shape identity mismatch: {key}")
        if _integer(row, "final_seq_len") != context + repeats:
            raise MultiLayerValidationError(f"final_seq_len mismatch: {key}")
        if _integer(row, "transaction_begin_count") != context + repeats:
            raise MultiLayerValidationError(f"transaction begin mismatch: {key}")
        if _integer(row, "transaction_commit_count") != context + repeats:
            raise MultiLayerValidationError(f"transaction commit mismatch: {key}")
        if _integer(row, "transaction_abort_count") != 0:
            raise MultiLayerValidationError(f"normal path contains aborts: {key}")
        if _integer(row, "transaction_layer_write_count") != (
            context + repeats
        ) * layers:
            raise MultiLayerValidationError(f"layer write count mismatch: {key}")
        if _integer(row, "profile_token_count") != profile_steps:
            raise MultiLayerValidationError(f"profile token count mismatch: {key}")
        if _integer(row, "profile_append_count") != profile_steps * layers:
            raise MultiLayerValidationError(f"profile append count mismatch: {key}")
        if _integer(row, "profile_decode_count") != profile_steps * layers:
            raise MultiLayerValidationError(f"profile decode count mismatch: {key}")
        for field in (
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "device_p50_ms",
            "layer_device_p50_ms",
            "begin_host_mean_ms",
            "commit_host_mean_ms",
            "decode_tokens_per_second",
            "layer_steps_per_second",
            "profile_cuda_event_count",
            "profile_append_device_ms_per_layer",
            "profile_decode_device_ms_per_layer",
        ):
            _positive_float(row, field)
        rollback_repeats = _integer(row, "rollback_repeats")
        rollback_p50 = _float(row, "rollback_p50_ms", allow_zero=True)
        if layers == 1 and (rollback_repeats != 0 or rollback_p50 != 0.0):
            raise MultiLayerValidationError(f"single-layer rollback must be N/A: {key}")
        if layers > 1 and (rollback_repeats <= 0 or rollback_p50 <= 0.0):
            raise MultiLayerValidationError(f"multi-layer rollback evidence missing: {key}")

    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        unexpected = sorted(set(indexed) - expected_keys)
        raise MultiLayerValidationError(
            f"matrix incomplete; missing={missing}, unexpected={unexpected}"
        )

    pairs = []
    for dtype in expected_dtypes:
        for case in expected_cases:
            for trial in range(1, expected_trials + 1):
                torch_row = indexed[(dtype, case, trial, "torch")]
                fused_row = indexed[(dtype, case, trial, "fused_cuda")]
                for field in PAIR_IDENTITY_FIELDS:
                    if torch_row[field] != fused_row[field]:
                        raise MultiLayerValidationError(
                            f"paired trajectory differs for {(dtype, case, trial)}: {field}"
                        )
                pair = PairedTrial(dtype, case, trial, torch_row, fused_row)
                reported = _positive_float(
                    fused_row, "speedup_vs_torch_p50"
                )
                if not math.isclose(
                    reported,
                    pair.ratios()["p50"],
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    raise MultiLayerValidationError(
                        f"reported p50 speedup mismatch: {(dtype, case, trial)}"
                    )
                if not math.isclose(
                    _positive_float(torch_row, "speedup_vs_torch_p50"),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    raise MultiLayerValidationError("torch speedup baseline must be 1")
                pairs.append(pair)
    _validate_trial_sequence(rows, expected_trials)
    return pairs


def _validate_trial_sequence(rows, expected_trials):
    previous_seed = None
    previous_order = None
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        seeds = {_integer(row, "seed") for row in trial_rows}
        orders = {row["backend_order"] for row in trial_rows}
        if len(seeds) != 1 or len(orders) != 1:
            raise MultiLayerValidationError(
                f"trial {trial} has inconsistent seed/order"
            )
        seed = next(iter(seeds))
        order = tuple(next(iter(orders)).split("->"))
        if set(order) != set(REQUIRED_BACKENDS) or len(order) != 2:
            raise MultiLayerValidationError(f"trial {trial} backend order is invalid")
        if previous_seed is not None and seed != previous_seed + 1:
            raise MultiLayerValidationError("trial seeds must increase by one")
        if previous_order is not None and order == previous_order:
            raise MultiLayerValidationError("adjacent trials must reverse backend order")
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
    for (dtype, case), group in sorted(grouped.items()):
        values = {
            metric: [pair.ratios()[metric] for pair in group]
            for metric in RATIO_METRICS
        }
        p50 = values["p50"]
        direction = (
            "fused_faster"
            if all(value > 1.0 for value in p50)
            else "torch_faster"
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
                    backend: {
                        field: statistics.median(
                            _float(
                                pair.torch_row
                                if backend == "torch"
                                else pair.fused_row,
                                field,
                                allow_zero=field == "rollback_p50_ms",
                            )
                            for pair in group
                        )
                        for field in (
                            "p50_ms",
                            "profile_append_device_ms_per_layer",
                            "profile_decode_device_ms_per_layer",
                            "profile_cuda_event_count",
                            "begin_host_mean_ms",
                            "commit_host_mean_ms",
                            "rollback_p50_ms",
                        )
                    }
                    for backend in REQUIRED_BACKENDS
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
    first = pairs[0].torch_row
    lines = [
        "# Multi-layer DecodeEngine Trial Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(pairs) * 2}; paired trials: {len(pairs)}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Matrix, pair trajectory, block accounting, transaction counts, profiler ranges, rollback evidence, seed, and backend order were validated.",
        "- Non-instrumented wall latency is the performance source; profiler fields are attribution-only.",
        "",
        "Ratios above 1 favor fused CUDA. Latency ratios are torch/fused; throughput is fused/torch; CUDA-event ratio means fewer events for fused.",
        "",
        "## Cross-trial Cases",
        "",
        "| dtype | case | p50 median [min,max] | p90 | p99 [min,max] | TPS | append device | decode device | CUDA events | direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
            f"{metrics['profile_append_device']['median']:.4f}x | "
            f"{metrics['profile_decode_device']['median']:.4f}x | "
            f"{metrics['profile_cuda_events']['median']:.4f}x | "
            f"{row['direction']} |"
        )
    lines.extend(
        [
            "",
            "## Absolute Attribution Medians",
            "",
            "| dtype | case | backend | token p50 ms | append device ms/layer | decode device ms/layer | CUDA events | begin host ms | commit host ms | rollback p50 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        for backend in REQUIRED_BACKENDS:
            absolute = row["absolute"][backend]
            rollback = (
                "N/A"
                if absolute["rollback_p50_ms"] == 0.0
                else f"{absolute['rollback_p50_ms']:.6f}"
            )
            lines.append(
                f"| {row['dtype']} | {row['case']} | {backend} | "
                f"{absolute['p50_ms']:.6f} | "
                f"{absolute['profile_append_device_ms_per_layer']:.6f} | "
                f"{absolute['profile_decode_device_ms_per_layer']:.6f} | "
                f"{absolute['profile_cuda_event_count']:.0f} | "
                f"{absolute['begin_host_mean_ms']:.6f} | "
                f"{absolute['commit_host_mean_ms']:.6f} | {rollback} |"
            )
    lines.extend(
        [
            "",
            "## Overall Geometric Mean",
            "",
            "| metric | fused vs torch |",
            "| --- | ---: |",
            f"| complete-token p50 | {overall['p50']:.4f}x |",
            f"| complete-token p90 | {overall['p90']:.4f}x |",
            f"| complete-token p99 | {overall['p99']:.4f}x |",
            f"| complete-token mean | {overall['mean']:.4f}x |",
            f"| total CUDA p50 | {overall['device_p50']:.4f}x |",
            f"| per-layer CUDA p50 | {overall['layer_device_p50']:.4f}x |",
            f"| decode tokens/s | {overall['decode_tokens_per_second']:.4f}x |",
            f"| profiler append device/layer | {overall['profile_append_device']:.4f}x |",
            f"| profiler decode device/layer | {overall['profile_decode_device']:.4f}x |",
            f"| profiler CUDA events | {overall['profile_cuda_events']:.4f}x |",
            "",
            "## Interpretation",
            "",
            "- `unstable_crosses_1` means p50 crosses 1 across trials; do not claim a stable backend win.",
            "- A ratio below 1 means fused is worse for that metric; inspect the absolute attribution table before explaining why.",
            "- p99 must be reported with its range.",
            "- Profiler device totals and event counts explain launch/stage behavior but are not release latency.",
            "- Rollback latency remains an error-path metric and is not mixed into normal throughput.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmarks/results/r2_multi_layer_engine_trials3_summary.md",
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
    except (OSError, ValueError, MultiLayerValidationError) as exc:
        raise SystemExit(f"invalid multi-layer trial CSV: {exc}") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(pairs) * 2} rows and wrote {output}")


if __name__ == "__main__":
    main()
