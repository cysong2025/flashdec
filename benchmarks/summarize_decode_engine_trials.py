"""Validate paired DecodeEngine trials and write a stability summary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


DEFAULT_WORKLOADS = ("short_churn", "mixed_steady", "long_pressure")
DEFAULT_DTYPES = ("float16", "bfloat16")
REQUIRED_BACKENDS = ("torch", "fused_cuda")
RATIO_METRICS = ("p50", "p90", "p99", "mean", "tokens_per_second")

PAIR_IDENTITY_FIELDS = (
    "name",
    "op",
    "workload",
    "decode_backend",
    "dtype",
    "device",
    "torch",
    "cuda",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "steps",
    "warmup_steps",
    "max_active",
    "arrivals_per_step",
    "decode_tokens_per_request",
    "initial_context_tokens",
    "context_stagger_tokens",
    "cancel_interval",
    "cancel_probability",
    "max_blocks",
    "trial",
    "trial_count",
    "backend_order",
    "repeats",
    "successful_steps",
    "completed_tokens",
    "admitted_requests",
    "finished_requests",
    "cancelled_requests",
    "prefilled_tokens",
    "backpressure_steps",
    "final_active_requests",
    "final_used_blocks",
    "final_free_blocks",
    "final_block_utilization",
    "final_internal_fragmentation_tokens",
    "allocations",
    "frees",
    "reuses",
    "engine_backpressure_count",
    "timing_scope",
    "seed",
)

GLOBAL_IDENTITY_FIELDS = (
    "op",
    "decode_backend",
    "device",
    "torch",
    "cuda",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "warmup_steps",
    "trial_count",
    "timing_scope",
)

REQUIRED_FIELDS = set(PAIR_IDENTITY_FIELDS) | {
    "append_backend",
    "validated_invariants",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "tokens_per_second",
    "speedup_vs_torch_p50",
}


class TrialValidationError(ValueError):
    """Raised when trial rows cannot support a paired performance claim."""


@dataclass(frozen=True)
class PairedTrial:
    dtype: str
    workload: str
    trial: int
    torch_row: dict[str, str]
    fused_row: dict[str, str]

    @property
    def seed(self):
        return int(self.torch_row["seed"])

    @property
    def backend_order(self):
        return self.torch_row["backend_order"]

    def ratios(self):
        """Return ratios where values above one mean fused is better."""
        torch = self.torch_row
        fused = self.fused_row
        return {
            "p50": _positive_float(torch, "p50_ms") / _positive_float(fused, "p50_ms"),
            "p90": _positive_float(torch, "p90_ms") / _positive_float(fused, "p90_ms"),
            "p99": _positive_float(torch, "p99_ms") / _positive_float(fused, "p99_ms"),
            "mean": _positive_float(torch, "mean_ms") / _positive_float(fused, "mean_ms"),
            "tokens_per_second": (
                _positive_float(fused, "tokens_per_second")
                / _positive_float(torch, "tokens_per_second")
            ),
        }


def _positive_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrialValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise TrialValidationError(f"{field} must be positive and finite")
    return value


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrialValidationError(f"{field} must be an integer") from exc


def read_trial_csv(path):
    path = Path(path)
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise TrialValidationError("trial CSV must contain at least one row")
    return rows


def validate_trial_rows(
    rows,
    expected_trials=3,
    expected_workloads=DEFAULT_WORKLOADS,
    expected_dtypes=DEFAULT_DTYPES,
):
    """Validate completeness, pair identity, state trajectory, and trial metadata."""
    rows = list(rows)
    expected_trials = int(expected_trials)
    expected_workloads = tuple(expected_workloads)
    expected_dtypes = tuple(expected_dtypes)
    if expected_trials <= 0:
        raise TrialValidationError("expected_trials must be positive")
    if not expected_workloads or len(set(expected_workloads)) != len(expected_workloads):
        raise TrialValidationError("expected_workloads must be non-empty and unique")
    if not expected_dtypes or len(set(expected_dtypes)) != len(expected_dtypes):
        raise TrialValidationError("expected_dtypes must be non-empty and unique")
    if not rows:
        raise TrialValidationError("trial rows must be non-empty")

    missing_fields = sorted(REQUIRED_FIELDS - set(rows[0]))
    if missing_fields:
        raise TrialValidationError(f"missing required columns: {', '.join(missing_fields)}")

    for field in GLOBAL_IDENTITY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise TrialValidationError(f"global field {field} is inconsistent: {sorted(values)}")

    expected_keys = {
        (dtype, workload, trial, backend)
        for dtype in expected_dtypes
        for workload in expected_workloads
        for trial in range(1, expected_trials + 1)
        for backend in REQUIRED_BACKENDS
    }
    indexed = {}
    for row in rows:
        trial = _integer(row, "trial")
        key = (row["dtype"], row["workload"], trial, row["append_backend"])
        if key in indexed:
            raise TrialValidationError(f"duplicate trial row: {key}")
        indexed[key] = row

        if row["validated_invariants"] != "True":
            raise TrialValidationError(f"invariant validation failed: {key}")
        if _integer(row, "trial_count") != expected_trials:
            raise TrialValidationError(f"trial_count does not match expected_trials: {key}")
        if (
            _integer(row, "final_used_blocks") + _integer(row, "final_free_blocks")
            != _integer(row, "max_blocks")
        ):
            raise TrialValidationError(f"block accounting failed: {key}")
        for field in ("mean_ms", "p50_ms", "p90_ms", "p99_ms", "tokens_per_second"):
            _positive_float(row, field)

    actual_keys = set(indexed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise TrialValidationError(
            f"trial matrix is incomplete; missing={missing}, unexpected={unexpected}"
        )

    pairs = []
    for dtype in expected_dtypes:
        for workload in expected_workloads:
            for trial in range(1, expected_trials + 1):
                torch_row = indexed[(dtype, workload, trial, "torch")]
                fused_row = indexed[(dtype, workload, trial, "fused_cuda")]
                for field in PAIR_IDENTITY_FIELDS:
                    if torch_row[field] != fused_row[field]:
                        raise TrialValidationError(
                            f"paired trajectory differs for {(dtype, workload, trial)}: {field}"
                        )
                pair = PairedTrial(dtype, workload, trial, torch_row, fused_row)
                computed = pair.ratios()["p50"]
                reported = _positive_float(fused_row, "speedup_vs_torch_p50")
                if not math.isclose(computed, reported, rel_tol=0.0, abs_tol=5e-4):
                    raise TrialValidationError(
                        f"reported p50 speedup differs from paired rows: {(dtype, workload, trial)}"
                    )
                if not math.isclose(
                    _positive_float(torch_row, "speedup_vs_torch_p50"),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=5e-4,
                ):
                    raise TrialValidationError(
                        f"torch baseline speedup must be 1: {(dtype, workload, trial)}"
                    )
                pairs.append(pair)

    _validate_trial_sequence(rows, expected_trials)
    return pairs


def _validate_trial_sequence(rows, expected_trials):
    seeds = {}
    orders = {}
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        trial_seeds = {_integer(row, "seed") for row in trial_rows}
        trial_orders = {row["backend_order"] for row in trial_rows}
        if len(trial_seeds) != 1:
            raise TrialValidationError(f"trial {trial} uses inconsistent seeds")
        if len(trial_orders) != 1:
            raise TrialValidationError(f"trial {trial} uses inconsistent backend order")
        seed = next(iter(trial_seeds))
        order = next(iter(trial_orders))
        order_parts = tuple(order.split("->"))
        if len(order_parts) != 2 or set(order_parts) != set(REQUIRED_BACKENDS):
            raise TrialValidationError(f"trial {trial} has invalid backend order: {order}")
        seeds[trial] = seed
        orders[trial] = order_parts

    for trial in range(2, expected_trials + 1):
        if seeds[trial] != seeds[trial - 1] + 1:
            raise TrialValidationError("trial seeds must increase by one")
        if orders[trial] == orders[trial - 1]:
            raise TrialValidationError("adjacent trials must reverse backend order")


def geometric_mean(values):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean values must be positive and finite")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def aggregate_trials(pairs):
    """Aggregate trial ratios by dtype/workload and across the full matrix."""
    pairs = list(pairs)
    grouped = {}
    for pair in pairs:
        grouped.setdefault((pair.dtype, pair.workload), []).append(pair)

    aggregates = []
    for (dtype, workload), group in grouped.items():
        group = sorted(group, key=lambda item: item.trial)
        metric_values = {
            metric: [pair.ratios()[metric] for pair in group]
            for metric in RATIO_METRICS
        }
        p50_values = metric_values["p50"]
        if all(value > 1.0 for value in p50_values):
            direction = "fused_faster"
        elif all(value < 1.0 for value in p50_values):
            direction = "torch_faster"
        else:
            direction = "unstable_crosses_1"
        aggregates.append(
            {
                "dtype": dtype,
                "workload": workload,
                "trials": len(group),
                "direction": direction,
                "metrics": {
                    metric: {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "geometric_mean": geometric_mean(values),
                    }
                    for metric, values in metric_values.items()
                },
            }
        )

    overall = {
        metric: geometric_mean(pair.ratios()[metric] for pair in pairs)
        for metric in RATIO_METRICS
    }
    return sorted(aggregates, key=lambda row: (row["dtype"], row["workload"])), overall


def render_markdown(input_path, pairs, aggregates, overall):
    """Render an auditable Markdown summary from validated paired trials."""
    pairs = list(pairs)
    first = pairs[0].torch_row
    lines = [
        "# DecodeEngine Multi-trial Stability Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(pairs) * 2}; paired trials: {len(pairs)}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Decode backend: {first['decode_backend']}; block size: {first['block_size']}; num warps: {first['num_warps']}.",
        "- All rows passed engine/cache invariants, block accounting, pair trajectory, seed, and backend-order validation.",
        "",
        "Ratios above 1 mean `fused_cuda` is better. Latency ratios are `torch/fused`; TPS ratio is `fused/torch`.",
        "",
        "## Per-trial Ratios",
        "",
        "| dtype | workload | trial | seed | backend order | p50 | p90 | p99 | mean | TPS |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in pairs:
        ratios = pair.ratios()
        lines.append(
            f"| {pair.dtype} | {pair.workload} | {pair.trial} | {pair.seed} | "
            f"{pair.backend_order} | {ratios['p50']:.4f}x | {ratios['p90']:.4f}x | "
            f"{ratios['p99']:.4f}x | {ratios['mean']:.4f}x | "
            f"{ratios['tokens_per_second']:.4f}x |"
        )

    lines.extend(
        [
            "",
            "## Cross-trial Aggregates",
            "",
            "| dtype | workload | trials | p50 median [min, max] | p90 median | p99 median [min, max] | TPS median | p50 direction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregates:
        metrics = row["metrics"]
        lines.append(
            f"| {row['dtype']} | {row['workload']} | {row['trials']} | "
            f"{metrics['p50']['median']:.4f}x [{metrics['p50']['min']:.4f}, {metrics['p50']['max']:.4f}] | "
            f"{metrics['p90']['median']:.4f}x | "
            f"{metrics['p99']['median']:.4f}x [{metrics['p99']['min']:.4f}, {metrics['p99']['max']:.4f}] | "
            f"{metrics['tokens_per_second']['median']:.4f}x | {row['direction']} |"
        )

    lines.extend(
        [
            "",
            "## Overall Geometric Mean",
            "",
            "| metric | fused vs torch |",
            "| --- | ---: |",
            f"| p50 | {overall['p50']:.4f}x |",
            f"| p90 | {overall['p90']:.4f}x |",
            f"| p99 | {overall['p99']:.4f}x |",
            f"| mean latency | {overall['mean']:.4f}x |",
            f"| tokens/s | {overall['tokens_per_second']:.4f}x |",
            "",
            "## Interpretation Rule",
            "",
            "- `fused_faster`: all p50 trials are above 1.",
            "- `torch_faster`: all p50 trials are below 1.",
            "- `unstable_crosses_1`: trials cross 1; do not claim a stable backend win.",
            "- p99 must be reported with its trial range; a single outlier is not a release-level conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmarks/results/week12_decode_engine_workload_trials3_summary.md",
    )
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--expected-workloads", nargs="+", default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--expected-dtypes", nargs="+", default=list(DEFAULT_DTYPES))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        rows = read_trial_csv(args.input)
        pairs = validate_trial_rows(
            rows,
            expected_trials=args.expected_trials,
            expected_workloads=args.expected_workloads,
            expected_dtypes=args.expected_dtypes,
        )
        aggregates, overall = aggregate_trials(pairs)
        markdown = render_markdown(args.input, pairs, aggregates, overall)
    except (OSError, TrialValidationError, ValueError) as exc:
        raise SystemExit(f"invalid DecodeEngine trial CSV: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(pairs) * 2} rows and wrote {output}")


if __name__ == "__main__":
    main()
