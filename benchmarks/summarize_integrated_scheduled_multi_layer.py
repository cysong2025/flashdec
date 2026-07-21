"""Strictly validate and summarize R4-C integrated workload evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.integrated_workload import (
    build_integrated_reference,
    standard_integrated_config,
)


DEFAULT_CASES = tuple(
    f"l{layers}_c{context}"
    for layers in (2, 4)
    for context in (64, 128)
)
DEFAULT_DTYPES = ("float16", "bfloat16")
GLOBAL_FIELDS = (
    "device",
    "torch",
    "cuda",
    "git_commit",
    "append_backend",
    "decode_backend",
    "metadata_policy",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "timing_scope",
)
REQUIRED_FIELDS = set(GLOBAL_FIELDS) | {
    "name",
    "op",
    "case",
    "dtype",
    "num_layers",
    "context_tokens",
    "prefix_blocks",
    "max_blocks",
    "trial",
    "trial_count",
    "case_order",
    "seed",
    "repeats",
    "reference_steps",
    "trajectory_digest",
    "reference_trajectory_digest",
    "trajectory_validated",
    "completed_request_ids",
    "cancelled_request_ids",
    "rejected_request_ids",
    "successful_steps",
    "aborted_steps",
    "completed_tokens",
    "block_reuse_count",
    "peak_used_blocks",
    "terminal_resident_prefix_blocks",
    "final_free_blocks",
    "bytes_per_block",
    "peak_allocated_kv_bytes",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "complete_step_p99_ms",
    "scheduler_p50_ms",
    "context_seed_p50_ms",
    "engine_p50_ms",
    "decode_tokens_per_second",
    "transaction_begin_count",
    "transaction_commit_count",
    "transaction_abort_count",
    "transaction_layer_write_count",
    "transaction_rollback_block_count",
    "engine_transaction_layer_step_count",
    "engine_transaction_abort_count",
    "prefix_registration_count",
    "prefix_hit_count",
    "prefix_eviction_count",
    "final_open_transaction_count",
    "final_used_blocks",
    "validated_invariants",
}


class IntegratedValidationError(ValueError):
    """Raised when R4-C rows cannot support the recorded conclusions."""


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegratedValidationError(f"{field} must be an integer") from exc


def _positive_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegratedValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise IntegratedValidationError(f"{field} must be positive and finite")
    return value


def read_csv(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise IntegratedValidationError("R4-C CSV must contain rows")
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
        raise IntegratedValidationError("expected_trials must be positive")
    if not rows:
        raise IntegratedValidationError("rows must be non-empty")
    missing_fields = sorted(REQUIRED_FIELDS - set(rows[0]))
    if missing_fields:
        raise IntegratedValidationError(
            f"missing required columns: {', '.join(missing_fields)}"
        )
    for field in GLOBAL_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise IntegratedValidationError(f"global field {field} is inconsistent")
    if rows[0]["append_backend"] != "fused_cuda":
        raise IntegratedValidationError("R4-C must use the frozen fused CUDA path")
    if rows[0]["decode_backend"] != "triton":
        raise IntegratedValidationError("R4-C must use the frozen Triton decode path")
    if rows[0]["metadata_policy"] != "materialized":
        raise IntegratedValidationError("R4-C must use the R4-A materialized baseline")

    expected_keys = {
        (dtype, case, trial)
        for dtype in expected_dtypes
        for case in expected_cases
        for trial in range(1, expected_trials + 1)
    }
    indexed = {}
    for row in rows:
        key = (row["dtype"], row["case"], _integer(row, "trial"))
        if key in indexed:
            raise IntegratedValidationError(f"duplicate row: {key}")
        indexed[key] = row
        if row["name"] != "integrated_scheduled_multi_layer" or row["op"] != row["name"]:
            raise IntegratedValidationError(f"operation identity mismatch: {key}")
        if row["trajectory_validated"] != "True":
            raise IntegratedValidationError(f"trajectory validation failed: {key}")
        if row["validated_invariants"] != "True":
            raise IntegratedValidationError(f"invariant validation failed: {key}")
        if _integer(row, "trial_count") != expected_trials:
            raise IntegratedValidationError(f"trial_count mismatch: {key}")

        layers = _integer(row, "num_layers")
        context = _integer(row, "context_tokens")
        block_size = _integer(row, "block_size")
        prefix_blocks = _integer(row, "prefix_blocks")
        max_blocks = _integer(row, "max_blocks")
        if row["case"] != f"l{layers}_c{context}":
            raise IntegratedValidationError(f"case geometry mismatch: {key}")
        if context % block_size or prefix_blocks != context // block_size:
            raise IntegratedValidationError(f"prefix geometry mismatch: {key}")
        if max_blocks != 2 * prefix_blocks + 4:
            raise IntegratedValidationError(f"max_blocks mismatch: {key}")

        reference = build_integrated_reference(
            standard_integrated_config(
                num_layers=layers,
                context_tokens=context,
            ),
            block_size=block_size,
            max_blocks=max_blocks,
            resident_prefix_blocks=prefix_blocks,
        )
        if row["trajectory_digest"] != reference.digest:
            raise IntegratedValidationError(f"trajectory digest mismatch: {key}")
        if row["reference_trajectory_digest"] != reference.digest:
            raise IntegratedValidationError(f"reference digest mismatch: {key}")
        expected_values = {
            "repeats": len(reference.steps),
            "reference_steps": len(reference.steps),
            "successful_steps": reference.successful_steps,
            "aborted_steps": reference.aborted_steps,
            "completed_tokens": reference.completed_tokens,
            "peak_used_blocks": max(step.used_blocks for step in reference.steps),
            "terminal_resident_prefix_blocks": prefix_blocks,
            "final_free_blocks": max_blocks,
            "transaction_begin_count": 2 * context + reference.successful_steps + reference.aborted_steps,
            "transaction_commit_count": 2 * context + reference.successful_steps,
            "transaction_abort_count": reference.aborted_steps,
            "transaction_layer_write_count": (
                (2 * context + reference.successful_steps) * layers + 1
            ),
            "transaction_rollback_block_count": 1,
            "engine_transaction_layer_step_count": reference.successful_steps * layers + 1,
            "engine_transaction_abort_count": reference.aborted_steps,
            "prefix_registration_count": 1,
            "prefix_hit_count": 2,
            "prefix_eviction_count": 1,
            "final_open_transaction_count": 0,
            "final_used_blocks": 0,
        }
        for field, expected in expected_values.items():
            if _integer(row, field) != expected:
                raise IntegratedValidationError(
                    f"{field} mismatch for {key}: expected {expected}"
                )
        if row["completed_request_ids"] != "|".join(reference.completed_request_ids):
            raise IntegratedValidationError(f"completion trajectory mismatch: {key}")
        if row["cancelled_request_ids"] != "|".join(reference.cancelled_request_ids):
            raise IntegratedValidationError(f"cancellation trajectory mismatch: {key}")
        if row["rejected_request_ids"] != "|".join(reference.rejected_request_ids):
            raise IntegratedValidationError(f"rejection trajectory mismatch: {key}")
        if _integer(row, "block_reuse_count") <= 0:
            raise IntegratedValidationError(f"released-block reuse missing: {key}")
        if _integer(row, "peak_allocated_kv_bytes") != (
            _integer(row, "peak_used_blocks") * _integer(row, "bytes_per_block")
        ):
            raise IntegratedValidationError(f"peak KV bytes mismatch: {key}")
        for field in (
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "complete_step_p99_ms",
            "scheduler_p50_ms",
            "context_seed_p50_ms",
            "engine_p50_ms",
            "decode_tokens_per_second",
        ):
            _positive_float(row, field)

    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        unexpected = sorted(set(indexed) - expected_keys)
        raise IntegratedValidationError(
            f"matrix mismatch; missing={missing}, unexpected={unexpected}"
        )
    _validate_trial_sequence(rows, expected_trials, expected_cases)
    return rows


def _validate_trial_sequence(rows, expected_trials, expected_cases):
    expected_cases = tuple(expected_cases)
    previous_seed = None
    for trial in range(1, expected_trials + 1):
        trial_rows = [row for row in rows if _integer(row, "trial") == trial]
        seeds = {_integer(row, "seed") for row in trial_rows}
        orders = {row["case_order"] for row in trial_rows}
        if len(seeds) != 1 or len(orders) != 1:
            raise IntegratedValidationError(f"trial {trial} has inconsistent seed/order")
        seed = next(iter(seeds))
        offset = (trial - 1) % len(expected_cases)
        expected_order = expected_cases[offset:] + expected_cases[:offset]
        if tuple(next(iter(orders)).split("->")) != expected_order:
            raise IntegratedValidationError(f"trial {trial} case order is invalid")
        if previous_seed is not None and seed != previous_seed + 1:
            raise IntegratedValidationError("trial seeds must increase by one")
        previous_seed = seed


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dtype"], row["case"]), []).append(row)
    result = []
    for (dtype, case), group in sorted(grouped.items()):
        metrics = {}
        for field in (
            "p50_ms",
            "p90_ms",
            "complete_step_p99_ms",
            "scheduler_p50_ms",
            "context_seed_p50_ms",
            "engine_p50_ms",
            "decode_tokens_per_second",
        ):
            values = [_positive_float(row, field) for row in group]
            metrics[field] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        result.append({"dtype": dtype, "case": case, "metrics": metrics})
    return result


def render_markdown(input_path, rows, aggregates):
    first = rows[0]
    lines = [
        "# R4-C Integrated Scheduled Multi-layer Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; trials: {first['trial_count']}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Frozen path: fused CUDA append + Triton decode + R4-A materialized transaction metadata.",
        "- Reference digest, dynamic admission/defer/completion/cancellation trajectory, rollback, transaction counts, prefix lifetime, released-block reuse, and final zero-used cleanup were validated.",
        "- This matrix reports one integrated workload; it is not a shared-prefix speedup A/B and does not reopen frozen kernel tuning.",
        "",
        "## Cross-trial Absolute Results",
        "",
        "| dtype | case | complete p50 ms [min,max] | p90 ms | p99 ms [min,max] | scheduler p50 ms | context seed p50 ms | Engine p50 ms | tokens/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregates:
        metrics = item["metrics"]
        lines.append(
            f"| {item['dtype']} | {item['case']} | "
            f"{metrics['p50_ms']['median']:.6f} "
            f"[{metrics['p50_ms']['min']:.6f},{metrics['p50_ms']['max']:.6f}] | "
            f"{metrics['p90_ms']['median']:.6f} | "
            f"{metrics['complete_step_p99_ms']['median']:.6f} "
            f"[{metrics['complete_step_p99_ms']['min']:.6f},{metrics['complete_step_p99_ms']['max']:.6f}] | "
            f"{metrics['scheduler_p50_ms']['median']:.6f} | "
            f"{metrics['context_seed_p50_ms']['median']:.6f} | "
            f"{metrics['engine_p50_ms']['median']:.6f} | "
            f"{metrics['decode_tokens_per_second']['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The primary R4-C gate is correctness and lifecycle closure, not a latency ratio.",
            "- p99 remains a small finite-trace tail statistic and is reported with the cross-trial range.",
            "- Context seeding is caller-supplied multi-layer prompt state for private misses; shared hits attach the fixed resident prefix. Random tensor construction and prefix registration are outside timing.",
            "- Terminal cleanup occurs only after every request is finished, cancelled, or rejected; no online prefix registration/eviction is part of this first slice.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmarks/results/r4_integrated_scheduled_multi_layer_summary.md",
    )
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--expected-cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--expected-dtypes", nargs="+", default=list(DEFAULT_DTYPES))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        rows = read_csv(args.input)
        validate_rows(
            rows,
            expected_trials=args.expected_trials,
            expected_cases=args.expected_cases,
            expected_dtypes=args.expected_dtypes,
        )
        markdown = render_markdown(args.input, rows, aggregate(rows))
    except (OSError, ValueError, IntegratedValidationError) as exc:
        raise SystemExit(f"invalid R4-C workload CSV: {exc}") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown)
    print(f"Validated {len(rows)} rows and wrote {output}")


if __name__ == "__main__":
    main()
