"""Validate and summarize the formal scheduler policy workload matrix."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import statistics


CASES = ("boundary_deadlock", "finite_queue")
DTYPES = ("float16", "bfloat16")
POLICIES = (
    "cancel_on_backpressure",
    "greedy_step_only",
    "lifetime_fifo_aging",
)


def _as_int(row, name):
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {name!r}") from exc


def _as_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {name!r}") from exc


def validate_scheduler_rows(rows, expected_trials=3):
    """Validate the exact matrix, provenance, and policy-specific invariants."""
    if isinstance(expected_trials, bool) or not isinstance(expected_trials, int) or expected_trials <= 0:
        raise ValueError("expected_trials must be positive")
    rows = list(rows)
    expected = set(product(CASES, DTYPES, POLICIES, range(1, expected_trials + 1)))
    actual = set()
    commits = set()
    devices = set()
    seeds_by_trial = {}
    for row in rows:
        key = (
            row.get("case"),
            row.get("dtype"),
            row.get("policy"),
            _as_int(row, "trial"),
        )
        if key in actual:
            raise ValueError(f"duplicate scheduler row: {key}")
        actual.add(key)
        commits.add(row.get("git_commit", ""))
        devices.add(row.get("device", ""))
        seeds_by_trial.setdefault(key[3], set()).add(_as_int(row, "seed"))
        if row.get("validated_invariants", "").lower() != "true":
            raise ValueError(f"scheduler invariant failure: {key}")
        if _as_int(row, "resource_deadlocks") < 0:
            raise ValueError("resource_deadlocks must be non-negative")
        completion = _as_float(row, "completion_rate")
        if not 0.0 <= completion <= 1.0:
            raise ValueError("completion_rate must be in [0, 1]")
        if _as_int(row, "useful_tokens") > _as_int(row, "completed_tokens"):
            raise ValueError("useful_tokens must not exceed completed_tokens")

        case, _, policy, _ = key
        trial = key[3]
        offset = (trial - 1) % len(POLICIES)
        expected_order = POLICIES[offset:] + POLICIES[:offset]
        if row.get("policy_order") != "->".join(expected_order):
            raise ValueError("policy_order does not match the expected trial rotation")
        deadlocks = _as_int(row, "resource_deadlocks")
        forced = _as_int(row, "forced_cancellations")
        cancelled = _as_int(row, "cancelled_requests")
        if case == "boundary_deadlock":
            if policy == "lifetime_fifo_aging":
                if completion != 1.0 or deadlocks or forced or cancelled:
                    raise ValueError("lifetime boundary case must complete without cancellation")
            elif policy == "greedy_step_only":
                if deadlocks != 1 or forced or cancelled:
                    raise ValueError("greedy boundary case must expose one deadlock")
            elif policy == "cancel_on_backpressure":
                if deadlocks or forced <= 0 or cancelled <= 0:
                    raise ValueError("cancel boundary case must recover through forced cancellation")
        elif case == "finite_queue":
            if completion != 1.0 or deadlocks:
                raise ValueError("finite_queue must complete without deadlock")

    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"scheduler matrix mismatch; missing={missing}, extra={extra}")
    if len(commits) != 1 or not next(iter(commits)):
        raise ValueError("scheduler rows must use one non-empty Git commit")
    if len(devices) != 1 or not next(iter(devices)):
        raise ValueError("scheduler rows must use one non-empty device")
    if any(len(values) != 1 for values in seeds_by_trial.values()):
        raise ValueError("each trial must use one seed across the matrix")
    ordered_seeds = [next(iter(seeds_by_trial[trial])) for trial in range(1, expected_trials + 1)]
    if ordered_seeds != list(range(ordered_seeds[0], ordered_seeds[0] + expected_trials)):
        raise ValueError("trial seeds must be consecutive")
    return rows


def _median(rows, field):
    return statistics.median(_as_float(row, field) for row in rows)


def write_summary(rows, output_path, input_path, expected_trials):
    rows = validate_scheduler_rows(rows, expected_trials=expected_trials)
    commit = rows[0]["git_commit"]
    device = rows[0]["device"]
    lines = [
        "# Scheduler Policy Workload Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; expected trials: {expected_trials}.",
        f"- Device: {device}.",
        f"- Git commit: `{commit}`.",
        "- Exact case/dtype/policy/trial matrix and policy-specific invariants passed.",
        "",
        "## Cross-trial Medians",
        "",
        "| case | dtype | policy | completion | cancellations | deadlocks | p50 ms | p99 ms | useful TPS | wait p90 | scheduler p50 ms | max committed/physical |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in CASES:
        for dtype in DTYPES:
            for policy in POLICIES:
                group = [
                    row
                    for row in rows
                    if row["case"] == case
                    and row["dtype"] == dtype
                    and row["policy"] == policy
                ]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            case,
                            dtype,
                            policy,
                            f"{_median(group, 'completion_rate'):.3f}",
                            f"{_median(group, 'forced_cancellations'):.0f}",
                            f"{_median(group, 'resource_deadlocks'):.0f}",
                            f"{_median(group, 'p50_ms'):.6f}",
                            f"{_median(group, 'p99_ms'):.6f}",
                            f"{_median(group, 'useful_tokens_per_second'):.3f}",
                            f"{_median(group, 'admission_wait_p90'):.1f}",
                            f"{_median(group, 'scheduler_p50_ms'):.6f}",
                            (
                                f"{_median(group, 'max_committed_blocks'):.0f}/"
                                f"{_median(group, 'max_physical_blocks'):.0f}"
                            ),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `useful TPS` only counts tokens belonging to requests that eventually completed.",
            "- Boundary deadlock is a correctness/progress result; latency is secondary when completion differs.",
            "- Scheduler and complete-step timings share one row but should still be interpreted with completion, wait, and memory metrics.",
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="benchmarks/results/r1_scheduler_workload_trials3.csv")
    parser.add_argument("--output", default="benchmarks/results/r1_scheduler_workload_trials3_summary.md")
    parser.add_argument("--expected-trials", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.input).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    write_summary(rows, args.output, args.input, args.expected_trials)
    print(f"Validated {len(rows)} scheduler rows and wrote {args.output}")


if __name__ == "__main__":
    main()
