"""Validate shared-prefix trials and write a cross-trial evidence summary."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


HIT_RATES = (0, 25, 50, 75)
DTYPES = ("float16", "bfloat16")

GLOBAL_IDENTITY_FIELDS = (
    "name",
    "op",
    "device",
    "torch",
    "cuda",
    "git_commit",
    "append_backend",
    "decode_backend",
    "num_layers",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_warps",
    "request_count",
    "context_tokens",
    "prefix_blocks",
    "decode_tokens",
    "tail_blocks",
    "warmup",
    "trial_count",
    "capacity_ratio",
    "capacity_probe_blocks",
    "latency_max_blocks",
    "bytes_per_block",
    "logical_context_blocks",
    "timing_scope",
)

REQUIRED_FIELDS = set(GLOBAL_IDENTITY_FIELDS) | {
    "dtype",
    "hit_rate_percent",
    "hit_count",
    "miss_count",
    "trial",
    "hit_rate_order",
    "seed",
    "repeats",
    "capacity_admitted_requests",
    "capacity_waiting_requests",
    "capacity_rejected_requests",
    "capacity_admission_rate",
    "capacity_committed_blocks",
    "capacity_physical_blocks",
    "physical_context_blocks",
    "physical_context_bytes",
    "context_memory_saving_ratio",
    "peak_used_blocks",
    "peak_allocated_kv_bytes",
    "resident_prefix_blocks",
    "active_prefix_references",
    "saved_prefix_blocks",
    "saved_prefix_bytes",
    "prefix_hit_count",
    "prefix_miss_count",
    "prefix_eviction_count",
    "registration_ms",
    "attach_mean_us",
    "attach_p50_us",
    "attach_p90_us",
    "eviction_us",
    "scheduler_p50_ms",
    "engine_step_p50_ms",
    "engine_step_p90_ms",
    "engine_step_p99_ms",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "complete_step_p99_ms",
    "decode_tokens_per_second",
    "final_free_blocks",
    "validated_invariants",
}


class SharedPrefixValidationError(ValueError):
    """Raised when shared-prefix evidence is incomplete or inconsistent."""


def _integer(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SharedPrefixValidationError(f"{field} must be an integer") from exc


def _number(row, field, *, positive=False):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SharedPrefixValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise SharedPrefixValidationError(f"{field} must be {qualifier} and finite")
    return value


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SharedPrefixValidationError("shared-prefix CSV must contain rows")
    return rows


def validate_rows(
    rows,
    *,
    expected_trials=3,
    expected_hit_rates=HIT_RATES,
    expected_dtypes=DTYPES,
):
    rows = list(rows)
    expected_hit_rates = tuple(int(value) for value in expected_hit_rates)
    expected_dtypes = tuple(expected_dtypes)
    if expected_trials <= 0:
        raise SharedPrefixValidationError("expected_trials must be positive")
    if not expected_hit_rates or len(set(expected_hit_rates)) != len(expected_hit_rates):
        raise SharedPrefixValidationError("expected_hit_rates must be non-empty and unique")
    if not expected_dtypes or len(set(expected_dtypes)) != len(expected_dtypes):
        raise SharedPrefixValidationError("expected_dtypes must be non-empty and unique")
    if not rows:
        raise SharedPrefixValidationError("rows must be non-empty")
    missing_fields = sorted(REQUIRED_FIELDS - set(rows[0]))
    if missing_fields:
        raise SharedPrefixValidationError(
            f"missing required columns: {', '.join(missing_fields)}"
        )
    for field in GLOBAL_IDENTITY_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise SharedPrefixValidationError(f"global field {field} is inconsistent")

    expected = {
        (dtype, rate, trial)
        for dtype in expected_dtypes
        for rate in expected_hit_rates
        for trial in range(1, expected_trials + 1)
    }
    indexed = {}
    seeds_by_trial = {}
    for row in rows:
        rate = _integer(row, "hit_rate_percent")
        trial = _integer(row, "trial")
        key = (row["dtype"], rate, trial)
        if key in indexed:
            raise SharedPrefixValidationError(f"duplicate row: {key}")
        indexed[key] = row
        seeds_by_trial.setdefault(trial, set()).add(_integer(row, "seed"))
        if row["validated_invariants"].lower() != "true":
            raise SharedPrefixValidationError(f"invariant failure: {key}")

        request_count = _integer(row, "request_count")
        prefix_blocks = _integer(row, "prefix_blocks")
        tail_blocks = _integer(row, "tail_blocks")
        hit_count = _integer(row, "hit_count")
        miss_count = _integer(row, "miss_count")
        if request_count * rate % 100 or hit_count != request_count * rate // 100:
            raise SharedPrefixValidationError(f"hit-count mismatch: {key}")
        if hit_count + miss_count != request_count:
            raise SharedPrefixValidationError(f"request-count mismatch: {key}")
        if _integer(row, "trial_count") != expected_trials:
            raise SharedPrefixValidationError(f"trial-count mismatch: {key}")
        if _integer(row, "decode_tokens") != _integer(row, "warmup") + _integer(row, "repeats"):
            raise SharedPrefixValidationError(f"decode-token mismatch: {key}")
        block_size = _integer(row, "block_size")
        if _integer(row, "context_tokens") != prefix_blocks * block_size:
            raise SharedPrefixValidationError(f"prefix/context mismatch: {key}")
        expected_tail_blocks = math.ceil(_integer(row, "decode_tokens") / block_size)
        if tail_blocks != expected_tail_blocks:
            raise SharedPrefixValidationError(f"tail-block mismatch: {key}")
        expected_latency_blocks = request_count * (prefix_blocks + tail_blocks)
        if _integer(row, "latency_max_blocks") != expected_latency_blocks:
            raise SharedPrefixValidationError(f"latency-capacity mismatch: {key}")
        expected_capacity_blocks = math.ceil(
            expected_latency_blocks * _number(row, "capacity_ratio", positive=True)
        )
        if _integer(row, "capacity_probe_blocks") != expected_capacity_blocks:
            raise SharedPrefixValidationError(f"capacity-probe mismatch: {key}")

        expected_resident = prefix_blocks if hit_count else 0
        expected_physical_context = (
            request_count * prefix_blocks
            if hit_count == 0
            else (miss_count + 1) * prefix_blocks
        )
        expected_saved = max(hit_count - 1, 0) * prefix_blocks
        expected_peak = expected_physical_context + request_count * tail_blocks
        bytes_per_block = _integer(row, "bytes_per_block")
        for field, value in (
            ("resident_prefix_blocks", expected_resident),
            ("active_prefix_references", hit_count),
            ("prefix_hit_count", hit_count),
            ("prefix_miss_count", 0),
            ("saved_prefix_blocks", expected_saved),
            ("saved_prefix_bytes", expected_saved * bytes_per_block),
            ("physical_context_blocks", expected_physical_context),
            ("physical_context_bytes", expected_physical_context * bytes_per_block),
            ("peak_used_blocks", expected_peak),
            ("peak_allocated_kv_bytes", expected_peak * bytes_per_block),
            ("prefix_eviction_count", int(hit_count > 0)),
            ("final_free_blocks", _integer(row, "latency_max_blocks")),
        ):
            if _integer(row, field) != value:
                raise SharedPrefixValidationError(f"{field} mismatch: {key}")

        logical_context = request_count * prefix_blocks
        expected_saving_ratio = 1.0 - expected_physical_context / logical_context
        if not math.isclose(
            _number(row, "context_memory_saving_ratio"),
            expected_saving_ratio,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise SharedPrefixValidationError(f"memory-saving mismatch: {key}")

        admitted = _integer(row, "capacity_admitted_requests")
        waiting = _integer(row, "capacity_waiting_requests")
        rejected = _integer(row, "capacity_rejected_requests")
        if admitted + waiting + rejected != request_count or rejected:
            raise SharedPrefixValidationError(f"capacity admission accounting failed: {key}")
        if not math.isclose(
            _number(row, "capacity_admission_rate"),
            admitted / request_count,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise SharedPrefixValidationError(f"capacity admission rate mismatch: {key}")
        if _integer(row, "capacity_committed_blocks") > _integer(row, "capacity_probe_blocks"):
            raise SharedPrefixValidationError(f"capacity overcommit: {key}")
        if _integer(row, "capacity_physical_blocks") != expected_resident:
            raise SharedPrefixValidationError(f"capacity physical accounting failed: {key}")
        remaining = expected_capacity_blocks - expected_resident
        expected_admitted = 0
        expected_private_commitment = 0
        commitments = [tail_blocks] * hit_count + [
            prefix_blocks + tail_blocks
        ] * miss_count
        for commitment in commitments:
            if commitment <= remaining:
                expected_admitted += 1
                expected_private_commitment += commitment
                remaining -= commitment
        if admitted != expected_admitted or waiting != request_count - expected_admitted:
            raise SharedPrefixValidationError(f"capacity FIFO admission mismatch: {key}")
        if _integer(row, "capacity_committed_blocks") != (
            expected_resident + expected_private_commitment
        ):
            raise SharedPrefixValidationError(f"capacity commitment mismatch: {key}")

        zero_only = (
            "registration_ms",
            "attach_mean_us",
            "attach_p50_us",
            "attach_p90_us",
            "eviction_us",
        )
        if hit_count == 0:
            if any(_number(row, field) != 0.0 for field in zero_only):
                raise SharedPrefixValidationError(f"zero-hit metadata is non-zero: {key}")
        else:
            if any(_number(row, field, positive=True) <= 0.0 for field in zero_only):
                raise SharedPrefixValidationError(f"prefix latency evidence missing: {key}")

        for field in (
            "scheduler_p50_ms",
            "engine_step_p50_ms",
            "engine_step_p90_ms",
            "engine_step_p99_ms",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "complete_step_p99_ms",
            "decode_tokens_per_second",
        ):
            _number(row, field, positive=True)

        offset = (trial - 1) % len(expected_hit_rates)
        expected_order = expected_hit_rates[offset:] + expected_hit_rates[:offset]
        if row["hit_rate_order"] != "->".join(str(value) for value in expected_order):
            raise SharedPrefixValidationError(f"hit-rate order mismatch: {key}")

    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise SharedPrefixValidationError(
            f"shared-prefix matrix mismatch; missing={missing}, extra={extra}"
        )
    if any(len(values) != 1 for values in seeds_by_trial.values()):
        raise SharedPrefixValidationError("each trial must use one seed")
    ordered_seeds = [next(iter(seeds_by_trial[trial])) for trial in range(1, expected_trials + 1)]
    if ordered_seeds != list(range(ordered_seeds[0], ordered_seeds[0] + expected_trials)):
        raise SharedPrefixValidationError("trial seeds must be consecutive")
    for dtype in expected_dtypes:
        for trial in range(1, expected_trials + 1):
            admissions = [
                _integer(indexed[(dtype, rate, trial)], "capacity_admitted_requests")
                for rate in expected_hit_rates
            ]
            if admissions != sorted(admissions):
                raise SharedPrefixValidationError(
                    f"capacity admission must be monotonic: {(dtype, trial)}"
                )
    return rows


def _median(rows, field):
    return statistics.median(_number(row, field) for row in rows)


def _paired_ratios(rows, dtype, rate, field, *, higher_is_better):
    indexed = {
        (_integer(row, "hit_rate_percent"), _integer(row, "trial")): row
        for row in rows
        if row["dtype"] == dtype
    }
    trials = sorted(
        trial
        for candidate_rate, trial in indexed
        if candidate_rate == 0
    )
    ratios = []
    for trial in trials:
        baseline = _number(indexed[(0, trial)], field, positive=True)
        candidate = _number(indexed[(int(rate), trial)], field, positive=True)
        ratios.append(
            candidate / baseline if higher_is_better else baseline / candidate
        )
    return ratios


def _ratio_cell(values):
    return (
        f"{statistics.median(values):.4f}x "
        f"[{min(values):.4f},{max(values):.4f}]"
    )


def _direction(values):
    if all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in values):
        return "baseline"
    if all(value > 1.0 for value in values):
        return "shared_faster"
    if all(value < 1.0 for value in values):
        return "shared_slower"
    return "crosses_1"


def write_summary(
    rows,
    output_path,
    input_path,
    *,
    expected_trials=3,
    expected_hit_rates=HIT_RATES,
    expected_dtypes=DTYPES,
):
    rows = validate_rows(
        rows,
        expected_trials=expected_trials,
        expected_hit_rates=expected_hit_rates,
        expected_dtypes=expected_dtypes,
    )
    first = rows[0]
    lines = [
        "# Shared Prefix Workload Summary",
        "",
        "## Validation",
        "",
        f"- Input: `{input_path}`.",
        f"- Rows: {len(rows)}; trials: {expected_trials}.",
        f"- Device: {first['device']}.",
        f"- PyTorch/CUDA: {first['torch']} / {first['cuda']}.",
        f"- Git commit: `{first['git_commit']}`.",
        "- Matrix, rotating hit-rate order, seed trajectory, capacity commitments, physical block/byte accounting, prefix lifecycle, materialized context, immutable prefix contents, and final cleanup were validated.",
        "- Capacity admission uses a fixed bounded pool; decode latency uses a separate fixed pool large enough to keep the request batch constant.",
        "",
        "## Cross-trial Medians",
        "",
        "| dtype | hit rate | admitted | context physical/logical blocks | context saved | peak blocks | saved KV-capacity MiB | attach p50 us | complete p50 ms | p90 ms | p99 ms | TPS | evictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dtype in expected_dtypes:
        for rate in expected_hit_rates:
            group = [
                row
                for row in rows
                if row["dtype"] == dtype
                and _integer(row, "hit_rate_percent") == int(rate)
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        dtype,
                        f"{int(rate)}%",
                        f"{_median(group, 'capacity_admitted_requests'):.0f}/{first['request_count']}",
                        (
                            f"{_median(group, 'physical_context_blocks'):.0f}/"
                            f"{_median(group, 'logical_context_blocks'):.0f}"
                        ),
                        f"{100.0 * _median(group, 'context_memory_saving_ratio'):.1f}%",
                        f"{_median(group, 'peak_used_blocks'):.0f}",
                        f"{_median(group, 'saved_prefix_bytes') / (1024 ** 2):.3f}",
                        f"{_median(group, 'attach_p50_us'):.3f}",
                        f"{_median(group, 'p50_ms'):.6f}",
                        f"{_median(group, 'p90_ms'):.6f}",
                        f"{_median(group, 'complete_step_p99_ms'):.6f}",
                        f"{_median(group, 'decode_tokens_per_second'):.3f}",
                        f"{_median(group, 'prefix_eviction_count'):.0f}",
                    ]
                )
                + " |"
            )
    if 0 in expected_hit_rates:
        lines.extend(
            [
                "",
                "## Paired vs 0% Hit Rate",
                "",
                "Ratios above 1 favor the shared-prefix case. Latency ratios are 0%/shared; TPS is shared/0%.",
                "",
                "| dtype | hit rate | p50 median [min,max] | p90 | p99 | TPS | p50 direction |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for dtype in expected_dtypes:
            for rate in expected_hit_rates:
                p50 = _paired_ratios(
                    rows, dtype, rate, "p50_ms", higher_is_better=False
                )
                p90 = _paired_ratios(
                    rows, dtype, rate, "p90_ms", higher_is_better=False
                )
                p99 = _paired_ratios(
                    rows,
                    dtype,
                    rate,
                    "complete_step_p99_ms",
                    higher_is_better=False,
                )
                tps = _paired_ratios(
                    rows,
                    dtype,
                    rate,
                    "decode_tokens_per_second",
                    higher_is_better=True,
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dtype,
                            f"{int(rate)}%",
                            _ratio_cell(p50),
                            _ratio_cell(p90),
                            _ratio_cell(p99),
                            _ratio_cell(tps),
                            _direction(p50),
                        ]
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "## Paired Latency Attribution vs 0%",
                "",
                "Ratios above 1 favor the shared-prefix case. Scheduler and Engine p50 values are measured separately and are not added together.",
                "",
                "| dtype | hit rate | scheduler p50 ratio | Engine p50 ratio | scheduler p50 ms | Engine p50 ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for dtype in expected_dtypes:
            for rate in expected_hit_rates:
                group = [
                    row
                    for row in rows
                    if row["dtype"] == dtype
                    and _integer(row, "hit_rate_percent") == int(rate)
                ]
                scheduler_ratio = _paired_ratios(
                    rows,
                    dtype,
                    rate,
                    "scheduler_p50_ms",
                    higher_is_better=False,
                )
                engine_ratio = _paired_ratios(
                    rows,
                    dtype,
                    rate,
                    "engine_step_p50_ms",
                    higher_is_better=False,
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dtype,
                            f"{int(rate)}%",
                            _ratio_cell(scheduler_ratio),
                            _ratio_cell(engine_ratio),
                            f"{_median(group, 'scheduler_p50_ms'):.6f}",
                            f"{_median(group, 'engine_step_p50_ms'):.6f}",
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The primary result is physical KV reduction and higher admission under the same bounded block pool.",
            "- Saved blocks/bytes are occupied KV-pool capacity avoided relative to private copies. The fixed-full-batch latency probe preallocates the same maximum tensor pool in every case, so this is not a direct process-VRAM measurement.",
            "- Prefix attach is a host metadata lookup; registration copy and final eviction are reported separately.",
            "- Decode latency keeps the same request count in every hit-rate case. Shared prefixes do not change the attention algorithm, so small latency differences should be treated as system noise unless repeated evidence is stable.",
            "- `crosses_1` means the paired p50 direction changes across trials; do not claim a stable latency win. p99 uses few samples per trial and must be read with its full range.",
            "- Non-instrumented synchronized wall time is the latency source; no profiler totals are mixed into release latency.",
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="benchmarks/results/r3_shared_prefix_workload_trials3.csv",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/r3_shared_prefix_workload_trials3_summary.md",
    )
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument(
        "--expected-hit-rates",
        nargs="+",
        type=int,
        default=list(HIT_RATES),
    )
    parser.add_argument(
        "--expected-dtypes",
        nargs="+",
        default=list(DTYPES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.input)
    write_summary(
        rows,
        args.output,
        args.input,
        expected_trials=args.expected_trials,
        expected_hit_rates=args.expected_hit_rates,
        expected_dtypes=args.expected_dtypes,
    )
    print(f"Validated {len(rows)} rows and wrote {args.output}")


if __name__ == "__main__":
    main()
