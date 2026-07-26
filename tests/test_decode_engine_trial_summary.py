"""Validation and aggregation coverage for DecodeEngine multi-trial CSVs."""

import pytest

from benchmarks.summarize_decode_engine_trials import (
    DEFAULT_DTYPES,
    DEFAULT_WORKLOADS,
    REQUIRED_FIELDS,
    TrialValidationError,
    aggregate_trials,
    render_markdown,
    validate_trial_rows,
)


def _row(dtype, workload, trial, backend):
    row = {field: "1" for field in REQUIRED_FIELDS}
    order = "torch->fused_cuda" if trial % 2 else "fused_cuda->torch"
    row.update(
        {
            "name": "decode_engine_workload",
            "op": "decode_engine_workload",
            "workload": workload,
            "append_backend": backend,
            "decode_backend": "triton",
            "dtype": dtype,
            "device": "NVIDIA GeForce RTX 5070",
            "torch": "2.11.0+cu128",
            "cuda": "12.8",
            "git_commit": "abc1234",
            "num_q_heads": "32",
            "num_kv_heads": "8",
            "head_dim": "128",
            "block_size": "32",
            "num_warps": "2",
            "steps": "120",
            "warmup_steps": "5",
            "max_active": "8",
            "arrivals_per_step": "4",
            "decode_tokens_per_request": "4",
            "initial_context_tokens": "8",
            "context_stagger_tokens": "0",
            "cancel_interval": "5",
            "cancel_probability": "0.0",
            "max_blocks": "16",
            "trial": str(trial),
            "trial_count": "3",
            "backend_order": order,
            "repeats": "120",
            "successful_steps": "120",
            "completed_tokens": "960",
            "admitted_requests": "246",
            "finished_requests": "221",
            "cancelled_requests": "24",
            "prefilled_tokens": "1968",
            "backpressure_steps": "0",
            "final_active_requests": "4",
            "final_used_blocks": "4",
            "final_free_blocks": "12",
            "final_block_utilization": "0.25",
            "final_internal_fragmentation_tokens": "89",
            "allocations": "258",
            "frees": "254",
            "reuses": "250",
            "engine_backpressure_count": "0",
            "validated_invariants": "True",
            "timing_scope": "complete engine step",
            "seed": str(430 + trial),
        }
    )
    if backend == "torch":
        row.update(
            {
                "mean_ms": "2.000000",
                "p50_ms": "2.000000",
                "p90_ms": "2.200000",
                "p99_ms": "2.500000",
                "tokens_per_second": "1000.0",
                "speedup_vs_torch_p50": "1.0000",
            }
        )
    else:
        row.update(
            {
                "mean_ms": "1.600000",
                "p50_ms": "1.600000",
                "p90_ms": "2.000000",
                "p99_ms": "2.000000",
                "tokens_per_second": "1250.0",
                "speedup_vs_torch_p50": "1.2500",
            }
        )
    return row


def _valid_rows():
    return [
        _row(dtype, workload, trial, backend)
        for dtype in DEFAULT_DTYPES
        for workload in DEFAULT_WORKLOADS
        for trial in range(1, 4)
        for backend in ("torch", "fused_cuda")
    ]


def test_validate_and_aggregate_complete_trial_matrix():
    pairs = validate_trial_rows(_valid_rows())
    aggregates, overall = aggregate_trials(pairs)
    markdown = render_markdown("trials.csv", pairs, aggregates, overall)

    assert markdown.startswith("# DecodeEngine Workload Multi-trial Summary\n")
    assert len(pairs) == 18
    assert len(aggregates) == 6
    assert overall["p50"] == pytest.approx(1.25)
    assert overall["p90"] == pytest.approx(1.1)
    assert overall["p99"] == pytest.approx(1.25)
    assert overall["tokens_per_second"] == pytest.approx(1.25)
    assert all(row["direction"] == "fused_faster" for row in aggregates)
    assert "Rows: 36; paired trials: 18" in markdown
    assert "Git commit: `abc1234`" in markdown
    assert "fused_faster" in markdown


def test_validate_rejects_missing_backend_row():
    rows = _valid_rows()[:-1]
    with pytest.raises(TrialValidationError, match="incomplete"):
        validate_trial_rows(rows)


def test_validate_rejects_paired_state_trajectory_drift():
    rows = _valid_rows()
    target = next(
        row
        for row in rows
        if row["dtype"] == "float16"
        and row["workload"] == "short_churn"
        and row["trial"] == "1"
        and row["append_backend"] == "fused_cuda"
    )
    target["completed_tokens"] = "959"

    with pytest.raises(TrialValidationError, match="completed_tokens"):
        validate_trial_rows(rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validated_invariants", "False", "invariant validation failed"),
        ("final_free_blocks", "11", "block accounting failed"),
        ("p50_ms", "0", "positive and finite"),
        ("speedup_vs_torch_p50", "1.1000", "reported p50 speedup"),
    ],
)
def test_validate_rejects_invalid_per_row_evidence(field, value, message):
    rows = _valid_rows()
    target = next(row for row in rows if row["append_backend"] == "fused_cuda")
    target[field] = value

    with pytest.raises(TrialValidationError, match=message):
        validate_trial_rows(rows)


def test_validate_rejects_nonconsecutive_trial_seeds():
    rows = _valid_rows()
    for row in rows:
        if row["trial"] == "2":
            row["seed"] = "999"

    with pytest.raises(TrialValidationError, match="increase by one"):
        validate_trial_rows(rows)


def test_validate_rejects_fixed_backend_order():
    rows = _valid_rows()
    for row in rows:
        if row["trial"] == "2":
            row["backend_order"] = "torch->fused_cuda"

    with pytest.raises(TrialValidationError, match="reverse backend order"):
        validate_trial_rows(rows)


def test_aggregate_marks_p50_direction_crossing_one_as_unstable():
    rows = _valid_rows()
    for row in rows:
        if (
            row["dtype"] == "float16"
            and row["workload"] == "short_churn"
            and row["trial"] == "2"
            and row["append_backend"] == "fused_cuda"
        ):
            row["p50_ms"] = "2.500000"
            row["speedup_vs_torch_p50"] = "0.8000"

    pairs = validate_trial_rows(rows)
    aggregates, _ = aggregate_trials(pairs)
    target = next(
        row
        for row in aggregates
        if row["dtype"] == "float16" and row["workload"] == "short_churn"
    )
    assert target["direction"] == "unstable_crosses_1"
