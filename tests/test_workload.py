"""CPU/reference coverage for synthetic dynamic DecodeEngine workloads."""

import pytest

torch = pytest.importorskip("torch")

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.engine import DecodeEngine
from flashdec.workload import WorkloadConfig, WorkloadResult, run_synthetic_workload


def _engine(block_size=2, max_blocks=8):
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=torch.float32,
        device="cpu",
    )
    return DecodeEngine(cache, append_backend="torch", decode_backend="reference")


def test_workload_symbols_are_public_api():
    assert flashdec.WorkloadConfig is WorkloadConfig
    assert flashdec.WorkloadResult is WorkloadResult
    assert flashdec.run_synthetic_workload is run_synthetic_workload


def test_synthetic_workload_runs_dynamic_churn_and_reports_measured_metrics():
    result = run_synthetic_workload(
        _engine(),
        WorkloadConfig(
            name="cpu_churn",
            steps=8,
            max_active=2,
            arrivals_per_step=1,
            decode_tokens_per_request=2,
            initial_context_tokens=1,
            context_stagger_tokens=1,
        ),
        num_q_heads=2,
        warmup_steps=1,
        seed=313,
    )

    assert len(result.latencies_ms) == 8
    assert result.successful_steps == 8
    assert result.completed_tokens > 0
    assert result.admitted_requests == 8
    assert result.prefilled_tokens > 0
    assert result.backpressure_steps == 0
    assert result.mean_ms > 0
    assert result.p50_ms > 0
    assert result.p90_ms > 0
    assert result.p99_ms > 0
    assert result.tokens_per_second > 0
    assert 1 <= result.mean_active_batch <= 2
    assert result.max_active_batch == 2
    assert result.engine_metrics["completed_step_count"] == 9
    assert result.engine_metrics["cache"]["used_blocks"] <= 8


def test_synthetic_workload_records_backpressure_and_recovers_with_cancellation():
    engine = _engine(block_size=1, max_blocks=1)
    result = run_synthetic_workload(
        engine,
        WorkloadConfig(
            name="cpu_pressure",
            steps=4,
            max_active=2,
            arrivals_per_step=2,
            decode_tokens_per_request=4,
            cancel_on_backpressure=True,
        ),
        num_q_heads=2,
        warmup_steps=0,
        seed=317,
    )

    assert len(result.latencies_ms) == 4
    assert result.successful_steps == 0
    assert result.completed_tokens == 0
    assert result.backpressure_steps == 4
    assert result.cancelled_requests == 4
    assert result.engine_metrics["backpressure_count"] == 4
    assert engine.metrics()["active_requests"] == 1
    assert engine.validate_invariants()


def test_synthetic_workload_supports_seeded_probabilistic_cancellation():
    result = run_synthetic_workload(
        _engine(),
        WorkloadConfig(
            name="cpu_probabilistic_cancel",
            steps=3,
            max_active=2,
            arrivals_per_step=2,
            decode_tokens_per_request=8,
            cancel_probability=1.0,
        ),
        num_q_heads=2,
        warmup_steps=0,
        seed=331,
    )

    assert result.successful_steps == 3
    assert result.completed_tokens == 6
    assert result.cancelled_requests == 6
    assert result.engine_metrics["active_requests"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"initial_context_tokens": -1}, "initial_context_tokens"),
        ({"context_stagger_tokens": -1}, "context_stagger_tokens"),
        ({"cancel_interval": -1}, "cancel_interval"),
        ({"cancel_probability": 1.1}, "cancel_probability"),
    ],
)
def test_workload_config_rejects_invalid_values(kwargs, message):
    defaults = {
        "name": "invalid",
        "steps": 1,
        "max_active": 1,
        "arrivals_per_step": 1,
        "decode_tokens_per_request": 1,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        WorkloadConfig(**defaults)
