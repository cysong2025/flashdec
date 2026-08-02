"""CPU/reference and GPU integration coverage for scheduler policy workloads."""

import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.engine import DecodeEngine
from flashdec.scheduled_workload import (
    CANCEL_ON_BACKPRESSURE,
    GREEDY_STEP_ONLY,
    LIFETIME_FIFO_AGING,
    RequestArrival,
    SchedulerWorkloadConfig,
    SchedulerWorkloadResult,
    boundary_deadlock_arrivals,
    run_scheduler_workload,
)
from flashdec.scheduler import RequestSpec


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required"


def _engine(*, block_size=2, max_blocks=2, device="cpu", dtype=torch.float32):
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device=device,
    )
    return DecodeEngine(cache, append_backend="torch", decode_backend="reference")


def _boundary_config(policy):
    return SchedulerWorkloadConfig(
        name="boundary_deadlock",
        arrivals=boundary_deadlock_arrivals(num_requests=2, max_new_tokens=4),
        policy=policy,
        max_active_requests=2,
        max_batch_requests=2,
        max_steps=20,
        max_stalled_steps=2,
    )


def test_scheduled_workload_symbols_are_public_api():
    assert flashdec.CANCEL_ON_BACKPRESSURE == CANCEL_ON_BACKPRESSURE
    assert flashdec.GREEDY_STEP_ONLY == GREEDY_STEP_ONLY
    assert flashdec.LIFETIME_FIFO_AGING == LIFETIME_FIFO_AGING
    assert flashdec.RequestArrival is RequestArrival
    assert flashdec.SchedulerWorkloadConfig is SchedulerWorkloadConfig
    assert flashdec.SchedulerWorkloadResult is SchedulerWorkloadResult
    assert flashdec.boundary_deadlock_arrivals is boundary_deadlock_arrivals
    assert flashdec.run_scheduler_workload is run_scheduler_workload


def test_boundary_deadlock_distinguishes_three_scheduler_policies():
    lifetime = run_scheduler_workload(
        _engine(),
        _boundary_config(LIFETIME_FIFO_AGING),
        num_q_heads=2,
        seed=401,
    )
    greedy = run_scheduler_workload(
        _engine(),
        _boundary_config(GREEDY_STEP_ONLY),
        num_q_heads=2,
        seed=401,
    )
    cancel = run_scheduler_workload(
        _engine(),
        _boundary_config(CANCEL_ON_BACKPRESSURE),
        num_q_heads=2,
        seed=401,
    )

    assert lifetime.completed_request_ids == (0, 1)
    assert lifetime.completion_rate == 1.0
    assert lifetime.completed_tokens == 8
    assert lifetime.useful_tokens == 8
    assert lifetime.forced_cancellation_count == 0
    assert lifetime.resource_deadlock_count == 0
    assert all(
        physical <= committed
        for physical, committed in zip(
            lifetime.physical_block_samples,
            lifetime.committed_block_samples,
        )
    )

    assert greedy.completed_request_ids == ()
    assert greedy.cancelled_request_ids == ()
    assert greedy.resource_deadlock_count == 1
    assert greedy.backpressure_steps == 2
    assert greedy.completed_tokens == 4
    assert greedy.useful_tokens == 0

    assert len(cancel.completed_request_ids) == 1
    assert len(cancel.cancelled_request_ids) == 1
    assert cancel.completion_rate == 0.5
    assert cancel.forced_cancellation_count == 1
    assert cancel.resource_deadlock_count == 0
    assert cancel.backpressure_steps == 1
    assert cancel.useful_tokens == 4
    assert cancel.completed_tokens > cancel.useful_tokens


def test_lifetime_scheduler_completes_staggered_context_requests_without_starvation():
    arrivals = (
        RequestArrival(RequestSpec("large", 2, 4, 0), arrival_step=0),
        RequestArrival(RequestSpec("small-a", 0, 2, 1), arrival_step=0),
        RequestArrival(RequestSpec("small-b", 0, 2, 2), arrival_step=1),
    )
    config = SchedulerWorkloadConfig(
        name="finite_fairness",
        arrivals=arrivals,
        policy=LIFETIME_FIFO_AGING,
        max_active_requests=2,
        max_batch_requests=1,
        aging_threshold_steps=2,
        max_steps=30,
        max_stalled_steps=3,
    )

    result = run_scheduler_workload(
        _engine(block_size=2, max_blocks=4),
        config,
        num_q_heads=2,
        seed=409,
    )

    assert set(result.completed_request_ids) == {"large", "small-a", "small-b"}
    assert result.completion_rate == 1.0
    assert result.cancelled_request_ids == ()
    assert result.rejected_request_ids == ()
    assert result.resource_deadlock_count == 0
    assert result.max_service_wait_steps > 0
    assert result.max_waiting_depth > 0
    assert result.admission_wait_p90 >= result.admission_wait_p50
    assert result.engine_metrics["cache"]["used_blocks"] == 0


def test_lifetime_scheduler_rejects_request_that_can_never_fit():
    arrivals = (
        RequestArrival(RequestSpec("too-large", 0, 5, 0), arrival_step=0),
        RequestArrival(RequestSpec("small", 0, 2, 1), arrival_step=0),
    )
    config = SchedulerWorkloadConfig(
        name="rejection",
        arrivals=arrivals,
        policy=LIFETIME_FIFO_AGING,
        max_active_requests=1,
        max_batch_requests=1,
        max_steps=10,
    )

    result = run_scheduler_workload(
        _engine(block_size=2, max_blocks=2),
        config,
        num_q_heads=2,
        seed=419,
    )

    assert result.completed_request_ids == ("small",)
    assert result.rejected_request_ids == ("too-large",)
    assert result.engine_metrics["rejected_requests"] == 1
    assert result.resource_deadlock_count == 0


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_scheduler_managed_fused_triton_path_matches_reference():
    pytest.importorskip("triton")
    from flashdec.paged_reference import paged_decode_attention_ref
    from flashdec.rope import rope_paged_kv_append_ref
    from flashdec.scheduler import BlockAwareScheduler, SchedulerConfig

    torch.manual_seed(421)
    engine_cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=32,
        max_blocks=2,
        dtype=torch.float16,
        device="cuda",
    )
    reference_cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        block_size=32,
        max_blocks=2,
        dtype=torch.float16,
        device="cuda",
    )
    engine = DecodeEngine(
        engine_cache,
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=2,
    )
    engine.submit_request(RequestSpec("scheduled", 0, 1, 0))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    snapshot = engine.scheduling_snapshot(logical_step=0)
    decision = scheduler.plan(snapshot)
    engine.apply_scheduler_decision(
        decision,
        scheduler=scheduler,
        snapshot=snapshot,
    )

    q = torch.randn((1, 2, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    expected_append = rope_paged_kv_append_ref(
        reference_cache, 0, ["scheduled"], q, k, v
    )
    expected = paged_decode_attention_ref(
        expected_append.q,
        reference_cache.k_cache[0],
        reference_cache.v_cache[0],
        expected_append.block_tables,
        expected_append.seq_lens,
    )

    result = engine.step(q, k, v, request_ids=["scheduled"])
    torch.cuda.synchronize()

    torch.testing.assert_close(result.output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(engine_cache.k_cache, reference_cache.k_cache, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(engine_cache.v_cache, reference_cache.v_cache)
    assert engine.validate_invariants()
