"""Dynamic request lifecycle and scheduler integration coverage for DecodeEngine."""

from dataclasses import replace
import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.engine import (
    AdmissionResult,
    DecodeEngine,
    DecodeStepResult,
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
    PROFILE_RANGE_PREFLIGHT,
)
from flashdec.paged_reference import paged_decode_attention_ref
from flashdec.rope import rope_paged_kv_append_ref
from flashdec.scheduler import BlockAwareScheduler, RequestSpec, SchedulerConfig


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required for fused engine tests"


def _make_cache(
    device="cpu",
    dtype=torch.float32,
    num_kv_heads=1,
    head_dim=4,
    block_size=2,
    max_blocks=8,
    prefix_cache_capacity_blocks=0,
):
    return PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device=device,
        prefix_cache_capacity_blocks=prefix_cache_capacity_blocks,
    )


def _step_inputs(batch_size, device="cpu", dtype=torch.float32, num_q_heads=2, num_kv_heads=1, head_dim=4):
    q = torch.randn((batch_size, num_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    return q, k, torch.randn_like(k)


def _reference_step(cache, request_ids, q, k, v):
    append = rope_paged_kv_append_ref(cache, 0, request_ids, q, k, v)
    output = paged_decode_attention_ref(
        append.q,
        cache.k_cache[0],
        cache.v_cache[0],
        append.block_tables,
        append.seq_lens,
    )
    return append, output


def _assert_step_matches(result, expected_append, expected_output):
    assert result.status == DecodeEngine.STEP_OK
    torch.testing.assert_close(result.output, expected_output)
    torch.testing.assert_close(result.positions, expected_append.positions)
    torch.testing.assert_close(result.block_tables, expected_append.block_tables)
    torch.testing.assert_close(result.seq_lens, expected_append.seq_lens)


def test_decode_engine_is_public_api():
    assert flashdec.DecodeEngine is DecodeEngine
    assert flashdec.AdmissionResult is AdmissionResult
    assert flashdec.DecodeStepResult is DecodeStepResult


def test_decode_engine_dynamic_lifecycle_matches_reference_path():
    torch.manual_seed(271)
    engine_cache = _make_cache()
    reference_cache = _make_cache()
    engine = DecodeEngine(engine_cache)

    assert engine.add_request(10) == AdmissionResult(10, DecodeEngine.WAITING)
    assert engine.add_request(20) == AdmissionResult(20, DecodeEngine.WAITING)
    assert engine.add_request(30) == AdmissionResult(30, DecodeEngine.WAITING)
    assert engine.admit([10, 20]) == (
        AdmissionResult(10, DecodeEngine.ACTIVE),
        AdmissionResult(20, DecodeEngine.ACTIVE),
    )
    assert engine.active_request_ids() == (10, 20)

    q, k, v = _step_inputs(2)
    expected_append, expected_output = _reference_step(reference_cache, [10, 20], q, k, v)
    result = engine.step(q, k, v, request_ids=[10, 20])
    _assert_step_matches(result, expected_append, expected_output)

    assert engine.finish_request(20) == reference_cache.finish_request(20)
    assert engine.admit([30]) == (AdmissionResult(30, DecodeEngine.ACTIVE),)
    assert engine.active_request_ids() == (10, 30)

    q, k, v = _step_inputs(2)
    expected_append, expected_output = _reference_step(reference_cache, [10, 30], q, k, v)
    result = engine.step(q, k, v)
    _assert_step_matches(result, expected_append, expected_output)

    assert engine.cancel_request(10) == reference_cache.cancel_request(10)
    q, k, v = _step_inputs(1)
    expected_append, expected_output = _reference_step(reference_cache, [30], q, k, v)
    result = engine.step(q, k, v)
    _assert_step_matches(result, expected_append, expected_output)

    assert engine.request_state(10)["status"] == DecodeEngine.CANCELLED
    assert engine.request_state(20)["status"] == DecodeEngine.FINISHED
    assert engine.request_state(30)["cache"]["seq_len"] == 2
    assert engine.metrics()["completed_step_count"] == 3
    assert engine.metrics()["appended_token_count"] == 5
    assert engine.validate_invariants()


def test_decode_engine_returns_backpressure_without_mutating_active_rows():
    engine = DecodeEngine(_make_cache(block_size=1, max_blocks=1))
    engine.add_request(1)
    engine.add_request(2)
    engine.admit()

    q, k, v = _step_inputs(1)
    first = engine.step(q, k, v, request_ids=[1])
    assert first.status == DecodeEngine.STEP_OK
    before = engine.request_state(2)

    backpressured = engine.step(q, k, v, request_ids=[2])
    assert backpressured.status == DecodeEngine.STEP_BACKPRESSURE
    assert backpressured.reason == "insufficient_physical_blocks"
    assert backpressured.needed_new_blocks == 1
    assert backpressured.free_blocks == 0
    assert backpressured.output is None
    assert engine.request_state(2) == before
    assert engine.metrics()["backpressure_count"] == 1

    assert engine.finish_request(1) == (0,)
    recovered = engine.step(q, k, v, request_ids=[2])
    assert recovered.status == DecodeEngine.STEP_OK
    assert engine.cache.request_block_ids(2) == (0,)
    assert engine.validate_invariants()


def test_decode_engine_preserves_explicit_request_row_order():
    torch.manual_seed(273)
    engine = DecodeEngine(_make_cache())
    reference_cache = _make_cache()
    engine.add_request(10)
    engine.add_request(20)
    engine.admit()

    q, k, v = _step_inputs(2)
    expected_append, expected_output = _reference_step(reference_cache, [20, 10], q, k, v)
    result = engine.step(q, k, v, request_ids=[20, 10])

    assert result.request_ids == (20, 10)
    _assert_step_matches(result, expected_append, expected_output)
    assert engine.validate_invariants()


def test_decode_engine_rejects_invalid_lifecycle_transitions():
    engine = DecodeEngine(_make_cache())
    engine.add_request("waiting")

    with pytest.raises(RuntimeError, match="waiting, expected active"):
        engine.finish_request("waiting")
    with pytest.raises(ValueError, match="at least one active"):
        q, k, v = _step_inputs(1)
        engine.step(q, k, v)

    engine.admit(["waiting"])
    with pytest.raises(RuntimeError, match="already exists"):
        engine.add_request("waiting")
    with pytest.raises(ValueError, match="unique"):
        engine.admit(["waiting", "waiting"])


def test_decode_engine_optional_profile_ranges_preserve_cpu_reference_step():
    engine = DecodeEngine(_make_cache(), profile_ranges=True)
    engine.add_request(1)
    engine.admit([1])
    q, k, v = _step_inputs(1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as prof:
        result = engine.step(q, k, v)

    keys = {event.key for event in prof.key_averages()}
    assert result.status == DecodeEngine.STEP_OK
    assert {
        PROFILE_RANGE_PREFLIGHT,
        PROFILE_RANGE_APPEND,
        PROFILE_RANGE_DECODE,
    } <= keys
    assert engine.profile_ranges is True
    assert engine.validate_invariants()


def test_decode_engine_rejects_non_bool_profile_ranges():
    with pytest.raises(ValueError, match="profile_ranges"):
        DecodeEngine(_make_cache(), profile_ranges="yes")


def test_scheduler_managed_engine_applies_decisions_and_releases_commitment():
    torch.manual_seed(275)
    engine = DecodeEngine(_make_cache(block_size=2, max_blocks=4))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=1)
    )
    assert engine.submit_request(RequestSpec("a", 0, 4, 0)).status == DecodeEngine.WAITING
    assert engine.submit_request(RequestSpec("b", 0, 4, 1)).status == DecodeEngine.WAITING

    snapshot = engine.scheduling_snapshot(logical_step=0)
    decision = scheduler.plan(snapshot)
    applied = engine.apply_scheduler_decision(decision)

    assert applied == (
        AdmissionResult("a", DecodeEngine.ACTIVE),
        AdmissionResult("b", DecodeEngine.ACTIVE),
    )
    assert decision.runnable_ids == ("a",)
    assert decision.deferred_ids == ("b",)
    assert engine.metrics()["committed_blocks"] == 4
    assert engine.metrics()["committed_but_unallocated_blocks"] == 4

    q, k, v = _step_inputs(1)
    first = engine.step(q, k, v, request_ids=["a"])
    assert first.status == DecodeEngine.STEP_OK
    assert engine.metrics()["committed_but_unallocated_blocks"] == 3

    snapshot = engine.scheduling_snapshot(
        logical_step=1,
        active_service_wait_steps={"a": 0, "b": 1},
    )
    decision = scheduler.plan(snapshot)
    assert decision.runnable_ids == ("b",)
    assert engine.apply_scheduler_decision(decision) == ()
    second = engine.step(q, k, v, request_ids=["b"])
    assert second.status == DecodeEngine.STEP_OK

    assert engine.finish_request("a") == (0,)
    assert engine.metrics()["committed_blocks"] == 2
    assert engine.metrics()["committed_but_unallocated_blocks"] == 1
    assert engine.validate_invariants()


def test_scheduler_managed_engine_counts_shared_prefix_once_and_private_tail_per_request():
    cache = _make_cache(
        block_size=2,
        max_blocks=6,
        prefix_cache_capacity_blocks=2,
    )
    engine = DecodeEngine(cache)
    prefix_shape = (1, 2, 1, 2, 4)
    prefix_k = torch.arange(
        16,
        dtype=cache.dtype,
        device=cache.device,
    ).reshape(prefix_shape)
    prefix_v = prefix_k + 100
    prefix_ids = engine.register_prefix("system", prefix_k, prefix_v)["block_ids"]
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=2)
    )
    engine.submit_request(RequestSpec("a", 4, 2, 0, prefix_id="system"))
    engine.submit_request(RequestSpec("b", 4, 2, 1, prefix_id="system"))

    snapshot = engine.scheduling_snapshot(logical_step=0)
    assert snapshot.resident_prefix_blocks == 2
    assert [item.shared_prefix_blocks for item in snapshot.waiting] == [2, 2]
    decision = scheduler.plan(snapshot)
    assert decision.committed_blocks_before == 2
    assert decision.committed_blocks_after == 4
    assert decision.needed_physical_blocks_now == 2

    engine.apply_scheduler_decision(decision)
    assert cache.request_block_ids("a") == prefix_ids
    assert cache.request_block_ids("b") == prefix_ids
    assert engine.metrics()["committed_blocks"] == 4
    assert engine.metrics()["committed_but_unallocated_blocks"] == 2
    assert cache.num_used_blocks == 2

    q, k, v = _step_inputs(2)
    result = engine.step(q, k, v, request_ids=["a", "b"])
    assert result.status == DecodeEngine.STEP_OK
    assert cache.request_state("a")["seq_len"] == 5
    assert cache.request_state("b")["seq_len"] == 5
    assert cache.metrics()["saved_prefix_blocks"] == 2
    assert cache.num_used_blocks == 4
    assert engine.metrics()["committed_but_unallocated_blocks"] == 0

    released_a = engine.finish_request("a")
    assert len(released_a) == 1
    assert cache.prefix_state("system")["active_refcount"] == 1
    assert engine.metrics()["committed_blocks"] == 3
    assert cache.num_used_blocks == 3

    released_b = engine.finish_request("b")
    assert len(released_b) == 1
    assert cache.prefix_state("system")["active_refcount"] == 0
    assert engine.metrics()["committed_blocks"] == 2
    assert cache.num_used_blocks == 2
    assert engine.validate_invariants()


def test_scheduler_caches_validated_shared_prefix_blocks_off_the_step_hot_path(
    monkeypatch,
):
    cache = _make_cache(
        block_size=2,
        max_blocks=6,
        prefix_cache_capacity_blocks=2,
    )
    engine = DecodeEngine(cache)
    prefix = torch.zeros(
        (1, 2, 1, 2, 4),
        dtype=cache.dtype,
        device=cache.device,
    )
    engine.register_prefix("system", prefix, prefix)

    prefix_state = cache.prefix_state
    lookup_count = 0

    def counted_prefix_state(prefix_id):
        nonlocal lookup_count
        lookup_count += 1
        return prefix_state(prefix_id)

    monkeypatch.setattr(cache, "prefix_state", counted_prefix_state)
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=2)
    )
    engine.submit_request(RequestSpec("a", 4, 2, 0, prefix_id="system"))
    engine.submit_request(RequestSpec("b", 4, 2, 1, prefix_id="system"))
    assert lookup_count == 2

    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    engine.apply_scheduler_decision(decision)
    q, k, v = _step_inputs(2)
    assert engine.step(q, k, v, request_ids=["a", "b"]).status == DecodeEngine.STEP_OK
    scheduler.plan(engine.scheduling_snapshot(logical_step=1))
    engine.metrics()
    assert engine.validate_invariants()

    assert lookup_count == 2


def test_scheduler_prefix_spec_requires_resident_exact_full_context():
    cache = _make_cache(
        block_size=2,
        max_blocks=4,
        prefix_cache_capacity_blocks=1,
    )
    engine = DecodeEngine(cache)
    with pytest.raises(ValueError, match="must be resident"):
        engine.submit_request(RequestSpec("missing", 2, 1, 0, prefix_id="missing"))
    assert not engine.metrics()["scheduler_managed"]

    prefix = torch.zeros(
        (1, 1, 1, 2, 4),
        dtype=cache.dtype,
        device=cache.device,
    )
    engine.register_prefix("system", prefix, prefix)
    with pytest.raises(ValueError, match="full initial context"):
        engine.submit_request(RequestSpec("mismatch", 4, 1, 0, prefix_id="system"))
    assert not engine.metrics()["scheduler_managed"]

    engine.submit_request(RequestSpec("valid", 2, 1, 0, prefix_id="system"))
    with pytest.raises(RuntimeError, match="before request submission"):
        engine.register_prefix("later", prefix, prefix)


def test_scheduler_decision_stale_version_is_rejected_without_partial_admission():
    engine = DecodeEngine(_make_cache(block_size=2, max_blocks=4))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=2)
    )
    engine.submit_request(RequestSpec("first", 0, 2, 0))
    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    before_version = engine.state_version
    engine.submit_request(RequestSpec("later", 0, 2, 1))

    with pytest.raises(RuntimeError, match="stale scheduler decision"):
        engine.apply_scheduler_decision(decision)

    assert engine.state_version == before_version + 1
    assert engine.request_state("first")["status"] == DecodeEngine.WAITING
    assert engine.request_state("later")["status"] == DecodeEngine.WAITING
    assert engine.cache.num_used_blocks == 0
    assert engine.metrics()["stale_decision_count"] == 1


def test_scheduler_decision_validation_is_atomic_and_rejects_external_cache_mutation():
    engine = DecodeEngine(_make_cache(block_size=2, max_blocks=4))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    engine.submit_request(RequestSpec("request", 0, 2, 0))
    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    malformed = replace(
        decision,
        committed_blocks_after=decision.committed_blocks_after + 1,
    )

    with pytest.raises(ValueError, match="committed_blocks_after"):
        engine.apply_scheduler_decision(malformed)
    assert engine.request_state("request")["status"] == DecodeEngine.WAITING
    assert engine.cache.num_used_blocks == 0

    engine.cache.add_request("request")
    with pytest.raises(RuntimeError, match="mutated outside"):
        engine.apply_scheduler_decision(decision)
    assert engine.request_state("request")["status"] == DecodeEngine.WAITING
    assert engine.metrics()["stale_decision_count"] == 1

    populated_cache = _make_cache()
    populated_cache.add_request("preexisting")
    populated_engine = DecodeEngine(populated_cache)
    with pytest.raises(RuntimeError, match="requires an empty PagedKVCache"):
        populated_engine.submit_request(RequestSpec("new", 0, 2, 0))


def test_scheduler_rejects_impossible_request_and_keeps_it_out_of_cache():
    engine = DecodeEngine(_make_cache(block_size=2, max_blocks=2))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    engine.submit_request(RequestSpec("too-large", 0, 5, 0))
    engine.submit_request(RequestSpec("small", 0, 2, 1))

    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    applied = engine.apply_scheduler_decision(decision)

    assert applied == (
        AdmissionResult("small", DecodeEngine.ACTIVE),
        AdmissionResult("too-large", DecodeEngine.REJECTED),
    )
    assert engine.request_state("too-large")["status"] == DecodeEngine.REJECTED
    with pytest.raises(KeyError, match="unknown request_id"):
        engine.cache.request_state("too-large")
    assert engine.metrics()["rejected_requests"] == 1
    assert engine.validate_invariants()


def test_scheduler_initial_context_must_be_seeded_then_replanned():
    engine = DecodeEngine(_make_cache(block_size=2, max_blocks=2))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    engine.submit_request(RequestSpec("context", 2, 2, 0))
    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    engine.apply_scheduler_decision(decision)

    q, k, v = _step_inputs(1)
    with pytest.raises(RuntimeError, match="requires an applied decision"):
        engine.step(q, k, v, request_ids=["context"])
    assert engine.prefill_request("context", k, v) == 1
    assert engine.prefill_request("context", k, v) == 2
    with pytest.raises(RuntimeError, match="already fully seeded"):
        engine.prefill_request("context", k, v)

    snapshot = engine.scheduling_snapshot(logical_step=1)
    assert snapshot.active[0].seq_len == 2
    assert snapshot.active[0].remaining_tokens == 2
    decision = scheduler.plan(snapshot)
    engine.apply_scheduler_decision(decision)
    result = engine.step(q, k, v, request_ids=["context"])
    assert result.status == DecodeEngine.STEP_OK
    assert engine.request_state("context")["cache"]["seq_len"] == 3
    assert engine.validate_invariants()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_decode_engine_fused_append_and_triton_decode_match_reference():
    pytest.importorskip("triton")
    torch.manual_seed(277)
    engine_cache = _make_cache(
        device="cuda",
        dtype=torch.float16,
        num_kv_heads=1,
        head_dim=64,
        block_size=32,
        max_blocks=4,
    )
    reference_cache = _make_cache(
        device="cuda",
        dtype=torch.float16,
        num_kv_heads=1,
        head_dim=64,
        block_size=32,
        max_blocks=4,
    )
    engine = DecodeEngine(
        engine_cache,
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=2,
    )
    engine.add_request(10)
    engine.add_request(20)
    engine.admit()

    q, k, v = _step_inputs(
        2,
        device="cuda",
        dtype=torch.float16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=64,
    )
    expected_append, expected_output = _reference_step(reference_cache, [10, 20], q, k, v)
    result = engine.step(q, k, v)
    torch.cuda.synchronize()

    assert result.status == DecodeEngine.STEP_OK
    torch.testing.assert_close(result.output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(result.positions, expected_append.positions)
    torch.testing.assert_close(result.block_tables, expected_append.block_tables)
    torch.testing.assert_close(result.seq_lens, expected_append.seq_lens)
    torch.testing.assert_close(engine_cache.k_cache, reference_cache.k_cache, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(engine_cache.v_cache, reference_cache.v_cache, rtol=0.0, atol=0.0)
    assert engine.validate_invariants()
