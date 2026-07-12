"""Dynamic request lifecycle and decode-step coverage for DecodeEngine v1."""

import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.engine import AdmissionResult, DecodeEngine, DecodeStepResult
from flashdec.paged_reference import paged_decode_attention_ref
from flashdec.rope import rope_paged_kv_append_ref


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required for fused engine tests"


def _make_cache(device="cpu", dtype=torch.float32, num_kv_heads=1, head_dim=4, block_size=2, max_blocks=8):
    return PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device=device,
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
