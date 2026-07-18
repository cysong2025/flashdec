"""DecodeEngine coverage for sequential multi-layer token transactions."""

from dataclasses import replace
import shutil

import pytest


torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.engine import (
    DecodeEngine,
    DecodeLayerResult,
    DecodeStepResult,
    DecodeStepTransaction,
)
from flashdec.paged_reference import paged_decode_attention_ref
from flashdec.rope import apply_rope
from flashdec.scheduler import BlockAwareScheduler, RequestSpec, SchedulerConfig


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = (
    "CUDA GPU, CUDA_HOME, and nvcc are required for fused multi-layer tests"
)
CUDA_DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    CUDA_DTYPES.append(torch.bfloat16)


def _cache(num_layers=2, block_size=2, max_blocks=4):
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=4,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=torch.float32,
        device="cpu",
    )


def _inputs(batch_size, seed):
    torch.manual_seed(seed)
    q = torch.randn((batch_size, 2, 4), dtype=torch.float32)
    k = torch.randn((batch_size, 1, 4), dtype=torch.float32)
    v = torch.randn_like(k)
    return q, k, v


def _cuda_cache(dtype):
    return PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=32,
        max_blocks=4,
        dtype=dtype,
        device="cuda",
    )


def _cuda_inputs(batch_size, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn((batch_size, 4, 64), device="cuda", dtype=dtype)
    k = torch.randn((batch_size, 2, 64), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def _cuda_tolerances(dtype):
    if dtype == torch.float16:
        return 2e-2, 2e-2, 3e-3, 3e-3
    return 4e-2, 4e-2, 2e-2, 2e-2


@pytest.mark.parametrize("num_layers", [2, 4])
def test_engine_multi_layer_outputs_match_per_layer_reference_and_commit_once(num_layers):
    engine = DecodeEngine(_cache(num_layers=num_layers))
    engine.add_request(10)
    engine.add_request(20)
    engine.admit()

    transaction = engine.begin_step([20, 10])
    assert isinstance(transaction, DecodeStepTransaction)
    assert flashdec.DecodeStepTransaction is DecodeStepTransaction
    assert flashdec.DecodeLayerResult is DecodeLayerResult
    assert transaction.request_ids == (20, 10)
    assert transaction.positions.tolist() == [0, 0]
    assert transaction.effective_seq_lens.tolist() == [1, 1]

    last_result = None
    for layer_idx in range(num_layers):
        q, k, v = _inputs(2, 601 + layer_idx)
        result = engine.step_layer(transaction, layer_idx, q, k, v)
        assert isinstance(result, DecodeLayerResult)
        assert result.request_ids == (20, 10)
        assert result.layer_idx == layer_idx
        assert engine.cache.seq_lens_tensor([20, 10]).tolist() == [0, 0]

        q_rotated = apply_rope(q, transaction.positions)
        k_rotated = apply_rope(k, transaction.positions)
        expected = paged_decode_attention_ref(
            q_rotated,
            engine.cache.k_cache[layer_idx],
            engine.cache.v_cache[layer_idx],
            transaction.block_tables,
            transaction.effective_seq_lens,
        )
        torch.testing.assert_close(result.output, expected)
        for row, block_id in enumerate(transaction.physical_block_ids.tolist()):
            torch.testing.assert_close(
                engine.cache.k_cache[layer_idx, block_id, :, 0, :],
                k_rotated[row],
            )
            torch.testing.assert_close(
                engine.cache.v_cache[layer_idx, block_id, :, 0, :],
                v[row],
            )
        last_result = result

    committed = engine.commit_step(transaction)
    assert isinstance(committed, DecodeStepResult)
    assert committed.status == DecodeEngine.STEP_OK
    assert committed.request_ids == (20, 10)
    torch.testing.assert_close(committed.output, last_result.output)
    assert committed.seq_lens.tolist() == [1, 1]
    assert engine.metrics()["completed_step_count"] == 1
    assert engine.metrics()["appended_token_count"] == 2
    assert engine.metrics()["transaction_layer_step_count"] == num_layers
    assert engine.metrics()["open_step_transaction_count"] == 0
    released = engine.finish_request(20)
    assert len(released) == 1
    assert engine.cache.num_used_blocks == 1
    assert engine.validate_invariants()


def test_engine_layer_failure_automatically_aborts_whole_token():
    engine = DecodeEngine(_cache(num_layers=2, block_size=1, max_blocks=1))
    engine.add_request("request")
    engine.admit()
    transaction = engine.begin_step(["request"])
    q, k, v = _inputs(1, 611)
    engine.step_layer(transaction, 0, q, k, v)
    bad_q = torch.ones((1, 2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="q shape"):
        engine.step_layer(transaction, 1, bad_q, k, v)

    assert engine.cache.request_state("request")["seq_len"] == 0
    assert engine.cache.request_block_ids("request") == ()
    assert engine.cache.num_used_blocks == 0
    assert engine.metrics()["transaction_abort_count"] == 1
    assert engine.metrics()["open_step_transaction_count"] == 0
    with pytest.raises(RuntimeError, match="no open"):
        engine.abort_step(transaction)
    assert engine.validate_invariants()


def test_engine_early_commit_stays_open_until_explicit_abort():
    engine = DecodeEngine(_cache(num_layers=2))
    engine.add_request(1)
    engine.admit()
    transaction = engine.begin_step([1])
    q, k, v = _inputs(1, 613)
    engine.step_layer(transaction, 0, q, k, v)

    with pytest.raises(RuntimeError, match="before all layers"):
        engine.commit_step(transaction)
    assert engine.metrics()["open_step_transaction_count"] == 1
    aborted = engine.abort_step(transaction)
    assert aborted.state == "aborted"
    assert engine.cache.request_state(1)["seq_len"] == 0
    assert engine.validate_invariants()


def test_engine_rejects_stale_transaction_handle_without_aborting_owner():
    engine = DecodeEngine(_cache(num_layers=2))
    engine.add_request(1)
    engine.admit()
    transaction = engine.begin_step([1])
    stale = replace(transaction, engine_version=transaction.engine_version + 1)
    q, k, v = _inputs(1, 615)

    with pytest.raises(RuntimeError, match="stale or invalid"):
        engine.step_layer(stale, 0, q, k, v)
    assert engine.metrics()["open_step_transaction_count"] == 1
    assert engine.cache.request_state(1)["seq_len"] == 0
    engine.abort_step(transaction)
    assert engine.validate_invariants()


def test_engine_open_transaction_blocks_lifecycle_and_single_step_calls():
    engine = DecodeEngine(_cache(num_layers=2))
    engine.add_request("active")
    engine.add_request("waiting")
    engine.admit(["active"])
    transaction = engine.begin_step(["active"])
    q, k, v = _inputs(1, 617)

    for action in (
        lambda: engine.add_request("new"),
        lambda: engine.admit(["waiting"]),
        lambda: engine.finish_request("active"),
        lambda: engine.cancel_request("active"),
        lambda: engine.step(q, k, v, ["active"]),
    ):
        with pytest.raises(RuntimeError, match="open decode step transaction"):
            action()

    engine.abort_step(transaction)
    assert engine.admit(["waiting"])[0].status == DecodeEngine.ACTIVE
    assert engine.validate_invariants()


def test_scheduler_cannot_replan_during_open_engine_transaction():
    engine = DecodeEngine(_cache(num_layers=2, max_blocks=2))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    engine.submit_request(RequestSpec("request", 0, 2, 0))
    decision = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    engine.apply_scheduler_decision(decision)
    transaction = engine.begin_step(["request"])

    with pytest.raises(RuntimeError, match="open decode step transaction"):
        engine.scheduling_snapshot(logical_step=1)
    with pytest.raises(RuntimeError, match="open decode step transaction"):
        engine.apply_scheduler_decision(decision)

    engine.abort_step(transaction)
    with pytest.raises(RuntimeError, match="stale scheduler decision"):
        engine.apply_scheduler_decision(decision)
    assert engine.scheduling_snapshot(logical_step=1).active[0].seq_len == 0
    assert engine.validate_invariants()


def test_multi_layer_fused_backend_cpu_failure_automatically_aborts_transaction():
    engine = DecodeEngine(_cache(num_layers=2), append_backend="fused_cuda")
    engine.add_request(1)
    engine.admit()
    transaction = engine.begin_step([1])
    q, k, v = _inputs(1, 621)

    with pytest.raises(ValueError, match="CUDA-resident"):
        engine.step_layer(transaction, 0, q, k, v)
    assert engine.cache.num_used_blocks == 0
    assert engine.cache.request_state(1)["seq_len"] == 0
    assert engine.metrics()["transaction_abort_count"] == 1
    assert engine.metrics()["open_step_transaction_count"] == 0
    assert engine.validate_invariants()


def test_engine_routes_fused_transaction_through_authoritative_cache_api(
    monkeypatch,
):
    engine = DecodeEngine(_cache(num_layers=2), append_backend="fused_cuda")
    engine.add_request(1)
    engine.admit()
    fused_layers = []

    def fused_path(
        transaction,
        layer_idx,
        q,
        k,
        v,
        *,
        rotary_dim=None,
        base=10_000.0,
    ):
        fused_layers.append(layer_idx)
        view = engine.cache.transaction_view(transaction)
        q_rotated = apply_rope(
            q,
            view.positions,
            rotary_dim=rotary_dim,
            base=base,
        )
        k_rotated = apply_rope(
            k,
            view.positions,
            rotary_dim=rotary_dim,
            base=base,
        )
        latest = engine.cache.write_token_layer(
            transaction,
            layer_idx,
            k_rotated,
            v,
        )
        return q_rotated, latest

    monkeypatch.setattr(
        engine.cache,
        "write_token_layer_fused_cuda",
        fused_path,
    )

    transaction = engine.begin_step([1])
    for layer_idx in range(2):
        q, k, v = _inputs(1, 625 + layer_idx)
        engine.step_layer(transaction, layer_idx, q, k, v)
    result = engine.commit_step(transaction)

    assert fused_layers == [0, 1]
    assert result.status == DecodeEngine.STEP_OK
    assert result.seq_lens.tolist() == [1]
    assert engine.validate_invariants()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_fused_cache_ignores_forged_detached_transaction_locations_and_rolls_back(
    dtype,
):
    flashdec.load_fused_rope_kv_append_extension()
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=2,
        max_blocks=4,
        dtype=dtype,
        device="cuda",
    )
    cache.add_request("request")
    transaction = cache.begin_token(["request"])
    authoritative_block = transaction.physical_block_ids.item()
    forged = replace(
        transaction,
        positions=torch.full_like(transaction.positions, 5),
        physical_block_ids=torch.full_like(
            transaction.physical_block_ids, cache.max_blocks - 1
        ),
        block_offsets=torch.full_like(
            transaction.block_offsets, cache.block_size - 1
        ),
    )

    atol, rtol, cache_atol, cache_rtol = _cuda_tolerances(dtype)
    for layer_idx in range(cache.num_layers):
        q, k, v = _cuda_inputs(1, dtype, 690 + layer_idx)
        q_rotated, latest = cache.write_token_layer_fused_cuda(
            forged,
            layer_idx,
            q,
            k,
            v,
        )
        expected_q = apply_rope(q, transaction.positions)
        expected_k = apply_rope(k, transaction.positions)
        torch.testing.assert_close(q_rotated, expected_q, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            cache.k_cache[layer_idx, authoritative_block, :, 0, :],
            expected_k[0],
            atol=cache_atol,
            rtol=cache_rtol,
        )
        torch.testing.assert_close(
            cache.v_cache[layer_idx, authoritative_block, :, 0, :],
            v[0],
            atol=cache_atol,
            rtol=cache_rtol,
        )
        assert latest.next_layer_idx == layer_idx + 1

    cache.commit_token(forged)
    assert cache.request_state("request")["seq_len"] == 1
    assert cache.request_block_ids("request") == (authoritative_block,)

    tail = cache.begin_token(["request"])
    for layer_idx in range(cache.num_layers):
        _q, k, v = _cuda_inputs(1, dtype, 696 + layer_idx)
        cache.write_token_layer(tail, layer_idx, k, v)
    cache.commit_token(tail)

    rollback = cache.begin_token(["request"])
    rollback_block = rollback.physical_block_ids.item()
    forged_rollback = replace(
        rollback,
        positions=torch.full_like(rollback.positions, 7),
        physical_block_ids=torch.full_like(
            rollback.physical_block_ids, cache.max_blocks - 1
        ),
        block_offsets=torch.full_like(
            rollback.block_offsets, cache.block_size - 1
        ),
    )
    q, k, v = _cuda_inputs(1, dtype, 699)
    cache.write_token_layer_fused_cuda(forged_rollback, 0, q, k, v)
    cache.abort_token(forged_rollback)

    assert rollback_block != authoritative_block
    assert cache.request_state("request")["seq_len"] == 2
    assert cache.request_block_ids("request") == (authoritative_block,)
    assert cache.metrics()["transaction_rollback_block_count"] == 1
    assert cache.validate_invariants()


def test_multi_layer_engine_rejects_unfused_cuda_transaction_backend():
    engine = DecodeEngine(_cache(num_layers=2), append_backend="cuda")
    engine.add_request(1)
    engine.admit()

    with pytest.raises(
        RuntimeError,
        match="append_backend='torch' or 'fused_cuda'",
    ):
        engine.begin_step([1])
    assert engine.cache.num_used_blocks == 0
    assert engine.validate_invariants()


def test_single_layer_step_remains_compatible_through_transaction_wrapper():
    engine = DecodeEngine(_cache(num_layers=1))
    engine.add_request(1)
    engine.admit()
    q, k, v = _inputs(1, 619)

    result = engine.step(q, k, v, [1])

    assert result.status == DecodeEngine.STEP_OK
    assert result.seq_lens.tolist() == [1]
    assert engine.cache.metrics()["transaction_commit_count"] == 1
    assert engine.metrics()["completed_step_count"] == 1
    assert engine.validate_invariants()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_multi_layer_fused_cuda_triton_matches_torch_reference(dtype):
    pytest.importorskip("triton")
    fused_engine = DecodeEngine(
        _cuda_cache(dtype),
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=2,
    )
    reference_engine = DecodeEngine(
        _cuda_cache(dtype),
        append_backend="torch",
        decode_backend="reference",
    )
    for engine in (fused_engine, reference_engine):
        engine.add_request(10)
        engine.add_request(20)
        engine.admit()

    output_rtol, output_atol, cache_rtol, cache_atol = _cuda_tolerances(dtype)
    for step_idx in range(2):
        fused_transaction = fused_engine.begin_step([20, 10])
        reference_transaction = reference_engine.begin_step([20, 10])
        assert fused_transaction.positions.tolist() == [step_idx, step_idx]
        torch.testing.assert_close(
            fused_transaction.physical_block_ids,
            reference_transaction.physical_block_ids,
        )
        torch.testing.assert_close(
            fused_transaction.block_offsets,
            reference_transaction.block_offsets,
        )

        for layer_idx in range(2):
            q, k, v = _cuda_inputs(
                2,
                dtype,
                seed=701 + step_idx * 10 + layer_idx,
            )
            fused_result = fused_engine.step_layer(
                fused_transaction,
                layer_idx,
                q,
                k,
                v,
                rotary_dim=32,
            )
            reference_result = reference_engine.step_layer(
                reference_transaction,
                layer_idx,
                q,
                k,
                v,
                rotary_dim=32,
            )
            torch.cuda.synchronize()

            torch.testing.assert_close(
                fused_result.output,
                reference_result.output,
                rtol=output_rtol,
                atol=output_atol,
            )
            torch.testing.assert_close(
                fused_engine.cache.k_cache[layer_idx],
                reference_engine.cache.k_cache[layer_idx],
                rtol=cache_rtol,
                atol=cache_atol,
            )
            torch.testing.assert_close(
                fused_engine.cache.v_cache[layer_idx],
                reference_engine.cache.v_cache[layer_idx],
                rtol=0.0,
                atol=0.0,
            )
            assert fused_engine.cache.seq_lens_tensor([20, 10]).tolist() == [
                step_idx,
                step_idx,
            ]

        fused_commit = fused_engine.commit_step(fused_transaction)
        reference_commit = reference_engine.commit_step(reference_transaction)
        assert fused_commit.seq_lens.tolist() == [step_idx + 1, step_idx + 1]
        torch.testing.assert_close(fused_commit.seq_lens, reference_commit.seq_lens)

    assert fused_engine.cache.metrics()["transaction_layer_write_count"] == 4
    assert fused_engine.metrics()["transaction_layer_step_count"] == 4
    assert fused_engine.validate_invariants()
    assert reference_engine.validate_invariants()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_multi_layer_fused_cuda_write_failure_rolls_back_reserved_block():
    engine = DecodeEngine(
        _cuda_cache(torch.float16),
        append_backend="fused_cuda",
        decode_backend="reference",
    )
    engine.add_request("request")
    engine.admit()
    transaction = engine.begin_step(["request"])
    q, k, v = _cuda_inputs(1, torch.float16, seed=731)
    engine.step_layer(transaction, 0, q, k, v)
    bad_k = torch.randn(
        (1, 64, 2),
        device="cuda",
        dtype=torch.float16,
    ).transpose(1, 2)
    assert bad_k.shape == k.shape
    assert not bad_k.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        engine.step_layer(transaction, 1, q, bad_k, v)

    assert engine.cache.request_state("request")["seq_len"] == 0
    assert engine.cache.request_block_ids("request") == ()
    assert engine.cache.num_used_blocks == 0
    assert engine.cache.metrics()["transaction_rollback_block_count"] == 1
    assert engine.cache.metrics()["transaction_layer_write_count"] == 1
    assert engine.metrics()["transaction_abort_count"] == 1
    assert engine.validate_invariants()
