"""DecodeEngine coverage for sequential multi-layer token transactions."""

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

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


def test_multi_layer_engine_requires_reference_transaction_append_backend():
    engine = DecodeEngine(_cache(num_layers=2), append_backend="fused_cuda")
    engine.add_request(1)
    engine.admit()

    with pytest.raises(RuntimeError, match="append_backend='torch'"):
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
