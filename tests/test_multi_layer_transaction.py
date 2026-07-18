"""Reference coverage for multi-layer KV token transactions."""

import pytest


torch = pytest.importorskip("torch")

import flashdec
from flashdec.cache import KVTokenTransactionView, PagedKVCache


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cache(num_layers=2, block_size=2, max_blocks=4):
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=2,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=torch.float32,
        device=DEVICE,
    )


def _values(batch_size, value):
    return torch.full(
        (batch_size, 1, 2),
        float(value),
        dtype=torch.float32,
        device=DEVICE,
    )


@pytest.mark.parametrize("num_layers", [2, 4])
def test_multi_layer_transaction_commits_seq_len_once_and_reuses_locations(num_layers):
    cache = _cache(num_layers=num_layers)
    cache.add_request(10).add_request(20)

    transaction = cache.begin_token([10, 20])
    assert isinstance(transaction, KVTokenTransactionView)
    assert flashdec.KVTokenTransactionView is KVTokenTransactionView
    assert transaction.state == "open"
    assert transaction.next_layer_idx == 0
    assert transaction.positions.tolist() == [0, 0]
    assert transaction.block_offsets.tolist() == [0, 0]
    assert transaction.effective_seq_lens.tolist() == [1, 1]
    assert transaction.positions.dtype == torch.int64
    assert transaction.physical_block_ids.dtype == torch.int64
    assert transaction.block_offsets.dtype == torch.int64
    assert cache.seq_lens_tensor([10, 20]).tolist() == [0, 0]

    physical_ids = transaction.physical_block_ids.tolist()
    for layer_idx in range(num_layers):
        k = _values(2, layer_idx + 1)
        v = _values(2, layer_idx + 101)
        latest = cache.write_token_layer(transaction, layer_idx, k, v)
        assert latest.next_layer_idx == layer_idx + 1
        assert latest.physical_block_ids.tolist() == physical_ids
        for row, block_id in enumerate(physical_ids):
            torch.testing.assert_close(
                cache.k_cache[layer_idx, block_id, :, 0, :], k[row]
            )
            torch.testing.assert_close(
                cache.v_cache[layer_idx, block_id, :, 0, :], v[row]
            )
        assert cache.seq_lens_tensor([10, 20]).tolist() == [0, 0]

    committed = cache.commit_token(transaction)
    assert committed.state == "committed"
    assert cache.seq_lens_tensor([10, 20]).tolist() == [1, 1]
    assert cache.metrics()["transaction_commit_count"] == 1
    assert cache.metrics()["transaction_layer_write_count"] == num_layers
    assert cache.metrics()["open_transaction_count"] == 0
    assert cache.validate_invariants()

    for layer_idx in range(num_layers):
        dense_k, dense_v, seq_lens = cache.to_dense(layer_idx, [10, 20])
        torch.testing.assert_close(dense_k[:, 0], _values(2, layer_idx + 1))
        torch.testing.assert_close(dense_v[:, 0], _values(2, layer_idx + 101))
        assert seq_lens.tolist() == [1, 1]


def test_transaction_mixed_tail_and_boundary_only_reserves_needed_block():
    cache = _cache(num_layers=2, block_size=2, max_blocks=3)
    cache.add_request("tail")
    first = cache.begin_token(["tail"])
    cache.write_token_layer(first, 0, _values(1, 1), _values(1, 101))
    cache.write_token_layer(first, 1, _values(1, 2), _values(1, 102))
    cache.commit_token(first)
    tail_block = cache.request_block_ids("tail")[0]

    cache.add_request("boundary")
    transaction = cache.begin_token(["tail", "boundary"])
    assert transaction.positions.tolist() == [1, 0]
    assert transaction.block_offsets.tolist() == [1, 0]
    assert transaction.physical_block_ids[0].item() == tail_block
    assert cache.metrics()["reserved_transaction_blocks"] == 1
    assert cache.metrics()["pending_request_count"] == 2
    assert cache.validate_invariants()

    cache.abort_token(transaction)
    assert cache.request_state("tail")["seq_len"] == 1
    assert cache.request_state("boundary")["seq_len"] == 0
    assert cache.request_block_ids("tail") == (tail_block,)
    assert cache.request_block_ids("boundary") == ()
    assert cache.validate_invariants()


def test_transaction_capacity_failure_has_no_request_or_ownership_mutation():
    cache = _cache(num_layers=2, block_size=1, max_blocks=1)
    cache.add_request(1).add_request(2)
    before_version = cache.state_version
    before_one = cache.request_state(1)
    before_two = cache.request_state(2)

    with pytest.raises(RuntimeError, match="out of physical blocks"):
        cache.begin_token([1, 2])

    assert cache.state_version == before_version
    assert cache.request_state(1) == before_one
    assert cache.request_state(2) == before_two
    assert cache.num_used_blocks == 0
    assert cache.metrics()["open_transaction_count"] == 0
    assert cache.metrics()["transaction_failure_count"] == 1
    assert cache.validate_invariants()


def test_begin_token_location_proof_failure_rolls_back_reservation(monkeypatch):
    cache = _cache(num_layers=2, block_size=1, max_blocks=2)
    cache.add_request("request")
    before_version = cache.state_version
    before_failures = cache.metrics()["transaction_failure_count"]
    validate_locations = cache._validate_reserved_transaction_locations

    def fail_location_proof(*_args, **_kwargs):
        raise RuntimeError("injected transaction location proof failure")

    monkeypatch.setattr(
        cache,
        "_validate_reserved_transaction_locations",
        fail_location_proof,
    )
    with pytest.raises(RuntimeError, match="injected transaction location proof"):
        cache.begin_token(["request"])

    assert cache.state_version == before_version
    assert cache.request_state("request")["seq_len"] == 0
    assert cache.request_block_ids("request") == ()
    assert cache.num_free_blocks == cache.max_blocks
    assert cache.metrics()["open_transaction_count"] == 0
    assert cache.metrics()["transaction_begin_count"] == 0
    assert cache.metrics()["transaction_failure_count"] == before_failures + 1
    assert cache.validate_invariants()

    monkeypatch.setattr(
        cache,
        "_validate_reserved_transaction_locations",
        validate_locations,
    )
    transaction = cache.begin_token(["request"])
    assert transaction.physical_block_ids.tolist() == [0]
    cache.abort_token(transaction)
    assert cache.validate_invariants()


def test_abort_hides_partial_layer_and_rolls_back_boundary_block():
    cache = _cache(num_layers=2, block_size=1, max_blocks=1)
    cache.add_request("request")
    transaction = cache.begin_token(["request"])
    reserved_block = transaction.physical_block_ids.item()
    cache.write_token_layer(transaction, 0, _values(1, 7), _values(1, 107))

    aborted = cache.abort_token(transaction)
    assert aborted.state == "aborted"
    assert cache.request_state("request")["seq_len"] == 0
    assert cache.request_block_ids("request") == ()
    assert cache.num_used_blocks == 0
    dense_k, dense_v, seq_lens = cache.to_dense(0, ["request"])
    assert seq_lens.tolist() == [0]
    assert torch.count_nonzero(dense_k).item() == 0
    assert torch.count_nonzero(dense_v).item() == 0
    assert cache.metrics()["transaction_rollback_block_count"] == 1

    replacement = cache.begin_token(["request"])
    assert replacement.physical_block_ids.item() == reserved_block
    cache.write_token_layer(replacement, 0, _values(1, 8), _values(1, 108))
    cache.write_token_layer(replacement, 1, _values(1, 9), _values(1, 109))
    cache.commit_token(replacement)
    assert cache.request_state("request")["seq_len"] == 1
    assert cache.metrics()["reuse_count"] == 1
    assert cache.validate_invariants()


def test_transaction_rejects_out_of_order_missing_and_terminal_transitions():
    cache = _cache(num_layers=2)
    cache.add_request(1)
    transaction = cache.begin_token([1])
    token = _values(1, 1)

    with pytest.raises(RuntimeError, match="requires layer 0"):
        cache.write_token_layer(transaction, 1, token, token)
    with pytest.raises(RuntimeError, match="before all layers"):
        cache.commit_token(transaction)
    with pytest.raises(RuntimeError, match="open token transaction"):
        cache.begin_token([1])
    with pytest.raises(RuntimeError, match="open token transaction"):
        cache.finish_request(1)
    with pytest.raises(RuntimeError, match="open token transaction"):
        cache.cancel_request(1)

    cache.write_token_layer(transaction, 0, token, token)
    with pytest.raises(RuntimeError, match="requires layer 1"):
        cache.write_token_layer(transaction, 0, token, token)
    cache.write_token_layer(transaction, 1, token, token)
    cache.commit_token(transaction)

    with pytest.raises(RuntimeError, match="already committed"):
        cache.commit_token(transaction)
    with pytest.raises(RuntimeError, match="already committed"):
        cache.abort_token(transaction)
    assert cache.validate_invariants()


def test_transaction_rejects_invalid_layer_values_before_write():
    cache = _cache(num_layers=2)
    cache.add_request(1)
    transaction = cache.begin_token([1])
    wrong = torch.ones((1, 1, 3), device=DEVICE, dtype=torch.float32)

    with pytest.raises(ValueError, match="shapes must match"):
        cache.write_token_layer(transaction, 0, wrong, wrong)
    latest = cache.transaction_view(transaction)
    assert latest.next_layer_idx == 0
    assert cache.request_state(1)["seq_len"] == 0
    cache.abort_token(transaction)
    assert cache.validate_invariants()


def test_single_layer_legacy_append_is_rejected_while_transaction_is_open():
    cache = _cache(num_layers=1)
    cache.add_request(1)
    transaction = cache.begin_token([1])
    token = _values(1, 1)

    with pytest.raises(RuntimeError, match="open token transaction"):
        cache.append(0, [1], token, token)
    cache.abort_token(transaction)
    cache.append(0, [1], token, token)
    assert cache.request_state(1)["seq_len"] == 1
    assert cache.validate_invariants()
