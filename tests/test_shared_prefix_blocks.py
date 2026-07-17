"""Correctness coverage for R3 shared immutable prefix blocks."""

import pytest


torch = pytest.importorskip("torch")

from flashdec.cache import PagedKVCache


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cache(*, num_layers=2, block_size=2, max_blocks=6, prefix_capacity=4):
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=3,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=torch.float32,
        device=DEVICE,
        prefix_cache_capacity_blocks=prefix_capacity,
    )


def _prefix_blocks(cache, num_blocks, start=0):
    shape = (
        cache.num_layers,
        num_blocks,
        cache.num_kv_heads,
        cache.block_size,
        cache.head_dim,
    )
    count = 1
    for size in shape:
        count *= size
    k = torch.arange(start, start + count, device=cache.device, dtype=cache.dtype).reshape(
        shape
    )
    return k, k + 1_000


def _dense_layer(blocks, layer_idx):
    return blocks[layer_idx].permute(0, 2, 1, 3).reshape(
        -1, blocks.shape[2], blocks.shape[4]
    )


def test_shared_prefix_reuses_blocks_and_keeps_request_tail_private():
    cache = _cache(max_blocks=5, prefix_capacity=4)
    prefix_k, prefix_v = _prefix_blocks(cache, num_blocks=2)
    registered = cache.register_prefix("system", prefix_k, prefix_v)
    prefix_ids = registered["block_ids"]

    cache.add_request("first").add_request("second")
    assert cache.attach_prefix("first", "system") == prefix_ids
    assert cache.attach_prefix("second", "system") == prefix_ids
    assert cache.request_block_ids("first") == prefix_ids
    assert cache.request_block_ids("second") == prefix_ids
    assert cache.prefix_state("system")["active_refcount"] == 2

    for layer_idx in range(cache.num_layers):
        dense_k, dense_v, seq_lens = cache.to_dense(
            layer_idx, ["first", "second"]
        )
        expected_k = _dense_layer(prefix_k, layer_idx)
        expected_v = _dense_layer(prefix_v, layer_idx)
        torch.testing.assert_close(dense_k[0, :4], expected_k)
        torch.testing.assert_close(dense_k[1, :4], expected_k)
        torch.testing.assert_close(dense_v[0, :4], expected_v)
        torch.testing.assert_close(dense_v[1, :4], expected_v)
        assert seq_lens.tolist() == [4, 4]

    stored_prefix_k = cache.k_cache[:, list(prefix_ids)].clone()
    stored_prefix_v = cache.v_cache[:, list(prefix_ids)].clone()
    transaction = cache.begin_token(["first"])
    for layer_idx in range(cache.num_layers):
        k = torch.full(
            (1, 1, 3),
            100 + layer_idx,
            device=cache.device,
            dtype=cache.dtype,
        )
        v = torch.full(
            (1, 1, 3),
            200 + layer_idx,
            device=cache.device,
            dtype=cache.dtype,
        )
        cache.write_token_layer(transaction, layer_idx, k, v)
    cache.commit_token(transaction)

    torch.testing.assert_close(cache.k_cache[:, list(prefix_ids)], stored_prefix_k)
    torch.testing.assert_close(cache.v_cache[:, list(prefix_ids)], stored_prefix_v)
    first_blocks = cache.request_block_ids("first")
    assert first_blocks[:2] == prefix_ids
    assert len(first_blocks) == 3
    assert cache.request_block_ids("second") == prefix_ids
    assert cache.request_state("first")["seq_len"] == 5

    metrics = cache.metrics()
    assert metrics["used_blocks"] == 3
    assert metrics["active_tokens"] == 9
    assert metrics["physical_data_tokens"] == 5
    assert metrics["shared_prefix_blocks"] == 2
    assert metrics["saved_prefix_blocks"] == 2
    assert metrics["active_prefix_references"] == 2

    assert cache.finish_request("first") == (first_blocks[-1],)
    assert cache.prefix_state("system")["active_refcount"] == 1
    dense_k, _, seq_lens = cache.to_dense(0, ["second"])
    torch.testing.assert_close(dense_k[0, :4], _dense_layer(prefix_k, 0))
    assert seq_lens.tolist() == [4]

    assert cache.finish_request("second") == ()
    assert cache.prefix_state("system")["active_refcount"] == 0
    assert cache.num_used_blocks == 2
    assert cache.evict_prefix("system") == prefix_ids
    assert cache.num_used_blocks == 0
    assert cache.num_free_blocks == cache.max_blocks
    assert cache.validate_invariants()


def test_shared_prefix_lru_only_evicts_inactive_entries():
    cache = _cache(num_layers=1, max_blocks=3, prefix_capacity=2)
    first_k, first_v = _prefix_blocks(cache, 1, start=0)
    protected_k, protected_v = _prefix_blocks(cache, 1, start=100)
    newest_k, newest_v = _prefix_blocks(cache, 1, start=200)
    cache.register_prefix("old", first_k, first_v)
    cache.register_prefix("protected", protected_k, protected_v)

    cache.add_request("owner")
    protected_ids = cache.attach_prefix("owner", "protected")
    cache.register_prefix("new", newest_k, newest_v)

    with pytest.raises(KeyError, match="unknown prefix_id"):
        cache.prefix_state("old")
    assert cache.prefix_state("protected")["block_ids"] == protected_ids
    assert cache.prefix_state("new")["active_refcount"] == 0
    assert cache.metrics()["prefix_eviction_count"] == 1

    cache.add_request("lookup")
    with pytest.raises(KeyError, match="unknown prefix_id"):
        cache.attach_prefix("lookup", "old")
    cache.attach_prefix("lookup", "new")
    with pytest.raises(RuntimeError, match="active request references"):
        cache.evict_prefix("protected")

    assert cache.metrics()["prefix_hit_count"] == 2
    assert cache.metrics()["prefix_miss_count"] == 1
    assert cache.validate_invariants()


def test_shared_prefix_capacity_failure_is_atomic():
    cache = _cache(num_layers=1, max_blocks=2, prefix_capacity=2)
    prefix_k, prefix_v = _prefix_blocks(cache, 2)
    other_k, other_v = _prefix_blocks(cache, 1, start=100)
    cache.register_prefix("active", prefix_k, prefix_v)
    cache.add_request("owner")
    cache.attach_prefix("owner", "active")

    request_before = cache.request_state("owner")
    prefix_before = cache.prefix_state("active")
    version_before = cache.state_version
    with pytest.raises(RuntimeError, match="insufficient evictable prefix capacity"):
        cache.register_prefix("blocked", other_k, other_v)

    assert cache.request_state("owner") == request_before
    assert cache.prefix_state("active") == prefix_before
    assert cache.state_version == version_before
    assert cache.num_free_blocks == 0
    assert cache.metrics()["capacity_failure_count"] == 1
    assert cache.metrics()["prefix_capacity_failure_count"] == 1
    with pytest.raises(KeyError, match="unknown prefix_id"):
        cache.prefix_state("blocked")
    assert cache.validate_invariants()


def test_shared_prefix_transaction_abort_returns_only_private_boundary_block():
    cache = _cache(max_blocks=3, prefix_capacity=1)
    prefix_k, prefix_v = _prefix_blocks(cache, 1)
    prefix_ids = cache.register_prefix("prefix", prefix_k, prefix_v)["block_ids"]
    cache.add_request("request")
    cache.attach_prefix("request", "prefix")

    transaction = cache.begin_token(["request"])
    cache.write_token_layer(
        transaction,
        0,
        torch.ones((1, 1, 3), device=cache.device, dtype=cache.dtype),
        torch.ones((1, 1, 3), device=cache.device, dtype=cache.dtype),
    )
    cache.abort_token(transaction)

    assert cache.request_block_ids("request") == prefix_ids
    assert cache.request_state("request")["seq_len"] == cache.block_size
    assert cache.prefix_state("prefix")["active_refcount"] == 1
    assert cache.num_used_blocks == 1
    assert cache.validate_invariants()


def test_shared_prefix_rejects_invalid_or_mutating_attach_inputs():
    cache = _cache(num_layers=1, max_blocks=3, prefix_capacity=1)
    prefix_k, prefix_v = _prefix_blocks(cache, 1)
    bad = prefix_k[:, :, :, :-1, :]
    with pytest.raises(ValueError, match="does not match"):
        cache.register_prefix("bad", bad, bad)

    cache.register_prefix("valid", prefix_k, prefix_v)
    cache.add_request("nonempty")
    token = torch.zeros((1, 1, 3), device=cache.device, dtype=cache.dtype)
    cache.append(0, ["nonempty"], token, token)
    before = cache.request_state("nonempty")
    with pytest.raises(RuntimeError, match="empty active request"):
        cache.attach_prefix("nonempty", "valid")
    assert cache.request_state("nonempty") == before
    assert cache.prefix_state("valid")["active_refcount"] == 0
    assert cache.validate_invariants()
