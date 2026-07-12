import pytest

torch = pytest.importorskip("torch")

from flashdec.cache import PagedKVCache
from flashdec.paged_reference import paged_decode_attention_ref
from flashdec.reference import dense_decode_attention_ref


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32]
if torch.cuda.is_available():
    DTYPES.append(torch.float16)


def _make_cache(num_kv_heads=1, head_dim=4, block_size=2, max_blocks=8, dtype=torch.float32):
    return PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device=DEVICE,
    )


def test_paged_kv_cache_multi_layer_requires_transaction_writes():
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        block_size=2,
        max_blocks=2,
        dtype=torch.float32,
        device=DEVICE,
    )
    cache.add_request(1)
    token = torch.ones((1, 1, 2), device=DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="transaction API"):
        cache.append(0, [1], token, token)


def test_paged_kv_cache_append_tracks_non_contiguous_blocks():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=2, max_blocks=4)

    def append_one(request_id, values):
        k = torch.tensor(values, device=DEVICE, dtype=torch.float32).reshape(1, 1, 2)
        v = (k + 100).clone()
        cache.append(layer_idx=0, request_ids=[request_id], k=k, v=v)

    append_one(10, [1.0, 2.0])
    append_one(20, [3.0, 4.0])
    append_one(10, [5.0, 6.0])
    append_one(10, [7.0, 8.0])

    assert cache.request_block_ids(10) == (0, 2)
    assert cache.request_block_ids(20) == (1,)
    assert cache.num_used_blocks == 3
    assert cache.num_free_blocks == 1

    block_tables = cache.block_tables([10, 20])
    expected_tables = torch.tensor([[0, 2], [1, -1]], device=DEVICE, dtype=torch.int32)
    torch.testing.assert_close(block_tables, expected_tables)

    seq_lens = cache.seq_lens_tensor([10, 20])
    expected_lens = torch.tensor([3, 1], device=DEVICE, dtype=torch.int32)
    torch.testing.assert_close(seq_lens, expected_lens)

    torch.testing.assert_close(cache.k_cache[0, 0, 0, 0], torch.tensor([1.0, 2.0], device=DEVICE))
    torch.testing.assert_close(cache.k_cache[0, 0, 0, 1], torch.tensor([5.0, 6.0], device=DEVICE))
    torch.testing.assert_close(cache.k_cache[0, 2, 0, 0], torch.tensor([7.0, 8.0], device=DEVICE))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_layout", ["token_major", "dim_major"])
def test_paged_decode_attention_matches_dense_reference(kv_layout, dtype):
    torch.manual_seed(41)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)

    request_ids = [101, 202, 303]
    target_seq_lens = [5, 1, 4]
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 2
    cache = _make_cache(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=8,
        dtype=dtype,
    )

    max_seq_len = max(target_seq_lens)
    token_k = torch.randn(
        (len(request_ids), max_seq_len, num_kv_heads, head_dim),
        device=DEVICE,
        dtype=dtype,
    )
    token_v = torch.randn_like(token_k)

    for step in range(max_seq_len):
        active_rows = [row for row, seq_len in enumerate(target_seq_lens) if step < seq_len]
        if not active_rows:
            continue
        active_ids = [request_ids[row] for row in active_rows]
        k = token_k[active_rows, step]
        v = token_v[active_rows, step]
        cache.append(layer_idx=0, request_ids=active_ids, k=k, v=v)

    q = torch.randn((len(request_ids), num_q_heads, head_dim), device=DEVICE, dtype=dtype)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    dense_k, dense_v, dense_seq_lens = cache.to_dense(layer_idx=0, request_ids=request_ids)

    k_cache = cache.k_cache[0]
    v_cache = cache.v_cache[0]
    if kv_layout == "dim_major":
        k_cache = k_cache.permute(0, 1, 3, 2).contiguous()
        v_cache = v_cache.permute(0, 1, 3, 2).contiguous()

    paged = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        kv_layout=kv_layout,
    )
    dense = dense_decode_attention_ref(q, dense_k, dense_v, dense_seq_lens)

    if dtype is torch.float16:
        torch.testing.assert_close(paged, dense, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-5)


def test_paged_decode_attention_zero_seq_len_outputs_zero():
    q = torch.randn((2, 2, 8), device=DEVICE, dtype=torch.float32)
    k_cache = torch.randn((2, 1, 4, 8), device=DEVICE, dtype=torch.float32)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0], [-1]], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([3, 0], device=DEVICE, dtype=torch.int32)

    actual = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))


def test_paged_decode_attention_rejects_missing_physical_block():
    q = torch.randn((1, 1, 8), device=DEVICE, dtype=torch.float32)
    k_cache = torch.randn((1, 1, 4, 8), device=DEVICE, dtype=torch.float32)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[-1]], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor([1], device=DEVICE, dtype=torch.int32)

    with pytest.raises(ValueError, match="invalid physical block"):
        paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)


def test_paged_kv_cache_rejects_capacity_overflow():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=1, max_blocks=1)
    k = torch.zeros((1, 1, 2), device=DEVICE, dtype=torch.float32)
    v = torch.zeros_like(k)

    cache.append(layer_idx=0, request_ids=[1], k=k, v=v)

    with pytest.raises(RuntimeError, match="out of physical blocks"):
        cache.append(layer_idx=0, request_ids=[1], k=k, v=v)


def test_paged_kv_cache_finish_releases_and_reuses_blocks_without_stale_tokens():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=2, max_blocks=2)

    for value in (1.0, 2.0, 3.0):
        k = torch.full((1, 1, 2), value, device=DEVICE, dtype=torch.float32)
        cache.append(0, [10], k, k + 100)

    assert cache.request_block_ids(10) == (0, 1)
    assert cache.finish_request(10) == (0, 1)
    assert cache.request_state(10) == {
        "request_id": 10,
        "status": "finished",
        "seq_len": 3,
        "block_ids": (),
    }
    assert cache.num_used_blocks == 0
    assert cache.num_free_blocks == 2

    new_k = torch.tensor([[[9.0, 10.0]]], device=DEVICE, dtype=torch.float32)
    new_v = new_k + 100
    cache.append(0, [20], new_k, new_v)

    assert cache.request_block_ids(20) == (0,)
    dense_k, dense_v, seq_lens = cache.to_dense(0, [20])
    torch.testing.assert_close(dense_k[0, 0], new_k[0])
    torch.testing.assert_close(dense_v[0, 0], new_v[0])
    torch.testing.assert_close(seq_lens, torch.tensor([1], device=DEVICE, dtype=torch.int32))

    metrics = cache.metrics()
    assert metrics["allocation_count"] == 3
    assert metrics["fresh_allocation_count"] == 2
    assert metrics["free_count"] == 2
    assert metrics["reuse_count"] == 1
    assert metrics["finished_requests"] == 1
    assert cache.validate_invariants()


def test_paged_kv_cache_cancel_releases_blocks_and_rejects_terminal_operations():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=1, max_blocks=1)
    k = torch.ones((1, 1, 2), device=DEVICE, dtype=torch.float32)
    cache.append(0, ["cancel-me"], k, k)

    assert cache.cancel_request("cancel-me") == (0,)
    assert cache.request_state("cancel-me")["status"] == "cancelled"
    assert cache.metrics()["cancelled_requests"] == 1

    with pytest.raises(RuntimeError, match="cancelled"):
        cache.append(0, ["cancel-me"], k, k)
    with pytest.raises(RuntimeError, match="cancelled"):
        cache.block_tables(["cancel-me"])
    with pytest.raises(RuntimeError, match="not active"):
        cache.finish_request("cancel-me")
    with pytest.raises(RuntimeError, match="cannot be reactivated"):
        cache.add_request("cancel-me")

    assert cache.validate_invariants()


def test_paged_kv_cache_capacity_failure_has_no_partial_request_mutation():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=2, max_blocks=2)
    first = torch.ones((1, 1, 2), device=DEVICE, dtype=torch.float32)
    cache.append(0, [1], first, first)
    before = cache.request_state(1)

    batch = torch.zeros((3, 1, 2), device=DEVICE, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="out of physical blocks"):
        cache.append(0, [1, 2, 3], batch, batch)

    assert cache.request_state(1) == before
    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(2)
    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(3)
    assert cache.num_used_blocks == 1
    assert cache.num_free_blocks == 1
    assert cache.metrics()["capacity_failure_count"] == 1
    assert cache.validate_invariants()


def test_paged_kv_cache_metrics_report_utilization_and_fragmentation():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=4, max_blocks=4)
    batch = torch.zeros((2, 1, 2), device=DEVICE, dtype=torch.float32)
    cache.append(0, [1, 2], batch, batch)
    cache.append(0, [1], batch[:1], batch[:1])

    assert cache.metrics() == {
        "max_blocks": 4,
        "used_blocks": 2,
        "free_blocks": 2,
        "block_utilization": 0.5,
        "active_tokens": 3,
        "reserved_tokens": 8,
        "internal_fragmentation_tokens": 5,
        "internal_fragmentation_ratio": 0.625,
        "allocation_count": 2,
        "fresh_allocation_count": 2,
        "free_count": 0,
        "reuse_count": 0,
        "capacity_failure_count": 0,
        "active_requests": 2,
        "finished_requests": 0,
        "cancelled_requests": 0,
        "transaction_begin_count": 0,
        "transaction_commit_count": 0,
        "transaction_abort_count": 0,
        "open_transaction_count": 0,
        "pending_request_count": 0,
        "reserved_transaction_blocks": 0,
        "transaction_layer_write_count": 0,
        "transaction_rollback_block_count": 0,
        "transaction_failure_count": 0,
        "bytes_per_block": 64,
        "allocated_kv_bytes": 128,
        "reserved_transaction_bytes": 0,
    }
    assert cache.validate_invariants()


def test_paged_kv_cache_state_version_tracks_only_successful_state_mutations():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=1, max_blocks=1)
    token = torch.ones((1, 1, 2), device=DEVICE, dtype=torch.float32)
    assert cache.state_version == 0

    cache.add_request("owner")
    assert cache.state_version == 1
    cache.append(0, ["owner"], token, token)
    assert cache.state_version == 2

    with pytest.raises(RuntimeError, match="out of physical blocks"):
        cache.append(0, ["other"], token, token)
    assert cache.state_version == 2

    cache.finish_request("owner")
    assert cache.state_version == 3


def test_paged_kv_cache_default_metadata_only_contains_active_requests():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=2, max_blocks=2)
    batch = torch.zeros((2, 1, 2), device=DEVICE, dtype=torch.float32)
    cache.append(0, [10, 20], batch, batch)
    cache.finish_request(10)

    tables = cache.block_tables()
    seq_lens = cache.seq_lens_tensor()

    torch.testing.assert_close(
        tables,
        torch.tensor([[1]], device=DEVICE, dtype=torch.int32),
    )
    torch.testing.assert_close(
        seq_lens,
        torch.tensor([1], device=DEVICE, dtype=torch.int32),
    )
    assert cache.request_state(10)["status"] == "finished"
    assert cache.request_state(20)["status"] == "active"
    assert cache.validate_invariants()


def test_paged_kv_cache_request_churn_does_not_leak_blocks():
    cache = _make_cache(num_kv_heads=1, head_dim=2, block_size=1, max_blocks=1)
    token = torch.ones((1, 1, 2), device=DEVICE, dtype=torch.float32)

    for request_id in range(20):
        cache.append(0, [request_id], token, token)
        if request_id % 2 == 0:
            cache.finish_request(request_id)
        else:
            cache.cancel_request(request_id)
        assert cache.num_used_blocks == 0
        assert cache.num_free_blocks == 1
        assert cache.validate_invariants()

    metrics = cache.metrics()
    assert metrics["allocation_count"] == 20
    assert metrics["fresh_allocation_count"] == 1
    assert metrics["reuse_count"] == 19
    assert metrics["free_count"] == 20
    assert metrics["active_requests"] == 0
    assert metrics["finished_requests"] == 10
    assert metrics["cancelled_requests"] == 10
