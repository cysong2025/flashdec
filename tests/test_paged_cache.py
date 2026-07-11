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
