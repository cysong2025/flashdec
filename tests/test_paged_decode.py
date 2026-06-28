import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashdec.cache import PagedKVCache
from flashdec.kernels.paged_decode import paged_decode_attention
from flashdec.paged_reference import paged_decode_attention_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for Triton kernel tests",
)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _make_paged_inputs(
    request_ids,
    target_seq_lens,
    num_q_heads,
    num_kv_heads,
    head_dim=64,
    block_size=16,
    max_blocks=None,
):
    torch.manual_seed(61)
    torch.cuda.manual_seed_all(61)
    if max_blocks is None:
        max_blocks = sum((seq_len + block_size - 1) // block_size for seq_len in target_seq_lens) + 1
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=torch.float16,
        device="cuda",
    )
    for request_id in request_ids:
        cache.add_request(request_id)

    max_seq_len = max(target_seq_lens) if target_seq_lens else 0
    token_k = torch.randn(
        (len(request_ids), max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    token_v = torch.randn_like(token_k)

    for step in range(max_seq_len):
        active_rows = [row for row, seq_len in enumerate(target_seq_lens) if step < seq_len]
        if not active_rows:
            continue
        active_ids = [request_ids[row] for row in active_rows]
        cache.append(
            layer_idx=0,
            request_ids=active_ids,
            k=token_k[active_rows, step],
            v=token_v[active_rows, step],
        )

    q = torch.randn((len(request_ids), num_q_heads, head_dim), device="cuda", dtype=torch.float16)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    return q, cache.k_cache[0], cache.v_cache[0], block_tables, seq_lens, cache


def test_paged_decode_attention_matches_reference_variable_lengths():
    q, k_cache, v_cache, block_tables, seq_lens, cache = _make_paged_inputs(
        request_ids=[101, 202, 303],
        target_seq_lens=[33, 1, 47],
        num_q_heads=4,
        num_kv_heads=4,
    )

    assert cache.request_block_ids(101)[1] != cache.request_block_ids(101)[0] + 1

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    _assert_close(actual, expected)


def test_paged_decode_attention_supports_gqa_mapping():
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[11, 22],
        target_seq_lens=[16, 31],
        num_q_heads=8,
        num_kv_heads=2,
    )

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    _assert_close(actual, expected)


def test_paged_decode_attention_zero_seq_len_outputs_zero():
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[1, 2],
        target_seq_lens=[0, 17],
        num_q_heads=2,
        num_kv_heads=2,
    )

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    _assert_close(actual, expected)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))


def test_paged_decode_attention_supports_custom_scale():
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[7],
        target_seq_lens=[19],
        num_q_heads=2,
        num_kv_heads=1,
    )

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=0.25)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=0.25)

    _assert_close(actual, expected)


def test_paged_decode_attention_rejects_unsupported_head_dim():
    q = torch.randn((1, 1, 128), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 16, 128), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="head_dim"):
        paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)


def test_paged_decode_attention_rejects_unsupported_block_size():
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([8], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="block_size"):
        paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=8)
