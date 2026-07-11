import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import flashdec


def test_decode_is_paged_decode_public_api():
    assert flashdec.decode is flashdec.paged_decode_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required for Triton kernel tests")
def test_decode_public_api_matches_reference():
    q = torch.randn((1, 2, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([7], device="cuda", dtype=torch.int32)

    actual = flashdec.decode(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size=8,
    )
    expected = flashdec.paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("kv_layout", "block_size"),
    [
        ("token_major", 16),
        ("token_major", 32),
        ("dim_major", 16),
        ("dim_major", 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required for Triton kernel tests")
def test_decode_infers_block_size_from_cache(kv_layout, block_size):
    q = torch.randn((1, 2, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, block_size, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    if kv_layout == "dim_major":
        k_cache = k_cache.permute(0, 1, 3, 2).contiguous()
        v_cache = v_cache.permute(0, 1, 3, 2).contiguous()
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([block_size - 1], device="cuda", dtype=torch.int32)

    actual = flashdec.decode(q, k_cache, v_cache, block_tables, seq_lens, kv_layout=kv_layout)
    expected = flashdec.paged_decode_attention_ref(
        q, k_cache, v_cache, block_tables, seq_lens, kv_layout=kv_layout
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
