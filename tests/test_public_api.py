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
