import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashdec.kernels.dense_decode import dense_decode_attention
from flashdec.reference import dense_decode_attention_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for Triton kernel tests",
)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_dense_decode_attention_ignores_padding_tokens():
    q = torch.zeros((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.zeros((1, 4, 1, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.zeros((1, 4, 1, 64), device="cuda", dtype=torch.float16)
    q[0, 0, 0] = 1.0
    k_cache[0, 0, 0, 0] = 1.0
    k_cache[0, 1, 0, 1] = 1.0
    k_cache[0, 2, 0, :] = 1000.0
    v_cache[0, 0, 0, 0] = 10.0
    v_cache[0, 1, 0, 1] = 20.0
    v_cache[0, 2, 0, :] = 999.0
    seq_lens = torch.tensor([2], device="cuda")

    actual = dense_decode_attention(q, k_cache, v_cache, seq_lens, sm_scale=1.0, block_seq=16)
    expected = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=1.0)

    _assert_close(actual, expected)


def test_dense_decode_attention_zero_seq_len_outputs_zero():
    q = torch.randn((2, 4, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((2, 17, 2, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn((2, 17, 2, 64), device="cuda", dtype=torch.float16)
    seq_lens = torch.tensor([0, 17], device="cuda")

    actual = dense_decode_attention(q, k_cache, v_cache, seq_lens, block_seq=16)
    expected = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens)

    _assert_close(actual, expected)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 8, 64, 128),
        (2, 4, 2, 64, 17),
        (4, 8, 1, 64, 33),
        (2, 8, 4, 128, 65),
        (3, 16, 4, 64, 80),
    ],
)
@pytest.mark.parametrize("block_seq", [16, 64])
def test_dense_decode_attention_matches_reference(shape, block_seq):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    q = torch.randn((num_seqs, num_q_heads, head_dim), device="cuda", dtype=torch.float16)
    k_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    seq_lens = torch.randint(0, max_seq_len + 1, (num_seqs,), device="cuda")

    actual = dense_decode_attention(q, k_cache, v_cache, seq_lens, block_seq=block_seq)
    expected = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens)

    _assert_close(actual, expected)


def test_dense_decode_attention_supports_custom_scale():
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    q = torch.randn((1, 4, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 31, 2, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn((1, 31, 2, 64), device="cuda", dtype=torch.float16)
    seq_lens = torch.tensor([31], device="cuda")

    actual = dense_decode_attention(q, k_cache, v_cache, seq_lens, sm_scale=0.25, block_seq=16)
    expected = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=0.25)

    _assert_close(actual, expected)


def test_dense_decode_attention_rejects_unsupported_head_dim():
    q = torch.randn((1, 1, 32), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 8, 1, 32), device="cuda", dtype=torch.float16)
    v_cache = torch.randn((1, 8, 1, 32), device="cuda", dtype=torch.float16)
    seq_lens = torch.tensor([8], device="cuda")

    with pytest.raises(ValueError, match="head_dim"):
        dense_decode_attention(q, k_cache, v_cache, seq_lens)
