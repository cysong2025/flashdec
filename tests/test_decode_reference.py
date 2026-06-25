import pytest

torch = pytest.importorskip("torch")

from flashdec.reference import dense_decode_attention_ref


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32]
if torch.cuda.is_available():
    DTYPES.append(torch.float16)


def _expanded_expected(q, k_cache, v_cache, seq_lens, sm_scale=None):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group_size = num_q_heads // num_kv_heads
    kv_indices = torch.arange(num_q_heads, device=q.device) // group_size
    out = torch.empty_like(q)

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len == 0:
            out[seq_idx].zero_()
            continue
        k = k_cache[seq_idx, :seq_len].index_select(1, kv_indices).to(torch.float32)
        v = v_cache[seq_idx, :seq_len].index_select(1, kv_indices).to(torch.float32)
        q_seq = q[seq_idx].to(torch.float32)
        scores = torch.einsum("hd,lhd->hl", q_seq, k) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[seq_idx] = torch.einsum("hl,lhd->hd", probs, v).to(q.dtype)

    return out


def test_dense_decode_attention_ignores_padding_tokens():
    q = torch.tensor([[[1.0, 0.0]]], device=DEVICE)
    k_cache = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1000.0, 1000.0]]]],
        device=DEVICE,
    )
    v_cache = torch.tensor(
        [[[[10.0, 0.0]], [[0.0, 20.0]], [[999.0, 999.0]]]],
        device=DEVICE,
    )
    seq_lens = torch.tensor([2], device=DEVICE)

    actual = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=1.0)
    weights = torch.softmax(torch.tensor([1.0, 0.0], device=DEVICE), dim=0)
    expected = (weights[0] * v_cache[0, 0, 0] + weights[1] * v_cache[0, 1, 0]).reshape(1, 1, 2)

    torch.testing.assert_close(actual, expected)


def test_dense_decode_attention_supports_mqa_mapping():
    q = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ]
        ],
        device=DEVICE,
    )
    k_cache = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]],
        device=DEVICE,
    )
    v_cache = torch.tensor(
        [[[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]]]],
        device=DEVICE,
    )
    seq_lens = torch.tensor([3], device=DEVICE)

    actual = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=1.0)
    expected = _expanded_expected(q, k_cache, v_cache, seq_lens, sm_scale=1.0)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 16, 1),
        (2, 4, 2, 16, 17),
        (4, 8, 4, 64, 33),
        (2, 8, 1, 32, 9),
        (2, 8, 2, 128, 7),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_dense_decode_attention_random_shapes(shape, dtype):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    q = torch.randn((num_seqs, num_q_heads, head_dim), device=DEVICE, dtype=dtype)
    k_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device=DEVICE,
        dtype=dtype,
    )
    v_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device=DEVICE,
        dtype=dtype,
    )
    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), device=DEVICE)

    actual = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens)
    expected = _expanded_expected(q, k_cache, v_cache, seq_lens)

    if dtype is torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_dense_decode_attention_rejects_invalid_gqa_ratio():
    q = torch.randn((1, 3, 16), device=DEVICE)
    k_cache = torch.randn((1, 4, 2, 16), device=DEVICE)
    v_cache = torch.randn((1, 4, 2, 16), device=DEVICE)
    seq_lens = torch.tensor([4], device=DEVICE)

    with pytest.raises(ValueError, match="num_q_heads"):
        dense_decode_attention_ref(q, k_cache, v_cache, seq_lens)
