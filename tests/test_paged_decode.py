import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashdec.cache import PagedKVCache
from flashdec.kernels.paged_decode import (
    _vllm_paged_decode_attention_into,
    paged_decode_attention,
    paged_decode_attention_into,
)
from flashdec.paged_reference import paged_decode_attention_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for Triton kernel tests",
)


DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)


def _assert_close(actual, expected):
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _make_paged_inputs(
    request_ids,
    target_seq_lens,
    num_q_heads,
    num_kv_heads,
    head_dim=64,
    block_size=16,
    kv_layout="token_major",
    max_blocks=None,
    dtype=torch.float16,
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
        dtype=dtype,
        device="cuda",
    )
    for request_id in request_ids:
        cache.add_request(request_id)

    max_seq_len = max(target_seq_lens) if target_seq_lens else 0
    token_k = torch.randn(
        (len(request_ids), max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
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

    q = torch.randn((len(request_ids), num_q_heads, head_dim), device="cuda", dtype=dtype)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    k_cache = cache.k_cache[0]
    v_cache = cache.v_cache[0]
    if kv_layout == "dim_major":
        k_cache = k_cache.permute(0, 1, 3, 2).contiguous()
        v_cache = v_cache.permute(0, 1, 3, 2).contiguous()
    elif kv_layout != "token_major":
        raise ValueError("kv_layout must be 'token_major' or 'dim_major'")
    return q, k_cache, v_cache, block_tables, seq_lens, cache


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("kv_layout", ["token_major", "dim_major"])
def test_paged_decode_attention_matches_reference_variable_lengths(
    kv_layout, block_size, head_dim, dtype
):
    q, k_cache, v_cache, block_tables, seq_lens, cache = _make_paged_inputs(
        request_ids=[101, 202, 303],
        target_seq_lens=[33, 1, 47],
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=head_dim,
        block_size=block_size,
        kv_layout=kv_layout,
        dtype=dtype,
    )

    assert cache.request_block_ids(101)[1] != cache.request_block_ids(101)[0] + 1

    actual = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size=block_size,
        kv_layout=kv_layout,
    )
    expected = paged_decode_attention_ref(
        q, k_cache, v_cache, block_tables, seq_lens, kv_layout=kv_layout
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("kv_layout", ["token_major", "dim_major"])
def test_paged_decode_attention_supports_gqa_mapping(kv_layout, block_size, dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[11, 22],
        target_seq_lens=[16, 31],
        num_q_heads=32,
        num_kv_heads=2,
        block_size=block_size,
        kv_layout=kv_layout,
        dtype=dtype,
    )

    actual = paged_decode_attention(
        q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size, kv_layout=kv_layout
    )
    expected = paged_decode_attention_ref(
        q, k_cache, v_cache, block_tables, seq_lens, kv_layout=kv_layout
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("kv_layout", ["token_major", "dim_major"])
def test_paged_decode_attention_supports_mqa_mapping(kv_layout, block_size, dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[31, 41],
        target_seq_lens=[32, 45],
        num_q_heads=16,
        num_kv_heads=1,
        block_size=block_size,
        kv_layout=kv_layout,
        dtype=dtype,
    )

    actual = paged_decode_attention(
        q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size, kv_layout=kv_layout
    )
    expected = paged_decode_attention_ref(
        q, k_cache, v_cache, block_tables, seq_lens, kv_layout=kv_layout
    )

    _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_decode_attention_zero_seq_len_outputs_zero(dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[1, 2],
        target_seq_lens=[0, 17],
        num_q_heads=2,
        num_kv_heads=2,
        dtype=dtype,
    )

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    _assert_close(actual, expected)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_decode_attention_supports_custom_scale(dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[7],
        target_seq_lens=[19],
        num_q_heads=2,
        num_kv_heads=1,
        dtype=dtype,
    )

    actual = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=0.25)
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens, sm_scale=0.25)

    _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_decode_attention_into_supports_strided_nhd_cache_view(dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[71, 72],
        target_seq_lens=[17, 31],
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        dtype=dtype,
    )
    # vLLM's NHD cache view is [block, token, head, dim]. Reinterpret it as
    # FlashDec token-major [block, head, token, dim] without materializing a
    # cache-sized contiguous copy.
    k_nhd = k_cache.permute(0, 2, 1, 3).contiguous()
    v_nhd = v_cache.permute(0, 2, 1, 3).contiguous()
    k_view = k_nhd.permute(0, 2, 1, 3)
    v_view = v_nhd.permute(0, 2, 1, 3)
    assert not k_view.is_contiguous()
    assert not v_view.is_contiguous()

    out = torch.empty_like(q)
    returned = paged_decode_attention_into(
        q,
        k_view,
        v_view,
        block_tables,
        seq_lens,
        out,
        block_size=16,
    )
    expected = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    )
    assert returned is out
    _assert_close(out, expected)


def test_paged_decode_attention_dynamic_loop_ignores_unused_invalid_blocks():
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[81],
        target_seq_lens=[7],
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=64,
        block_size=8,
        dtype=torch.float16,
    )
    extended_table = torch.full((1, 64), 2**30, device="cuda", dtype=torch.int32)
    extended_table[:, : block_tables.shape[1]] = block_tables

    actual = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        extended_table,
        seq_lens,
        block_size=8,
    )
    expected = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    )
    _assert_close(actual, expected)


def test_paged_decode_attention_into_validates_output_contract():
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([7], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="same shape"):
        paged_decode_attention_into(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            torch.empty((1, 64), device="cuda", dtype=torch.float16),
        )


@pytest.mark.parametrize("num_splits", [2, 8, 16])
@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_decode_attention_split_kv_matches_reference(num_splits, dtype):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[91, 92],
        target_seq_lens=[257, 193],
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        dtype=dtype,
    )
    workspace = (
        torch.empty((2, 16, 16, 128), device="cuda", dtype=torch.float32),
        torch.empty((2, 16, 16), device="cuda", dtype=torch.float32),
        torch.empty((2, 16, 16), device="cuda", dtype=torch.float32),
    )
    out = torch.empty_like(q)

    paged_decode_attention_into(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        out,
        block_size=16,
        split_kv_workspace=workspace,
        num_splits=num_splits,
    )
    expected = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    )

    _assert_close(out, expected)


@pytest.mark.parametrize("num_splits", [1, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("prior_lengths", [(0, 17), (256, 193)])
def test_paged_decode_attention_fuses_current_kv_append(
    prior_lengths, num_splits, dtype
):
    request_ids = [111, 222]
    q, k_cache, v_cache, _block_tables, _seq_lens, cache = _make_paged_inputs(
        request_ids=request_ids,
        target_seq_lens=list(prior_lengths),
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        dtype=dtype,
    )
    append_k = torch.randn((2, 2, 128), device="cuda", dtype=dtype)
    append_v = torch.randn_like(append_k)
    cache.append(0, request_ids, append_k, append_v)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    slots = []
    for request_id in request_ids:
        state = cache.request_state(request_id)
        token_offset = (state["seq_len"] - 1) % cache.block_size
        block_id = cache.request_block_ids(request_id)[-1]
        slots.append(block_id * cache.block_size + token_offset)

    expected = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    )
    expected_k_cache = k_cache.clone()
    expected_v_cache = v_cache.clone()
    for row, request_id in enumerate(request_ids):
        state = cache.request_state(request_id)
        token_offset = (state["seq_len"] - 1) % cache.block_size
        block_id = cache.request_block_ids(request_id)[-1]
        k_cache[block_id, :, token_offset].zero_()
        v_cache[block_id, :, token_offset].zero_()

    workspace = (
        torch.empty((2, 16, 16, 128), device="cuda", dtype=torch.float32),
        torch.empty((2, 16, 16), device="cuda", dtype=torch.float32),
        torch.empty((2, 16, 16), device="cuda", dtype=torch.float32),
    )
    out = torch.empty_like(q)
    paged_decode_attention_into(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        out,
        block_size=16,
        split_kv_workspace=workspace,
        num_splits=num_splits,
        append_k=append_k,
        append_v=append_v,
        slot_mapping=torch.tensor(slots, device="cuda", dtype=torch.int64),
    )

    _assert_close(out, expected)
    torch.testing.assert_close(k_cache, expected_k_cache)
    torch.testing.assert_close(v_cache, expected_v_cache)
    for row, request_id in enumerate(request_ids):
        state = cache.request_state(request_id)
        token_offset = (state["seq_len"] - 1) % cache.block_size
        block_id = cache.request_block_ids(request_id)[-1]
        torch.testing.assert_close(
            k_cache[block_id, :, token_offset], append_k[row]
        )
        torch.testing.assert_close(
            v_cache[block_id, :, token_offset], append_v[row]
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("num_splits", "cache_layout"),
    [(1, "NHD"), (8, "HND")],
)
def test_vllm_unchecked_launcher_handles_full_cache_and_graph_padding(
    num_splits, cache_layout, dtype
):
    request_ids = [411, 422]
    q_active, k_cache, v_cache, _, _, cache = _make_paged_inputs(
        request_ids=request_ids,
        target_seq_lens=[0, 256],
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        # Both requests cross a physical-page boundary on append.
        max_blocks=18,
        dtype=dtype,
    )
    append_k_active = torch.randn((2, 2, 128), device="cuda", dtype=dtype)
    append_v_active = torch.randn_like(append_k_active)
    cache.append(0, request_ids, append_k_active, append_v_active)
    block_tables_active = cache.block_tables(request_ids)
    seq_lens_active = cache.seq_lens_tensor(request_ids)

    slots = []
    for request_id in request_ids:
        state = cache.request_state(request_id)
        token_offset = (state["seq_len"] - 1) % cache.block_size
        block_id = cache.request_block_ids(request_id)[-1]
        slots.append(block_id * cache.block_size + token_offset)

    expected_out = paged_decode_attention_ref(
        q_active,
        k_cache,
        v_cache,
        block_tables_active,
        seq_lens_active,
    )
    expected_cache = torch.stack(
        (k_cache.permute(0, 2, 1, 3), v_cache.permute(0, 2, 1, 3)),
        dim=1,
    ).contiguous()
    if cache_layout == "NHD":
        full_cache = expected_cache.clone()
    else:
        hnd_storage = torch.empty(
            (
                expected_cache.shape[0],
                2,
                2,
                cache.block_size,
                128,
            ),
            device="cuda",
            dtype=dtype,
        )
        full_cache = hnd_storage.permute(0, 1, 3, 2, 4)
        full_cache.copy_(expected_cache)
        assert not full_cache.is_contiguous()

    for request_id in request_ids:
        state = cache.request_state(request_id)
        token_offset = (state["seq_len"] - 1) % cache.block_size
        block_id = cache.request_block_ids(request_id)[-1]
        full_cache[block_id, :, token_offset].zero_()

    # Four tensor rows model a static CUDA Graph buffer. The first two are
    # active, the third is padding inside the launch grid (slot=-1), and the
    # fourth lies beyond num_reqs and must remain untouched.
    q = torch.cat(
        (
            q_active,
            torch.randn((2, 16, 128), device="cuda", dtype=dtype),
        )
    )
    append_k = torch.cat(
        (
            append_k_active,
            torch.randn((2, 2, 128), device="cuda", dtype=dtype),
        )
    )
    append_v = torch.cat(
        (
            append_v_active,
            torch.randn((2, 2, 128), device="cuda", dtype=dtype),
        )
    )
    block_tables = torch.cat(
        (block_tables_active, torch.zeros_like(block_tables_active[:1]))
    )
    seq_lens_storage = torch.zeros((3, 2), device="cuda", dtype=torch.int32)
    seq_lens_storage[:2, 1] = seq_lens_active
    seq_lens_storage[2, 1] = 17
    seq_lens = seq_lens_storage[:, 1]
    assert not seq_lens.is_contiguous()
    slot_mapping = torch.tensor(
        [*slots, -1], device="cuda", dtype=torch.int64
    )
    workspace = (
        torch.empty((4, 16, 16, 128), device="cuda", dtype=torch.float32),
        torch.empty((4, 16, 16), device="cuda", dtype=torch.float32),
        torch.empty((4, 16, 16), device="cuda", dtype=torch.float32),
    )
    out = torch.full_like(q, 37)

    returned = _vllm_paged_decode_attention_into(
        q,
        append_k,
        append_v,
        full_cache,
        block_tables,
        seq_lens,
        out,
        slot_mapping,
        *workspace,
        num_reqs=3,
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=128,
        block_size=16,
        sm_scale=1.0 / math.sqrt(128),
        num_splits=num_splits,
    )

    assert returned is out
    _assert_close(out[:2], expected_out)
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]))
    torch.testing.assert_close(out[3], torch.full_like(out[3], 37))
    torch.testing.assert_close(full_cache, expected_cache)


def test_paged_decode_attention_rejects_partial_fused_append_arguments():
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[333],
        target_seq_lens=[16],
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=128,
        block_size=16,
        dtype=torch.float16,
    )
    with pytest.raises(ValueError, match="must be provided together"):
        paged_decode_attention_into(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            torch.empty_like(q),
            append_k=torch.empty((1, 1, 128), device="cuda", dtype=torch.float16),
        )


def test_paged_decode_attention_rejects_unsupported_head_dim():
    q = torch.randn((1, 1, 32), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 16, 32), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="head_dim"):
        paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)


def test_paged_decode_attention_rejects_unsupported_block_size():
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 4, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="block_size 8, 16, or 32"):
        paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, block_size=4)


def test_paged_decode_attention_rejects_unknown_kv_layout():
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="kv_layout"):
        paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, kv_layout="unknown")


@pytest.mark.parametrize("num_warps", [0, 3, 16, True])
def test_paged_decode_attention_rejects_invalid_num_warps(num_warps):
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="num_warps"):
        paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            num_warps=num_warps,
        )


@pytest.mark.parametrize("num_stages", [0, 5, -1, True, 1.5])
def test_paged_decode_attention_rejects_invalid_num_stages(num_stages):
    q = torch.randn((1, 1, 64), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="num_stages"):
        paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            num_stages=num_stages,
        )


@pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
def test_paged_decode_attention_supports_explicit_num_stages(num_stages):
    q, k_cache, v_cache, block_tables, seq_lens, _ = _make_paged_inputs(
        request_ids=[1, 2],
        target_seq_lens=[33, 47],
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        block_size=32,
        dtype=torch.float16,
    )

    actual = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size=32,
        num_stages=num_stages,
    )
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)

    _assert_close(actual, expected)
