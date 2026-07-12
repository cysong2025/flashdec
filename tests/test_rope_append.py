import math
import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.paged_reference import paged_decode_attention_ref
from flashdec.reference import dense_decode_attention_ref
from flashdec.rope import (
    RopeAppendResult,
    apply_rope,
    rope_paged_kv_append,
    rope_paged_kv_append_ref,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32]
if torch.cuda.is_available():
    DTYPES.append(torch.float16)
    if torch.cuda.is_bf16_supported():
        DTYPES.append(torch.bfloat16)

HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required for native append tests"
CUDA_DTYPES = [torch.float16, torch.float32]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    CUDA_DTYPES.append(torch.bfloat16)


def _make_cache(
    block_size=2,
    max_blocks=8,
    dtype=torch.float32,
    num_kv_heads=1,
    head_dim=4,
    device=DEVICE,
):
    return PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device=device,
    )


def test_rope_is_public_api():
    assert flashdec.RopeAppendResult is RopeAppendResult
    assert flashdec.apply_rope is apply_rope
    assert flashdec.rope_paged_kv_append is rope_paged_kv_append
    assert flashdec.rope_paged_kv_append_ref is rope_paged_kv_append_ref


def test_apply_rope_matches_split_half_hand_calculation():
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0]]],
        device=DEVICE,
        dtype=torch.float32,
    )
    positions = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)

    actual = apply_rope(x, positions)
    expected_position_one = torch.tensor(
        [
            1.0 * math.cos(1.0) - 3.0 * math.sin(1.0),
            2.0 * math.cos(0.01) - 4.0 * math.sin(0.01),
            3.0 * math.cos(1.0) + 1.0 * math.sin(1.0),
            4.0 * math.cos(0.01) + 2.0 * math.sin(0.01),
        ],
        device=DEVICE,
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual[0], x[0])
    torch.testing.assert_close(actual[1, 0], expected_position_one, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_rope_preserves_tail_dtype_and_rotary_norm(dtype):
    torch.manual_seed(211)
    x = torch.randn((3, 2, 6), device=DEVICE, dtype=dtype)
    positions = torch.tensor([1, 7, 31], device=DEVICE, dtype=torch.int32)

    actual = apply_rope(x, positions, rotary_dim=4)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual[..., 4:], x[..., 4:], rtol=0.0, atol=0.0)
    tolerance = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(
        torch.linalg.vector_norm(actual[..., :4].float(), dim=-1),
        torch.linalg.vector_norm(x[..., :4].float(), dim=-1),
        rtol=tolerance,
        atol=tolerance,
    )


def test_rope_paged_kv_append_tracks_positions_and_block_boundary():
    cache = _make_cache(block_size=2, max_blocks=4)

    first_q = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]],
        ],
        device=DEVICE,
        dtype=torch.float32,
    )
    first_k = first_q[:, :1].clone()
    first_v = first_k + 100
    first = rope_paged_kv_append_ref(cache, 0, [10, 20], first_q, first_k, first_v)

    torch.testing.assert_close(first.positions, torch.tensor([0, 0], device=DEVICE))
    torch.testing.assert_close(first.q, first_q)
    torch.testing.assert_close(first.block_tables, torch.tensor([[0], [1]], device=DEVICE, dtype=torch.int32))
    torch.testing.assert_close(first.seq_lens, torch.tensor([1, 1], device=DEVICE, dtype=torch.int32))

    second_q = first_q[:1] + 10
    second_k = first_k[:1] + 10
    second_v = first_v[:1] + 10
    second = rope_paged_kv_append_ref(cache, 0, [10], second_q, second_k, second_v)
    expected_second_k = apply_rope(
        second_k,
        torch.tensor([1], device=DEVICE, dtype=torch.int64),
    )

    torch.testing.assert_close(second.positions, torch.tensor([1], device=DEVICE))
    torch.testing.assert_close(cache.k_cache[0, 0, :, 1, :], expected_second_k[0])
    torch.testing.assert_close(cache.v_cache[0, 0, :, 1, :], second_v[0])
    torch.testing.assert_close(second.seq_lens, torch.tensor([2], device=DEVICE, dtype=torch.int32))

    third = rope_paged_kv_append_ref(cache, 0, [10], second_q, second_k, second_v)
    torch.testing.assert_close(third.positions, torch.tensor([2], device=DEVICE))
    torch.testing.assert_close(
        third.block_tables,
        torch.tensor([[0, 2]], device=DEVICE, dtype=torch.int32),
    )
    assert cache.validate_invariants()


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_paged_kv_append_supports_runtime_dtypes(dtype):
    cache = _make_cache(block_size=2, max_blocks=2, dtype=dtype)
    torch.manual_seed(223)
    q = torch.randn((1, 2, 4), device=DEVICE, dtype=dtype)
    k = torch.randn((1, 1, 4), device=DEVICE, dtype=dtype)
    v = torch.randn((1, 1, 4), device=DEVICE, dtype=dtype)

    first = rope_paged_kv_append_ref(cache, 0, [1], q, k, v)
    second = rope_paged_kv_append_ref(cache, 0, [1], q, k, v)
    expected_k = apply_rope(
        k,
        torch.tensor([1], device=DEVICE, dtype=torch.int64),
    )

    assert first.q.dtype == dtype
    assert second.q.dtype == dtype
    torch.testing.assert_close(cache.k_cache[0, 0, :, 1, :], expected_k[0])
    torch.testing.assert_close(cache.v_cache[0, 0, :, 1, :], v[0])


def test_rope_paged_kv_append_default_backend_matches_reference():
    torch.manual_seed(229)
    reference_cache = _make_cache(block_size=2, max_blocks=2)
    default_cache = _make_cache(block_size=2, max_blocks=2)
    q = torch.randn((1, 2, 4), device=DEVICE, dtype=torch.float32)
    k = torch.randn((1, 1, 4), device=DEVICE, dtype=torch.float32)
    v = torch.randn_like(k)

    expected = rope_paged_kv_append_ref(reference_cache, 0, [10], q, k, v)
    actual = rope_paged_kv_append(default_cache, 0, [10], q, k, v)

    torch.testing.assert_close(actual.q, expected.q)
    torch.testing.assert_close(actual.positions, expected.positions)
    torch.testing.assert_close(actual.block_tables, expected.block_tables)
    torch.testing.assert_close(actual.seq_lens, expected.seq_lens)
    torch.testing.assert_close(default_cache.k_cache, reference_cache.k_cache)
    torch.testing.assert_close(default_cache.v_cache, reference_cache.v_cache)
    assert default_cache.metrics() == reference_cache.metrics()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_rope_paged_kv_append_cuda_matches_reference_for_gqa(dtype):
    torch.manual_seed(233)
    reference_cache = _make_cache(
        block_size=2,
        max_blocks=6,
        dtype=dtype,
        num_kv_heads=2,
        head_dim=8,
        device="cuda",
    )
    native_cache = _make_cache(
        block_size=2,
        max_blocks=6,
        dtype=dtype,
        num_kv_heads=2,
        head_dim=8,
        device="cuda",
    )

    for request_ids in ([10, 20], [10], [20], [10]):
        batch_size = len(request_ids)
        q = torch.randn((batch_size, 4, 8), device="cuda", dtype=dtype)
        k = torch.randn((batch_size, 2, 8), device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        expected = rope_paged_kv_append_ref(reference_cache, 0, request_ids, q, k, v)
        actual = rope_paged_kv_append(
            native_cache,
            0,
            request_ids,
            q,
            k,
            v,
            append_backend="cuda",
        )

        tolerance = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 0.0
        torch.testing.assert_close(actual.q, expected.q, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual.positions, expected.positions)
        torch.testing.assert_close(actual.block_tables, expected.block_tables)
        torch.testing.assert_close(actual.seq_lens, expected.seq_lens)

    torch.cuda.synchronize()
    tolerance = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 0.0
    torch.testing.assert_close(native_cache.k_cache, reference_cache.k_cache, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(native_cache.v_cache, reference_cache.v_cache, rtol=tolerance, atol=tolerance)
    assert native_cache.request_state(10) == reference_cache.request_state(10)
    assert native_cache.request_state(20) == reference_cache.request_state(20)
    assert native_cache.metrics() == reference_cache.metrics()
    assert native_cache.validate_invariants()


def test_rope_paged_kv_append_rejects_unknown_append_backend_without_mutation():
    cache = _make_cache()
    q = torch.ones((1, 2, 4), device=DEVICE, dtype=torch.float32)
    k = torch.ones((1, 1, 4), device=DEVICE, dtype=torch.float32)

    with pytest.raises(ValueError, match="append_backend"):
        rope_paged_kv_append(cache, 0, [10], q, k, k, append_backend="triton")

    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(10)
    assert cache.num_used_blocks == 0


def test_rope_paged_kv_append_cuda_backend_rejects_cpu_cache_without_mutation():
    cache = _make_cache(device="cpu")
    q = torch.ones((1, 2, 4), device="cpu", dtype=torch.float32)
    k = torch.ones((1, 1, 4), device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="CUDA-resident"):
        rope_paged_kv_append(cache, 0, [10], q, k, k, append_backend="cuda")

    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(10)
    assert cache.num_used_blocks == 0


def test_rope_paged_kv_append_capacity_failure_keeps_request_state_atomic():
    cache = _make_cache(block_size=1, max_blocks=1)
    q = torch.ones((1, 2, 4), device=DEVICE, dtype=torch.float32)
    k = torch.ones((1, 1, 4), device=DEVICE, dtype=torch.float32)
    v = k + 100
    rope_paged_kv_append_ref(cache, 0, [1], q, k, v)
    before = cache.request_state(1)

    with pytest.raises(RuntimeError, match="out of physical blocks"):
        rope_paged_kv_append_ref(cache, 0, [2], q, k, v)

    assert cache.request_state(1) == before
    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(2)
    assert cache.validate_invariants()


def test_rope_append_result_feeds_paged_decode_reference():
    cache = _make_cache(block_size=2, max_blocks=4)
    torch.manual_seed(227)

    result = None
    for step, request_ids in enumerate(([10, 20], [10, 20], [10])):
        batch = len(request_ids)
        q = torch.randn((batch, 2, 4), device=DEVICE, dtype=torch.float32)
        k = torch.randn((batch, 1, 4), device=DEVICE, dtype=torch.float32)
        v = torch.randn((batch, 1, 4), device=DEVICE, dtype=torch.float32)
        result = rope_paged_kv_append_ref(cache, 0, request_ids, q, k, v)
        assert result.positions.tolist() == [step] * batch

    assert result is not None
    dense_k, dense_v, dense_seq_lens = cache.to_dense(0, [10])
    paged = paged_decode_attention_ref(
        result.q,
        cache.k_cache[0],
        cache.v_cache[0],
        result.block_tables,
        result.seq_lens,
    )
    dense = dense_decode_attention_ref(result.q, dense_k, dense_v, dense_seq_lens)
    torch.testing.assert_close(paged, dense, rtol=1e-5, atol=1e-5)


def test_rope_paged_kv_append_rejects_terminal_request():
    cache = _make_cache(block_size=1, max_blocks=1)
    q = torch.ones((1, 2, 4), device=DEVICE, dtype=torch.float32)
    k = torch.ones((1, 1, 4), device=DEVICE, dtype=torch.float32)
    rope_paged_kv_append_ref(cache, 0, [1], q, k, k)
    cache.finish_request(1)

    with pytest.raises(RuntimeError, match="finished"):
        rope_paged_kv_append_ref(cache, 0, [1], q, k, k)


@pytest.mark.parametrize(
    ("positions", "rotary_dim", "match"),
    [
        (torch.tensor([0.0]), 4, "positions must use"),
        (torch.tensor([-1]), 4, "non-negative"),
        (torch.tensor([0]), 3, "positive, even"),
        (torch.tensor([0]), 6, "no larger"),
    ],
)
def test_apply_rope_rejects_invalid_inputs(positions, rotary_dim, match):
    x = torch.ones((1, 1, 4), device=DEVICE, dtype=torch.float32)
    positions = positions.to(DEVICE)
    with pytest.raises(ValueError, match=match):
        apply_rope(x, positions, rotary_dim=rotary_dim)
