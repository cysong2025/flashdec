"""Correctness coverage for the fused RoPE + paged K/V append CUDA primitive."""

import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec._fused_rope_kv_append import fused_rope_kv_append
from flashdec.rope import apply_rope


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required for fused CUDA tests"

DTYPES = [torch.float16, torch.float32]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)


def _assert_close(actual, expected, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    elif dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_fused_rope_kv_append_is_lazy_public_api_without_building_extension():
    assert flashdec.fused_rope_kv_append is fused_rope_kv_append
    assert callable(flashdec.load_fused_rope_kv_append_extension)


def test_fused_rope_kv_append_rejects_cpu_inputs_before_building_extension():
    q = torch.zeros((1, 2, 4), dtype=torch.float32)
    k = torch.zeros((1, 1, 4), dtype=torch.float32)
    cache = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    locations = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match="q must be a CUDA tensor"):
        fused_rope_kv_append(q, k, k, cache, cache, locations, locations, locations)


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_rope_kv_append_matches_pytorch_rope_and_paged_slots(dtype):
    torch.manual_seed(241)
    q = torch.randn((2, 4, 8), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 8), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    k_cache = torch.zeros((4, 2, 3, 8), device="cuda", dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_ids = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    block_offsets = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    positions = torch.tensor([0, 7], device="cuda", dtype=torch.int64)

    actual_q = fused_rope_kv_append(
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_ids,
        block_offsets,
        positions,
        rotary_dim=4,
    )
    expected_q = apply_rope(q, positions, rotary_dim=4)
    expected_k = apply_rope(k, positions, rotary_dim=4)
    torch.cuda.synchronize()

    _assert_close(actual_q, expected_q, dtype)
    _assert_close(k_cache[3, :, 2, :], expected_k[0], dtype)
    _assert_close(v_cache[3, :, 2, :], v[0], dtype)
    _assert_close(k_cache[1, :, 0, :], expected_k[1], dtype)
    _assert_close(v_cache[1, :, 0, :], v[1], dtype)
    assert torch.count_nonzero(k_cache[0]).item() == 0
    assert torch.count_nonzero(v_cache[2]).item() == 0
