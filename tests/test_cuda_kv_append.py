"""Correctness coverage for the independently built CUDA K/V append path."""

import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
from flashdec.cache import PagedKVCache
from flashdec.cuda_kv_append import cuda_kv_append


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available() and CUDA_HOME is not None and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required for JIT extension tests"

DTYPES = [torch.float16, torch.float32]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)


def _make_cache(dtype, max_blocks=6):
    return PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        block_size=2,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )


def _assert_close(actual, expected, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_cuda_kv_append_is_public_api_without_building_extension():
    assert flashdec.cuda_kv_append is cuda_kv_append
    assert callable(flashdec.load_cuda_kv_append_extension)


def test_cuda_kv_append_rejects_cpu_inputs_before_building_extension():
    k_cache = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    values = torch.zeros((1, 1, 4), dtype=torch.float32)
    locations = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match="k_cache must be a CUDA tensor"):
        cuda_kv_append(k_cache, k_cache, locations, locations, values, values)


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_kv_append_writes_requested_physical_slots(dtype):
    torch.manual_seed(311)
    k_cache = torch.zeros((4, 2, 3, 8), device="cuda", dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    k = torch.randn((2, 2, 8), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    block_ids = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    block_offsets = torch.tensor([2, 0], device="cuda", dtype=torch.int64)

    cuda_kv_append(k_cache, v_cache, block_ids, block_offsets, k, v)
    torch.cuda.synchronize()

    _assert_close(k_cache[3, :, 2, :], k[0], dtype)
    _assert_close(v_cache[3, :, 2, :], v[0], dtype)
    _assert_close(k_cache[1, :, 0, :], k[1], dtype)
    _assert_close(v_cache[1, :, 0, :], v[1], dtype)
    assert torch.count_nonzero(k_cache[0]).item() == 0
    assert torch.count_nonzero(v_cache[2]).item() == 0


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_cuda_kv_append_rejects_out_of_range_physical_location():
    cache = torch.zeros((2, 1, 2, 4), device="cuda", dtype=torch.float16)
    values = torch.ones((1, 1, 4), device="cuda", dtype=torch.float16)
    valid_offset = torch.zeros((1,), device="cuda", dtype=torch.int64)
    invalid_block = torch.tensor([2], device="cuda", dtype=torch.int64)
    invalid_offset = torch.tensor([2], device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="block_ids must be within"):
        cuda_kv_append(cache, cache, invalid_block, valid_offset, values, values)
    with pytest.raises(ValueError, match="block_offsets must be within"):
        cuda_kv_append(cache, cache, valid_offset, invalid_offset, values, values)


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", DTYPES)
def test_paged_kv_cache_append_cuda_matches_python_allocator_and_writes(dtype):
    torch.manual_seed(313)
    reference = _make_cache(dtype)
    native = _make_cache(dtype)
    schedule = ([10, 20], [10], [20], [10])

    for request_ids in schedule:
        batch_size = len(request_ids)
        k = torch.randn((batch_size, 2, 8), device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        expected_tables = reference.append(0, request_ids, k, v)
        actual_tables = native.append_cuda(0, request_ids, k, v)
        _assert_close(actual_tables, expected_tables, dtype)

    torch.cuda.synchronize()
    _assert_close(native.k_cache, reference.k_cache, dtype)
    _assert_close(native.v_cache, reference.v_cache, dtype)
    assert native.request_state(10) == reference.request_state(10)
    assert native.request_state(20) == reference.request_state(20)
    assert native.metrics() == reference.metrics()
    assert native.validate_invariants()


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_append_cuda_capacity_failure_keeps_allocator_state_atomic():
    cache = _make_cache(torch.float16, max_blocks=1)
    k = torch.ones((1, 2, 8), device="cuda", dtype=torch.float16)
    cache.append_cuda(0, [1], k, k)
    before = cache.request_state(1)

    with pytest.raises(RuntimeError, match="out of physical blocks"):
        cache.append_cuda(0, [2], k, k)

    assert cache.request_state(1) == before
    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(2)
    assert cache.metrics()["capacity_failure_count"] == 1
    assert cache.validate_invariants()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_append_cuda_rejects_non_contiguous_values_before_allocator_mutation():
    cache = _make_cache(torch.float16)
    values = torch.ones((1, 8, 2), device="cuda", dtype=torch.float16).transpose(1, 2)
    assert values.shape == (1, 2, 8)
    assert not values.is_contiguous()

    with pytest.raises(ValueError, match="requires contiguous"):
        cache.append_cuda(0, [1], values, values)

    with pytest.raises(KeyError, match="unknown request_id"):
        cache.request_state(1)
    assert cache.num_used_blocks == 0
