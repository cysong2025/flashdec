"""Correctness coverage for the fused RoPE + paged K/V append CUDA primitive."""

import shutil

import pytest

torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

import flashdec
import flashdec._fused_rope_kv_append as fused_module
from flashdec._fused_rope_kv_append import (
    _fused_rope_kv_append_trusted,
    fused_rope_kv_append,
)
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
    assert "_fused_rope_kv_append_trusted" not in flashdec.__all__
    assert not hasattr(flashdec, "_fused_rope_kv_append_trusted")


def test_fused_rope_kv_append_rejects_cpu_inputs_before_building_extension():
    q = torch.zeros((1, 2, 4), dtype=torch.float32)
    k = torch.zeros((1, 1, 4), dtype=torch.float32)
    cache = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    locations = torch.zeros((1,), dtype=torch.int64)

    with pytest.raises(ValueError, match="q must be a CUDA tensor"):
        fused_rope_kv_append(q, k, k, cache, cache, locations, locations, locations)


def test_checked_and_trusted_raw_dispatch_differ_only_in_value_validation(
    monkeypatch,
):
    prepared = []
    validated = []
    launched = []
    sentinel = object()
    q = torch.zeros((1, 2, 4), dtype=torch.float32)
    k = torch.zeros((1, 1, 4), dtype=torch.float32)
    cache = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
    indices = torch.zeros((1,), dtype=torch.int64)

    def prepare(*_args, **_kwargs):
        prepared.append(True)
        return 4, 10_000.0

    def validate(*_args, **_kwargs):
        validated.append(True)

    def launch(*_args, **_kwargs):
        launched.append(True)
        return sentinel

    monkeypatch.setattr(fused_module, "_prepare_fused_rope_kv_append", prepare)
    monkeypatch.setattr(fused_module, "_validate_index_values", validate)
    monkeypatch.setattr(fused_module, "_launch_fused_rope_kv_append", launch)

    raw_args = (q, k, k, cache, cache)
    assert fused_rope_kv_append(
        *raw_args,
        indices,
        indices,
        indices,
    ) is sentinel
    assert _fused_rope_kv_append_trusted(
        *raw_args,
        indices,
        indices,
        indices,
    ) is sentinel

    assert len(prepared) == 2
    assert len(validated) == 1
    assert len(launched) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("block_ids", "block_offsets", "positions", "message"),
    [
        ([-1], [0], [0], "block_ids"),
        ([2], [0], [0], "block_ids"),
        ([0], [-1], [0], "block_offsets"),
        ([0], [2], [0], "block_offsets"),
        ([0], [0], [-1], "positions"),
    ],
)
def test_public_fused_primitive_keeps_device_value_checks(
    block_ids,
    block_offsets,
    positions,
    message,
    monkeypatch,
):
    q = torch.zeros((1, 2, 4), device="cuda", dtype=torch.float32)
    k = torch.zeros((1, 1, 4), device="cuda", dtype=torch.float32)
    cache = torch.zeros((2, 1, 2, 4), device="cuda", dtype=torch.float32)

    def fail_if_launched(*_args, **_kwargs):
        raise AssertionError("invalid public indices must not reach the CUDA launch")

    monkeypatch.setattr(
        fused_module,
        "_launch_fused_rope_kv_append",
        fail_if_launched,
    )

    with pytest.raises(ValueError, match=message):
        fused_rope_kv_append(
            q,
            k,
            k,
            cache,
            cache,
            torch.tensor(block_ids, device="cuda", dtype=torch.int64),
            torch.tensor(block_offsets, device="cuda", dtype=torch.int64),
            torch.tensor(positions, device="cuda", dtype=torch.int64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_trusted_fused_primitive_skips_only_device_value_checks(monkeypatch):
    q = torch.zeros((1, 2, 4), device="cuda", dtype=torch.float32)
    k = torch.zeros((1, 1, 4), device="cuda", dtype=torch.float32)
    cache = torch.zeros((2, 1, 2, 4), device="cuda", dtype=torch.float32)
    block_ids = torch.zeros((1,), device="cuda", dtype=torch.int64)
    block_offsets = torch.zeros((1,), device="cuda", dtype=torch.int64)
    positions = torch.zeros((1,), device="cuda", dtype=torch.int64)

    def fail_value_validation(*_args, **_kwargs):
        raise RuntimeError("device-value validation called")

    monkeypatch.setattr(fused_module, "_validate_index_values", fail_value_validation)
    monkeypatch.setattr(
        fused_module,
        "_launch_fused_rope_kv_append",
        lambda *_args, **_kwargs: q,
    )

    with pytest.raises(RuntimeError, match="device-value validation called"):
        fused_rope_kv_append(
            q,
            k,
            k,
            cache,
            cache,
            block_ids,
            block_offsets,
            positions,
        )
    assert (
        _fused_rope_kv_append_trusted(
            q,
            k,
            k,
            cache,
            cache,
            block_ids,
            block_offsets,
            positions,
        )
        is q
    )
    with pytest.raises(ValueError, match="trusted transaction metadata must use int64"):
        _fused_rope_kv_append_trusted(
            q,
            k,
            k,
            cache,
            cache,
            block_ids.to(torch.int32),
            block_offsets,
            positions,
        )


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


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
@pytest.mark.parametrize("dtype", DTYPES)
def test_trusted_fused_primitive_matches_checked_primitive(dtype):
    torch.manual_seed(251)
    q = torch.randn((2, 4, 8), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 8), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    checked_k_cache = torch.zeros((4, 2, 3, 8), device="cuda", dtype=dtype)
    checked_v_cache = torch.zeros_like(checked_k_cache)
    trusted_k_cache = torch.zeros_like(checked_k_cache)
    trusted_v_cache = torch.zeros_like(checked_k_cache)
    block_ids = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    block_offsets = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    positions = torch.tensor([0, 7], device="cuda", dtype=torch.int64)

    checked_q = fused_rope_kv_append(
        q,
        k,
        v,
        checked_k_cache,
        checked_v_cache,
        block_ids,
        block_offsets,
        positions,
        rotary_dim=4,
    )
    trusted_q = _fused_rope_kv_append_trusted(
        q,
        k,
        v,
        trusted_k_cache,
        trusted_v_cache,
        block_ids,
        block_offsets,
        positions,
        rotary_dim=4,
    )
    torch.cuda.synchronize()

    _assert_close(trusted_q, checked_q, dtype)
    _assert_close(trusted_k_cache, checked_k_cache, dtype)
    _assert_close(trusted_v_cache, checked_v_cache, dtype)
