"""Internal JIT-built fused RoPE + paged K/V append CUDA primitive."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import shutil


def _torch():
    import torch

    return torch


def _require_cuda_toolchain():
    torch = _torch()
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available

    if not torch.cuda.is_available():
        raise RuntimeError("fused RoPE + KV append requires torch.cuda.is_available()")
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set. Export CUDA_HOME=/usr/local/cuda-12.8 before starting Python."
        )
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is not on PATH. Export $CUDA_HOME/bin into PATH before building.")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to JIT-build fused RoPE + KV append.")


@lru_cache(maxsize=1)
def load_fused_rope_kv_append_extension():
    """Build and cache the fused CUDA extension on its first explicit use."""
    _require_cuda_toolchain()
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).with_name("csrc")
    verbose = os.environ.get("FLASHDEC_CUDA_VERBOSE", "0") == "1"
    return load(
        name="flashdec_fused_rope_kv_append_cuda_v1",
        sources=[
            str(source_dir / "fused_rope_kv_append.cpp"),
            str(source_dir / "fused_rope_kv_append_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def _check_cuda_contiguous_tensor(name, tensor):
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(q, k, v, k_cache, v_cache, block_ids, block_offsets, positions, rotary_dim, base):
    torch = _torch()
    for name, tensor in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_ids", block_ids),
        ("block_offsets", block_offsets),
        ("positions", positions),
    ]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        _check_cuda_contiguous_tensor(name, tensor)

    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, and v must have shape [batch, heads, head_dim]")
    if q.shape[1] <= 0 or k.shape[1] <= 0:
        raise ValueError("q and k must have at least one head")
    if v.shape != k.shape:
        raise ValueError("v must match k shape")
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must match shape [blocks, kv_heads, block_size, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q batch and head_dim must match k/v")
    if k.shape[1] != k_cache.shape[1] or k.shape[2] != k_cache.shape[3]:
        raise ValueError("k/v heads and head_dim must match cache")
    if block_ids.dim() != 1 or block_offsets.dim() != 1 or positions.dim() != 1:
        raise ValueError("block_ids, block_offsets, and positions must be rank-1 tensors")
    if block_ids.numel() != q.shape[0] or block_offsets.numel() != q.shape[0]:
        raise ValueError("block_ids and block_offsets lengths must match batch")
    if positions.numel() != q.shape[0]:
        raise ValueError("positions length must match batch")
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in (block_ids, block_offsets, positions)):
        raise ValueError("block_ids, block_offsets, and positions must use int32 or int64")
    if any(tensor.device != q.device for tensor in (k, v, k_cache, v_cache, block_ids, block_offsets, positions)):
        raise ValueError("all fused RoPE + KV append inputs must share one CUDA device")
    if any(tensor.dtype != q.dtype for tensor in (k, v, k_cache, v_cache)):
        raise ValueError("q, k, v, and K/V caches must share one dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("fused RoPE + KV append supports float16, bfloat16, and float32")

    head_dim = q.shape[-1]
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int):
        raise ValueError("rotary_dim must be an even integer")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
    base = float(base)
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("base must be a positive finite number")

    if bool(torch.any(block_ids < 0).item()) or bool(torch.any(block_ids >= k_cache.shape[0]).item()):
        raise ValueError("block_ids must be within [0, num_blocks)")
    if bool(torch.any(block_offsets < 0).item()) or bool(torch.any(block_offsets >= k_cache.shape[2]).item()):
        raise ValueError("block_offsets must be within [0, block_size)")
    if bool(torch.any(positions < 0).item()):
        raise ValueError("positions must be non-negative")
    return base


def fused_rope_kv_append(
    q,
    k,
    v,
    k_cache,
    v_cache,
    block_ids,
    block_offsets,
    positions,
    rotary_dim=None,
    base=10_000.0,
):
    """Rotate Q/K and append rotated K/raw V through one CUDA kernel.

    The caller supplies already allocated physical ``block_ids`` and offsets.
    The returned Q has the same shape/dtype as ``q``; K/V cache writes happen
    in-place. This primitive does not mutate Python request or allocator state.
    """
    if rotary_dim is None:
        if not isinstance(q, _torch().Tensor):
            raise TypeError("q must be a torch.Tensor")
        rotary_dim = q.shape[-1]
    base = _validate_inputs(
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_ids,
        block_offsets,
        positions,
        rotary_dim,
        base,
    )
    q_out = _torch().empty_like(q)
    extension = load_fused_rope_kv_append_extension()
    extension.fused_rope_kv_append(
        q_out,
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_ids.to(dtype=_torch().int64).contiguous(),
        block_offsets.to(dtype=_torch().int64).contiguous(),
        positions.to(dtype=_torch().int64).contiguous(),
        int(rotary_dim),
        float(base),
    )
    return q_out
