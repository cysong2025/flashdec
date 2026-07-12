"""JIT-built CUDA K/V append primitive for the token-major PagedKVCache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import shutil


_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")


def _torch():
    import torch

    return torch


def _require_cuda_toolchain():
    torch = _torch()
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA KV append requires torch.cuda.is_available()")
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set. Export CUDA_HOME=/usr/local/cuda-12.8 before starting Python."
        )
    if shutil.which("nvcc") is None:
        raise RuntimeError("nvcc is not on PATH. Export $CUDA_HOME/bin into PATH before building.")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to JIT-build CUDA KV append. Install it in the active venv.")


@lru_cache(maxsize=1)
def load_cuda_kv_append_extension():
    """Build and cache the CUDA extension on its first explicit use.

    Importing :mod:`flashdec` never builds an extension. This keeps CPU-only
    reference tests usable while making CUDA build errors explicit at the call
    site that needs the native path.
    """
    _require_cuda_toolchain()
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).with_name("csrc")
    verbose = os.environ.get("FLASHDEC_CUDA_VERBOSE", "0") == "1"
    return load(
        name="flashdec_kv_append_cuda_v1",
        sources=[
            str(source_dir / "kv_append.cpp"),
            str(source_dir / "kv_append_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )


def _check_cuda_tensor(name, tensor):
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(k_cache, v_cache, block_ids, block_offsets, k, v):
    torch = _torch()
    for name, tensor in [
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_ids", block_ids),
        ("block_offsets", block_offsets),
        ("k", k),
        ("v", v),
    ]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        _check_cuda_tensor(name, tensor)

    if k_cache.device != v_cache.device or k_cache.device != k.device or k_cache.device != v.device:
        raise ValueError("K/V caches and values must share one CUDA device")
    if block_ids.device != k_cache.device or block_offsets.device != k_cache.device:
        raise ValueError("block_ids and block_offsets must be on the cache CUDA device")
    if k_cache.dim() != 4:
        raise ValueError("k_cache must have shape [blocks, heads, block_size, head_dim]")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must match k_cache shape")
    if k.dim() != 3 or v.dim() != 3:
        raise ValueError("k and v must have shape [batch, heads, head_dim]")
    if v.shape != k.shape:
        raise ValueError("v must match k shape")
    if k.shape[1:] != (k_cache.shape[1], k_cache.shape[3]):
        raise ValueError("k/v heads and head_dim must match cache")
    if block_ids.dim() != 1 or block_offsets.dim() != 1:
        raise ValueError("block_ids and block_offsets must be rank-1 tensors")
    if block_ids.numel() != k.shape[0] or block_offsets.numel() != k.shape[0]:
        raise ValueError("block_ids and block_offsets lengths must match batch")
    if block_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_ids must use int32 or int64")
    if block_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_offsets must use int32 or int64")
    if k_cache.dtype != v_cache.dtype or k_cache.dtype != k.dtype or k_cache.dtype != v.dtype:
        raise ValueError("K/V caches and values must share one dtype")
    if str(k_cache.dtype).removeprefix("torch.") not in _SUPPORTED_DTYPES:
        raise ValueError("CUDA KV append supports float16, bfloat16, and float32")

    if bool(torch.any(block_ids < 0).item()) or bool(torch.any(block_ids >= k_cache.shape[0]).item()):
        raise ValueError("block_ids must be within [0, num_blocks)")
    if bool(torch.any(block_offsets < 0).item()) or bool(torch.any(block_offsets >= k_cache.shape[2]).item()):
        raise ValueError("block_offsets must be within [0, block_size)")


def cuda_kv_append(k_cache, v_cache, block_ids, block_offsets, k, v):
    """Write one K/V token per request into token-major physical cache blocks.

    Args:
        k_cache/v_cache: contiguous CUDA tensors shaped
            ``[num_blocks, num_kv_heads, block_size, head_dim]``.
        block_ids/block_offsets: CUDA integer tensors shaped ``[batch]``.
        k/v: contiguous CUDA tensors shaped ``[batch, num_kv_heads, head_dim]``.

    The caller owns allocation and request-state updates. This function only
    writes values to validated physical locations and returns ``None``.
    """
    _validate_inputs(k_cache, v_cache, block_ids, block_offsets, k, v)
    extension = load_cuda_kv_append_extension()
    extension.kv_append(
        k_cache,
        v_cache,
        block_ids.to(dtype=_torch().int64).contiguous(),
        block_offsets.to(dtype=_torch().int64).contiguous(),
        k,
        v,
    )
