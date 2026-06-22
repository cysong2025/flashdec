"""Triton vector add kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x, y, block_size=1024):
    """Return x + y using a Triton kernel."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("x and y must be CUDA tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")

    x_contig = x.contiguous()
    y_contig = y.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](x_contig, y_contig, out, n_elements, BLOCK_SIZE=block_size)
    return out.reshape_as(x)

