"""Triton RMSNorm forward kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(
        x_ptr + row_idx * stride_x_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / n_cols
    scale = tl.rsqrt(variance + eps)
    out = x * scale * weight
    tl.store(out_ptr + row_idx * stride_out_row + offsets, out, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    """Apply RMSNorm over the last dimension of a 2D CUDA tensor."""
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("x and weight must be CUDA tensors")
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")
    if weight.dim() != 1:
        raise ValueError("weight must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("weight length must match x.shape[-1]")

    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    out = torch.empty_like(x_contig)
    n_rows, n_cols = x_contig.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    _rmsnorm_kernel[(n_rows,)](
        x_contig,
        weight_contig,
        out,
        n_cols,
        x_contig.stride(0),
        out.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
