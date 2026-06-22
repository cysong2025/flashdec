"""Triton row-wise softmax kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(
        x_ptr + row_idx * stride_x_row + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(out_ptr + row_idx * stride_out_row + offsets, out, mask=mask)


def row_softmax(x):
    """Return softmax(x, dim=-1) for a 2D CUDA tensor."""
    if x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor")

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_rows, n_cols = x_contig.shape
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    _row_softmax_kernel[(n_rows,)](
        x_contig,
        out,
        n_cols,
        x_contig.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out

