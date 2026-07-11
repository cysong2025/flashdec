"""Triton paged single-token decode attention kernel."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride_seq: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_block: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_block: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_dim: tl.constexpr,
    block_tables_stride_seq: tl.constexpr,
    block_tables_stride_block: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    seq_idx = tl.program_id(axis=0)
    q_head = tl.program_id(axis=1)
    kv_head = q_head // GROUP_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + seq_idx * q_stride_seq + q_head * q_stride_head + offs_d * q_stride_dim,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    m_i = tl.full((), 0.0, dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK_SIZE)
    for logical_block in range(0, MAX_BLOCKS_PER_SEQ):
        physical_block = tl.load(
            block_tables_ptr
            + seq_idx * block_tables_stride_seq
            + logical_block * block_tables_stride_block
        )
        token_idxs = logical_block * BLOCK_SIZE + offs_t
        valid_tokens = (token_idxs < seq_len) & (physical_block >= 0)

        k = tl.load(
            k_ptr
            + physical_block * k_stride_block
            + kv_head * k_stride_head
            + offs_t[:, None] * k_stride_token
            + offs_d[None, :] * k_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new)
        l_new = l_i * alpha + tl.sum(probs, axis=0)

        v = tl.load(
            v_ptr
            + physical_block * v_stride_block
            + kv_head * v_stride_head
            + offs_t[:, None] * v_stride_token
            + offs_d[None, :] * v_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    denom = tl.where(seq_len > 0, l_i, 1.0)
    out = tl.where(seq_len > 0, acc / denom, 0.0)
    tl.store(
        out_ptr + seq_idx * out_stride_seq + q_head * out_stride_head + offs_d * out_stride_dim,
        out,
        mask=offs_d < HEAD_DIM,
    )


def _validate_inputs(q, k_cache, v_cache, block_tables, seq_lens, block_size, num_warps):
    if q.device.type != "cuda" or k_cache.device.type != "cuda" or v_cache.device.type != "cuda":
        raise ValueError("q, k_cache, and v_cache must be CUDA tensors")
    if block_tables.device.type != "cuda" or seq_lens.device.type != "cuda":
        raise ValueError("block_tables and seq_lens must be CUDA tensors")
    if (
        q.device != k_cache.device
        or q.device != v_cache.device
        or q.device != block_tables.device
        or q.device != seq_lens.device
    ):
        raise ValueError("q, k_cache, v_cache, block_tables, and seq_lens must be on the same CUDA device")
    if q.dim() != 3:
        raise ValueError("q must have shape [num_seqs, num_q_heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must have shape [num_blocks, num_kv_heads, block_size, head_dim]")
    if block_tables.dim() != 2:
        raise ValueError("block_tables must have shape [num_seqs, max_blocks_per_seq]")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must have shape [num_seqs]")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("paged_decode_attention currently supports float16 and bfloat16 tensors")
    if block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_tables must be int32 or int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must be int32 or int64")

    num_seqs, num_q_heads, head_dim = q.shape
    num_blocks, num_kv_heads, cache_block_size, k_head_dim = k_cache.shape
    v_num_blocks, v_num_kv_heads, v_block_size, v_head_dim = v_cache.shape
    if block_tables.shape[0] != num_seqs or seq_lens.numel() != num_seqs:
        raise ValueError("block_tables and seq_lens must have one row/value per sequence")
    if v_num_blocks != num_blocks:
        raise ValueError("k_cache and v_cache must have the same num_blocks")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_seqs <= 0:
        raise ValueError("num_seqs must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if block_tables.shape[1] <= 0:
        raise ValueError("max_blocks_per_seq must be positive")
    if v_num_kv_heads != num_kv_heads:
        raise ValueError("k_cache and v_cache must have the same num_kv_heads")
    if v_block_size != cache_block_size:
        raise ValueError("k_cache and v_cache must have the same block_size")
    if cache_block_size != block_size:
        raise ValueError("block_size must match k_cache/v_cache shape")
    if block_size not in (8, 16, 32):
        raise ValueError("paged_decode_attention currently supports block_size 8, 16, or 32")
    if k_head_dim != head_dim or v_head_dim != head_dim:
        raise ValueError("q, k_cache, and v_cache must have the same head_dim")
    if head_dim not in (64, 128):
        raise ValueError("paged_decode_attention currently supports head_dim 64 or 128")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")


def paged_decode_attention(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=None,
    block_size=16,
    num_warps=2,
):
    """Return paged single-token decode attention using Triton.

    Shapes:
    - q: [num_seqs, num_q_heads, head_dim]
    - k_cache/v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
    - block_tables: [num_seqs, max_blocks_per_seq]
    - seq_lens: [num_seqs]
    - return: [num_seqs, num_q_heads, head_dim]

    Supported block sizes are 8, 16, and 32. Week 8 default config:
    block_size=16 and num_warps=2 based on the current RTX 5070 sweep.
    """
    block_size = int(block_size)
    _validate_inputs(q, k_cache, v_cache, block_tables, seq_lens, block_size, num_warps)

    q_contig = q.contiguous()
    k_contig = k_cache.contiguous()
    v_contig = v_cache.contiguous()
    block_tables_contig = block_tables.contiguous()
    seq_lens_contig = seq_lens.contiguous()
    num_seqs, num_q_heads, head_dim = q_contig.shape
    _, num_kv_heads, _, _ = k_contig.shape
    max_blocks_per_seq = block_tables_contig.shape[1]
    group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q_contig)
    grid = (num_seqs, num_q_heads)
    _paged_decode_attention_kernel[grid](
        q_contig,
        k_contig,
        v_contig,
        block_tables_contig,
        seq_lens_contig,
        out,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        k_contig.stride(0),
        k_contig.stride(1),
        k_contig.stride(2),
        k_contig.stride(3),
        v_contig.stride(0),
        v_contig.stride(1),
        v_contig.stride(2),
        v_contig.stride(3),
        block_tables_contig.stride(0),
        block_tables_contig.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
        SM_SCALE=float(sm_scale),
        num_warps=num_warps,
    )
    return out
