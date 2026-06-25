"""Triton dense single-token decode attention kernel."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride_seq: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    k_stride_seq: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_seq: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
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

    offs_t = tl.arange(0, BLOCK_SEQ)
    for token_start in range(0, MAX_SEQ_LEN, BLOCK_SEQ):
        token_idxs = token_start + offs_t
        valid_tokens = (token_idxs < seq_len) & (token_idxs < MAX_SEQ_LEN)
        k = tl.load(
            k_ptr
            + seq_idx * k_stride_seq
            + token_idxs[:, None] * k_stride_token
            + kv_head * k_stride_head
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
            + seq_idx * v_stride_seq
            + token_idxs[:, None] * v_stride_token
            + kv_head * v_stride_head
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


def _validate_inputs(q, k_cache, v_cache, seq_lens):
    if q.device.type != "cuda" or k_cache.device.type != "cuda" or v_cache.device.type != "cuda":
        raise ValueError("q, k_cache, and v_cache must be CUDA tensors")
    if seq_lens.device.type != "cuda":
        raise ValueError("seq_lens must be a CUDA tensor")
    if q.device != k_cache.device or q.device != v_cache.device or q.device != seq_lens.device:
        raise ValueError("q, k_cache, v_cache, and seq_lens must be on the same CUDA device")
    if q.dim() != 3:
        raise ValueError("q must have shape [num_seqs, num_q_heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must have shape [num_seqs, max_seq_len, num_kv_heads, head_dim]")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must have shape [num_seqs]")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if q.dtype not in (torch.float16, torch.float32):
        raise ValueError("dense_decode_attention currently supports float16 and float32 tensors")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must be int32 or int64")

    num_seqs, num_q_heads, head_dim = q.shape
    k_num_seqs, max_seq_len, num_kv_heads, k_head_dim = k_cache.shape
    v_num_seqs, v_max_seq_len, v_num_kv_heads, v_head_dim = v_cache.shape
    if k_num_seqs != num_seqs or v_num_seqs != num_seqs:
        raise ValueError("q, k_cache, and v_cache must have the same num_seqs")
    if seq_lens.numel() != num_seqs:
        raise ValueError("seq_lens length must match num_seqs")
    if v_max_seq_len != max_seq_len:
        raise ValueError("k_cache and v_cache must have the same max_seq_len")
    if v_num_kv_heads != num_kv_heads:
        raise ValueError("k_cache and v_cache must have the same num_kv_heads")
    if k_head_dim != head_dim or v_head_dim != head_dim:
        raise ValueError("q, k_cache, and v_cache must have the same head_dim")
    if head_dim not in (64, 128):
        raise ValueError("dense_decode_attention currently supports head_dim 64 or 128")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")


def dense_decode_attention(q, k_cache, v_cache, seq_lens, sm_scale=None, block_seq=64, num_warps=4):
    """Return dense single-token decode attention using Triton.

    Shapes:
    - q: [num_seqs, num_q_heads, head_dim]
    - k_cache/v_cache: [num_seqs, max_seq_len, num_kv_heads, head_dim]
    - seq_lens: [num_seqs]
    - return: [num_seqs, num_q_heads, head_dim]
    """
    _validate_inputs(q, k_cache, v_cache, seq_lens)
    if block_seq not in (16, 32, 64, 128):
        raise ValueError("block_seq must be one of 16, 32, 64, or 128")

    q_contig = q.contiguous()
    k_contig = k_cache.contiguous()
    v_contig = v_cache.contiguous()
    seq_lens_contig = seq_lens.contiguous()
    num_seqs, num_q_heads, head_dim = q_contig.shape
    _, max_seq_len, num_kv_heads, _ = k_contig.shape
    group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q_contig)
    grid = (num_seqs, num_q_heads)
    _dense_decode_attention_kernel[grid](
        q_contig,
        k_contig,
        v_contig,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        MAX_SEQ_LEN=max_seq_len,
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_SEQ=block_seq,
        SM_SCALE=float(sm_scale),
        num_warps=num_warps,
    )
    return out
