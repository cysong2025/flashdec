"""Triton paged single-token decode attention kernel."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_KV_LAYOUT_AXES = {
    "token_major": (2, 3),
    "dim_major": (3, 2),
}


def _layout_axes(kv_layout):
    try:
        return _KV_LAYOUT_AXES[kv_layout]
    except KeyError as exc:
        raise ValueError("kv_layout must be 'token_major' or 'dim_major'") from exc


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
    num_logical_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for logical_block in tl.range(0, num_logical_blocks):
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


@triton.jit
def _paged_decode_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    out_ptr,
    append_k_ptr,
    append_v_ptr,
    slot_mapping_ptr,
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
    append_k_stride_seq: tl.constexpr,
    append_k_stride_head: tl.constexpr,
    append_k_stride_dim: tl.constexpr,
    append_v_stride_seq: tl.constexpr,
    append_v_stride_head: tl.constexpr,
    append_v_stride_dim: tl.constexpr,
    slot_mapping_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    FUSE_KV_APPEND: tl.constexpr,
):
    """Decode one GQA KV head and all of its query heads together.

    Qwen2.5-3B maps eight query heads to each KV head. Grouping those heads
    keeps one K/V tile in the same program instead of issuing eight independent
    cache reads. GROUP_BLOCK pads the query-head dimension to a tensor-core
    friendly power of two while stores remain masked to GROUP_SIZE.
    """
    seq_idx = tl.program_id(axis=0)
    kv_head = tl.program_id(axis=1)
    offs_g = tl.arange(0, GROUP_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_heads = kv_head * GROUP_SIZE + offs_g
    valid_heads = offs_g < GROUP_SIZE

    q = tl.load(
        q_ptr
        + seq_idx * q_stride_seq
        + q_heads[:, None] * q_stride_head
        + offs_d[None, :] * q_stride_dim,
        mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if FUSE_KV_APPEND:
        append_k = tl.load(
            append_k_ptr
            + seq_idx * append_k_stride_seq
            + kv_head * append_k_stride_head
            + offs_d * append_k_stride_dim,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        )
        append_v = tl.load(
            append_v_ptr
            + seq_idx * append_v_stride_seq
            + kv_head * append_v_stride_head
            + offs_d * append_v_stride_dim,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        )
        append_slot = tl.load(
            slot_mapping_ptr + seq_idx * slot_mapping_stride
        )
        append_block = append_slot // BLOCK_SIZE
        append_offset = append_slot % BLOCK_SIZE
        tl.store(
            k_ptr
            + append_block * k_stride_block
            + kv_head * k_stride_head
            + append_offset * k_stride_token
            + offs_d * k_stride_dim,
            append_k,
            mask=(append_slot >= 0) & (offs_d < HEAD_DIM),
        )
        tl.store(
            v_ptr
            + append_block * v_stride_block
            + kv_head * v_stride_head
            + append_offset * v_stride_token
            + offs_d * v_stride_dim,
            append_v,
            mask=(append_slot >= 0) & (offs_d < HEAD_DIM),
        )
    if FUSE_KV_APPEND:
        append_score = tl.sum(
            q.to(tl.float32) * append_k[None, :].to(tl.float32), axis=1
        ) * SM_SCALE
        m_i = append_score
        l_i = tl.full((GROUP_BLOCK,), 1.0, dtype=tl.float32)
        acc = tl.broadcast_to(append_v[None, :], (GROUP_BLOCK, HEAD_DIM)).to(
            tl.float32
        )
    else:
        m_i = tl.full((GROUP_BLOCK,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP_BLOCK,), dtype=tl.float32)
        acc = tl.zeros((GROUP_BLOCK, HEAD_DIM), dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK_SIZE)
    num_logical_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for logical_block in tl.range(0, num_logical_blocks):
        physical_block = tl.load(
            block_tables_ptr
            + seq_idx * block_tables_stride_seq
            + logical_block * block_tables_stride_block
        )
        token_idxs = logical_block * BLOCK_SIZE + offs_t
        if FUSE_KV_APPEND:
            valid_tokens = (token_idxs < seq_len - 1) & (physical_block >= 0)
        else:
            valid_tokens = (token_idxs < seq_len) & (physical_block >= 0)

        k = tl.load(
            k_ptr
            + physical_block * k_stride_block
            + kv_head * k_stride_head
            + offs_t[:, None] * k_stride_token
            + offs_d[None, :] * k_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SM_SCALE
        scores = tl.where(valid_tokens[None, :], scores, -float("inf"))

        block_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(probs, axis=1)

        v = tl.load(
            v_ptr
            + physical_block * v_stride_block
            + kv_head * v_stride_head
            + offs_t[:, None] * v_stride_token
            + offs_d[None, :] * v_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(
            probs.to(v.dtype), v, out_dtype=tl.float32
        )
        m_i = m_new
        l_i = l_new

    denom = tl.where(seq_len > 0, l_i, 1.0)
    result = tl.where(seq_len > 0, acc / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + seq_idx * out_stride_seq
        + q_heads[:, None] * out_stride_head
        + offs_d[None, :] * out_stride_dim,
        result,
        mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


@triton.jit
def _paged_decode_gqa_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    split_acc_ptr,
    split_max_ptr,
    split_sum_ptr,
    append_k_ptr,
    append_v_ptr,
    slot_mapping_ptr,
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
    split_acc_stride_seq: tl.constexpr,
    split_acc_stride_head: tl.constexpr,
    split_acc_stride_split: tl.constexpr,
    split_acc_stride_dim: tl.constexpr,
    split_stats_stride_seq: tl.constexpr,
    split_stats_stride_head: tl.constexpr,
    split_stats_stride_split: tl.constexpr,
    append_k_stride_seq: tl.constexpr,
    append_k_stride_head: tl.constexpr,
    append_k_stride_dim: tl.constexpr,
    append_v_stride_seq: tl.constexpr,
    append_v_stride_head: tl.constexpr,
    append_v_stride_dim: tl.constexpr,
    slot_mapping_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SM_SCALE: tl.constexpr,
    FUSE_KV_APPEND: tl.constexpr,
):
    seq_idx = tl.program_id(axis=0)
    kv_head = tl.program_id(axis=1)
    split_idx = tl.program_id(axis=2)
    offs_g = tl.arange(0, GROUP_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_heads = kv_head * GROUP_SIZE + offs_g
    valid_heads = offs_g < GROUP_SIZE

    q = tl.load(
        q_ptr
        + seq_idx * q_stride_seq
        + q_heads[:, None] * q_stride_head
        + offs_d[None, :] * q_stride_dim,
        mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_logical_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    blocks_per_split = tl.cdiv(num_logical_blocks, NUM_SPLITS)
    first_block = split_idx * blocks_per_split
    last_block = tl.minimum(first_block + blocks_per_split, num_logical_blocks)

    if FUSE_KV_APPEND:
        append_slot = tl.load(
            slot_mapping_ptr + seq_idx * slot_mapping_stride
        )
        append_block = append_slot // BLOCK_SIZE
        append_offset = append_slot % BLOCK_SIZE
        append_logical_block = (seq_len - 1) // BLOCK_SIZE
        owns_append = (
            (append_logical_block >= first_block)
            & (append_logical_block < last_block)
            & (append_slot >= 0)
        )
        append_k = tl.load(
            append_k_ptr
            + seq_idx * append_k_stride_seq
            + kv_head * append_k_stride_head
            + offs_d * append_k_stride_dim,
            mask=owns_append & (offs_d < HEAD_DIM),
            other=0.0,
        )
        append_v = tl.load(
            append_v_ptr
            + seq_idx * append_v_stride_seq
            + kv_head * append_v_stride_head
            + offs_d * append_v_stride_dim,
            mask=owns_append & (offs_d < HEAD_DIM),
            other=0.0,
        )
        tl.store(
            k_ptr
            + append_block * k_stride_block
            + kv_head * k_stride_head
            + append_offset * k_stride_token
            + offs_d * k_stride_dim,
            append_k,
            mask=owns_append & (offs_d < HEAD_DIM),
        )
        tl.store(
            v_ptr
            + append_block * v_stride_block
            + kv_head * v_stride_head
            + append_offset * v_stride_token
            + offs_d * v_stride_dim,
            append_v,
            mask=owns_append & (offs_d < HEAD_DIM),
        )

    if FUSE_KV_APPEND:
        append_score = tl.sum(
            q.to(tl.float32) * append_k[None, :].to(tl.float32), axis=1
        ) * SM_SCALE
        m_i = tl.where(owns_append, append_score, -float("inf"))
        l_i = tl.where(owns_append, 1.0, 0.0).to(tl.float32)
        append_acc = tl.broadcast_to(
            append_v[None, :], (GROUP_BLOCK, HEAD_DIM)
        ).to(tl.float32)
        acc = tl.where(owns_append, append_acc, 0.0)
    else:
        m_i = tl.full((GROUP_BLOCK,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((GROUP_BLOCK,), dtype=tl.float32)
        acc = tl.zeros((GROUP_BLOCK, HEAD_DIM), dtype=tl.float32)
    offs_t = tl.arange(0, BLOCK_SIZE)
    for logical_block in tl.range(first_block, last_block):
        physical_block = tl.load(
            block_tables_ptr
            + seq_idx * block_tables_stride_seq
            + logical_block * block_tables_stride_block
        )
        token_idxs = logical_block * BLOCK_SIZE + offs_t
        if FUSE_KV_APPEND:
            valid_tokens = (token_idxs < seq_len - 1) & (physical_block >= 0)
        else:
            valid_tokens = (token_idxs < seq_len) & (physical_block >= 0)
        k = tl.load(
            k_ptr
            + physical_block * k_stride_block
            + kv_head * k_stride_head
            + offs_t[:, None] * k_stride_token
            + offs_d[None, :] * k_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SM_SCALE
        scores = tl.where(valid_tokens[None, :], scores, -float("inf"))
        block_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(probs, axis=1)
        v = tl.load(
            v_ptr
            + physical_block * v_stride_block
            + kv_head * v_stride_head
            + offs_t[:, None] * v_stride_token
            + offs_d[None, :] * v_stride_dim,
            mask=valid_tokens[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(
            probs.to(v.dtype), v, out_dtype=tl.float32
        )
        m_i = m_new
        l_i = l_new

    tl.store(
        split_acc_ptr
        + seq_idx * split_acc_stride_seq
        + q_heads[:, None] * split_acc_stride_head
        + split_idx * split_acc_stride_split
        + offs_d[None, :] * split_acc_stride_dim,
        acc,
        mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
    )
    stats_offsets = (
        seq_idx * split_stats_stride_seq
        + q_heads * split_stats_stride_head
        + split_idx * split_stats_stride_split
    )
    tl.store(split_max_ptr + stats_offsets, m_i, mask=valid_heads)
    tl.store(split_sum_ptr + stats_offsets, l_i, mask=valid_heads)


@triton.jit
def _paged_decode_gqa_reduce_kernel(
    split_acc_ptr,
    split_max_ptr,
    split_sum_ptr,
    seq_lens_ptr,
    out_ptr,
    split_acc_stride_seq: tl.constexpr,
    split_acc_stride_head: tl.constexpr,
    split_acc_stride_split: tl.constexpr,
    split_acc_stride_dim: tl.constexpr,
    split_stats_stride_seq: tl.constexpr,
    split_stats_stride_head: tl.constexpr,
    split_stats_stride_split: tl.constexpr,
    out_stride_seq: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    seq_idx = tl.program_id(axis=0)
    kv_head = tl.program_id(axis=1)
    offs_g = tl.arange(0, GROUP_BLOCK)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)
    q_heads = kv_head * GROUP_SIZE + offs_g
    valid_heads = offs_g < GROUP_SIZE

    stats_offsets = (
        seq_idx * split_stats_stride_seq
        + q_heads[:, None] * split_stats_stride_head
        + offs_s[None, :] * split_stats_stride_split
    )
    local_max = tl.load(
        split_max_ptr + stats_offsets,
        mask=valid_heads[:, None],
        other=-float("inf"),
    )
    global_max = tl.max(local_max, axis=1)
    denominator = tl.zeros((GROUP_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((GROUP_BLOCK, HEAD_DIM), dtype=tl.float32)
    for split_idx in range(0, NUM_SPLITS):
        stat_offset = (
            seq_idx * split_stats_stride_seq
            + q_heads * split_stats_stride_head
            + split_idx * split_stats_stride_split
        )
        split_max = tl.load(
            split_max_ptr + stat_offset,
            mask=valid_heads,
            other=-float("inf"),
        )
        split_sum = tl.load(
            split_sum_ptr + stat_offset,
            mask=valid_heads,
            other=0.0,
        )
        weight = tl.exp(split_max - global_max)
        split_acc = tl.load(
            split_acc_ptr
            + seq_idx * split_acc_stride_seq
            + q_heads[:, None] * split_acc_stride_head
            + split_idx * split_acc_stride_split
            + offs_d[None, :] * split_acc_stride_dim,
            mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        denominator += split_sum * weight
        acc += split_acc * weight[:, None]

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    denominator = tl.where(seq_len > 0, denominator, 1.0)
    result = tl.where(seq_len > 0, acc / denominator[:, None], 0.0)
    tl.store(
        out_ptr
        + seq_idx * out_stride_seq
        + q_heads[:, None] * out_stride_head
        + offs_d[None, :] * out_stride_dim,
        result,
        mask=valid_heads[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


def _validate_inputs(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    block_size,
    num_warps,
    num_stages,
    kv_layout,
):
    token_axis, dim_axis = _layout_axes(kv_layout)
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
        raise ValueError("k_cache and v_cache must be 4D tensors")
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
    num_blocks, num_kv_heads = k_cache.shape[:2]
    v_num_blocks, v_num_kv_heads = v_cache.shape[:2]
    cache_block_size = k_cache.shape[token_axis]
    k_head_dim = k_cache.shape[dim_axis]
    v_block_size = v_cache.shape[token_axis]
    v_head_dim = v_cache.shape[dim_axis]
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
    if num_stages is not None and (
        isinstance(num_stages, bool)
        or not isinstance(num_stages, int)
        or num_stages not in (1, 2, 3, 4)
    ):
        raise ValueError("num_stages must be None or one of 1, 2, 3, or 4")


def paged_decode_attention(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=None,
    block_size=None,
    num_warps=2,
    kv_layout="token_major",
    num_stages=None,
):
    """Return paged single-token decode attention using Triton.

    Shapes:
    - q: [num_seqs, num_q_heads, head_dim]
    - token_major k_cache/v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
    - dim_major k_cache/v_cache: [num_blocks, num_kv_heads, head_dim, block_size]
    - block_tables: [num_seqs, max_blocks_per_seq]
    - seq_lens: [num_seqs]
    - return: [num_seqs, num_q_heads, head_dim]

    Supported block sizes are 8, 16, and 32. When block_size is omitted, it
    is inferred from k_cache/v_cache according to kv_layout. The current
    benchmark default is block_size=32 and num_warps=2. When num_stages is
    omitted, Triton selects its implicit staging configuration.
    """
    token_axis, dim_axis = _layout_axes(kv_layout)
    if block_size is None:
        if k_cache.dim() != 4:
            raise ValueError("k_cache must be a 4D tensor")
        block_size = k_cache.shape[token_axis]
    else:
        block_size = int(block_size)
    out = torch.empty_like(q)
    return paged_decode_attention_into(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        out,
        sm_scale=sm_scale,
        block_size=block_size,
        num_warps=num_warps,
        kv_layout=kv_layout,
        num_stages=num_stages,
    )


def paged_decode_attention_into(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    out,
    sm_scale=None,
    block_size=None,
    num_warps=2,
    kv_layout="token_major",
    num_stages=None,
    split_kv_workspace=None,
    num_splits=1,
    append_k=None,
    append_v=None,
    slot_mapping=None,
):
    """Write paged single-token decode attention into ``out``.

    Unlike :func:`paged_decode_attention`, this entry point performs no output
    allocation. K/V and block-table tensors may be strided views; their strides
    are passed to Triton directly instead of materializing a cache-sized copy.
    This is the integration path used by runtimes that own their KV storage.
    ``append_k``, ``append_v``, and ``slot_mapping`` may be provided together
    for grouped GQA decode. The kernel then stores the current K/V token and
    consumes those register values directly for attention, avoiding a separate
    cache-update launch and any cross-program visibility dependency.
    """
    token_axis, dim_axis = _layout_axes(kv_layout)
    if block_size is None:
        if k_cache.dim() != 4:
            raise ValueError("k_cache must be a 4D tensor")
        block_size = k_cache.shape[token_axis]
    else:
        block_size = int(block_size)

    _validate_inputs(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size,
        num_warps,
        num_stages,
        kv_layout,
    )
    if out.shape != q.shape:
        raise ValueError("out must have the same shape as q")
    if out.dtype != q.dtype:
        raise ValueError("out must have the same dtype as q")
    if out.device != q.device:
        raise ValueError("out must be on the same device as q")
    if isinstance(num_splits, bool) or not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError("num_splits must be a positive integer")

    # seq_lens is indexed as a dense vector by the kernel. The other inputs
    # carry explicit strides, which avoids copies for vLLM's NHD cache views.
    seq_lens_contig = seq_lens.contiguous()
    num_seqs, num_q_heads, head_dim = q.shape
    _, num_kv_heads = k_cache.shape[:2]
    group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    use_grouped_gqa = (
        head_dim == 128
        and group_size in (4, 8, 16)
        and block_size in (16, 32)
    )
    append_values = (append_k, append_v, slot_mapping)
    fuse_kv_append = all(value is not None for value in append_values)
    if any(value is not None for value in append_values) and not fuse_kv_append:
        raise ValueError(
            "append_k, append_v, and slot_mapping must be provided together"
        )
    if fuse_kv_append:
        if not use_grouped_gqa:
            raise ValueError("fused KV append requires grouped GQA decode")
        expected_append_shape = (num_seqs, num_kv_heads, head_dim)
        if append_k.shape != expected_append_shape or append_v.shape != expected_append_shape:
            raise ValueError(
                "append_k and append_v must have shape "
                "[num_seqs, num_kv_heads, head_dim]"
            )
        if append_k.dtype != q.dtype or append_v.dtype != q.dtype:
            raise ValueError("append_k and append_v must match q.dtype")
        if append_k.device != q.device or append_v.device != q.device:
            raise ValueError("append_k and append_v must be on q.device")
        if slot_mapping.dim() != 1 or slot_mapping.numel() < num_seqs:
            raise ValueError("slot_mapping must have at least one entry per sequence")
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError("slot_mapping must be int32 or int64")
        if slot_mapping.device != q.device:
            raise ValueError("slot_mapping must be on q.device")

    append_k_arg = append_k if fuse_kv_append else q
    append_v_arg = append_v if fuse_kv_append else q
    slot_mapping_arg = slot_mapping if fuse_kv_append else seq_lens_contig
    append_k_strides = append_k.stride() if fuse_kv_append else (0, 0, 0)
    append_v_strides = append_v.stride() if fuse_kv_append else (0, 0, 0)
    slot_mapping_stride = slot_mapping.stride(0) if fuse_kv_append else 0
    use_split_kv = use_grouped_gqa and num_splits > 1
    if use_split_kv:
        if split_kv_workspace is None or len(split_kv_workspace) != 3:
            raise ValueError("split_kv_workspace must contain acc, max, and sum tensors")
        split_acc, split_max, split_sum = split_kv_workspace
        expected_acc = (num_seqs, num_q_heads, num_splits, head_dim)
        expected_stats = (num_seqs, num_q_heads, num_splits)
        if split_acc.dim() != 4 or split_max.dim() != 3 or split_sum.dim() != 3:
            raise ValueError("split KV workspace ranks must be 4, 3, and 3")
        if any(
            tensor.device != q.device or tensor.dtype != torch.float32
            for tensor in split_kv_workspace
        ):
            raise ValueError("split KV workspace tensors must be float32 on q.device")
        if any(actual < expected for actual, expected in zip(split_acc.shape, expected_acc)):
            raise ValueError("split acc workspace is too small")
        if any(actual < expected for actual, expected in zip(split_max.shape, expected_stats)):
            raise ValueError("split max workspace is too small")
        if any(actual < expected for actual, expected in zip(split_sum.shape, expected_stats)):
            raise ValueError("split sum workspace is too small")
        if split_sum.stride() != split_max.stride():
            raise ValueError("split max and sum workspaces must have identical strides")
        split_grid = (num_seqs, num_kv_heads, num_splits)
        _paged_decode_gqa_split_kernel[split_grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens_contig,
            split_acc,
            split_max,
            split_sum,
            append_k_arg,
            append_v_arg,
            slot_mapping_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(token_axis),
            k_cache.stride(dim_axis),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(token_axis),
            v_cache.stride(dim_axis),
            block_tables.stride(0),
            block_tables.stride(1),
            split_acc.stride(0),
            split_acc.stride(1),
            split_acc.stride(2),
            split_acc.stride(3),
            split_max.stride(0),
            split_max.stride(1),
            split_max.stride(2),
            append_k_strides[0],
            append_k_strides[1],
            append_k_strides[2],
            append_v_strides[0],
            append_v_strides[1],
            append_v_strides[2],
            slot_mapping_stride,
            HEAD_DIM=head_dim,
            GROUP_SIZE=group_size,
            GROUP_BLOCK=16,
            BLOCK_SIZE=block_size,
            NUM_SPLITS=num_splits,
            SM_SCALE=float(sm_scale),
            FUSE_KV_APPEND=fuse_kv_append,
            num_warps=4,
        )
        reduce_grid = (num_seqs, num_kv_heads)
        _paged_decode_gqa_reduce_kernel[reduce_grid](
            split_acc,
            split_max,
            split_sum,
            seq_lens_contig,
            out,
            split_acc.stride(0),
            split_acc.stride(1),
            split_acc.stride(2),
            split_acc.stride(3),
            split_max.stride(0),
            split_max.stride(1),
            split_max.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            HEAD_DIM=head_dim,
            GROUP_SIZE=group_size,
            GROUP_BLOCK=16,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
        return out

    grid = (num_seqs, num_kv_heads if use_grouped_gqa else num_q_heads)
    launch_kwargs = {"num_warps": num_warps}
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages
    kernel = _paged_decode_gqa_kernel if use_grouped_gqa else _paged_decode_attention_kernel
    if use_grouped_gqa:
        launch_kwargs["num_warps"] = max(num_warps, 4)
    if use_grouped_gqa:
        kernel[grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens_contig,
            out,
            append_k_arg,
            append_v_arg,
            slot_mapping_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(token_axis),
            k_cache.stride(dim_axis),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(token_axis),
            v_cache.stride(dim_axis),
            block_tables.stride(0),
            block_tables.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            append_k_strides[0],
            append_k_strides[1],
            append_k_strides[2],
            append_v_strides[0],
            append_v_strides[1],
            append_v_strides[2],
            slot_mapping_stride,
            HEAD_DIM=head_dim,
            GROUP_SIZE=group_size,
            GROUP_BLOCK=16,
            BLOCK_SIZE=block_size,
            SM_SCALE=float(sm_scale),
            FUSE_KV_APPEND=fuse_kv_append,
            **launch_kwargs,
        )
    else:
        kernel[grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens_contig,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(token_axis),
            k_cache.stride(dim_axis),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(token_axis),
            v_cache.stride(dim_axis),
            block_tables.stride(0),
            block_tables.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            HEAD_DIM=head_dim,
            GROUP_SIZE=group_size,
            BLOCK_SIZE=block_size,
            SM_SCALE=float(sm_scale),
            **launch_kwargs,
        )
    return out
