"""PyTorch reference implementation for paged decode attention."""

from __future__ import annotations


_KV_LAYOUT_AXES = {
    "token_major": (2, 3),
    "dim_major": (3, 2),
}


def _layout_axes(kv_layout):
    try:
        return _KV_LAYOUT_AXES[kv_layout]
    except KeyError as exc:
        raise ValueError("kv_layout must be 'token_major' or 'dim_major'") from exc


def paged_decode_attention_ref(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=None,
    kv_layout="token_major",
):
    """Reference single-token decode attention over a paged KV cache.

    Shapes:
    - q: [num_seqs, num_q_heads, head_dim]
    - token_major k_cache/v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
    - dim_major k_cache/v_cache: [num_blocks, num_kv_heads, head_dim, block_size]
    - block_tables: [num_seqs, max_blocks_per_seq]
    - seq_lens: [num_seqs]
    - return: [num_seqs, num_q_heads, head_dim]
    """
    import torch

    token_axis, dim_axis = _layout_axes(kv_layout)

    if q.dim() != 3:
        raise ValueError("q must have shape [num_seqs, num_q_heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if block_tables.dim() != 2:
        raise ValueError("block_tables must have shape [num_seqs, max_blocks_per_seq]")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must be a 1D tensor")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if not q.is_floating_point():
        raise ValueError("q, k_cache, and v_cache must be floating point tensors")
    if block_tables.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("block_tables must contain integer physical block ids")

    num_seqs, num_q_heads, head_dim = q.shape
    num_blocks, num_kv_heads = k_cache.shape[:2]
    v_num_blocks, v_num_kv_heads = v_cache.shape[:2]
    block_size = k_cache.shape[token_axis]
    k_head_dim = k_cache.shape[dim_axis]
    v_block_size = v_cache.shape[token_axis]
    v_head_dim = v_cache.shape[dim_axis]
    if v_num_blocks != num_blocks:
        raise ValueError("k_cache and v_cache must have the same num_blocks")
    if v_num_kv_heads != num_kv_heads:
        raise ValueError("k_cache and v_cache must have the same num_kv_heads")
    if v_block_size != block_size:
        raise ValueError("k_cache and v_cache must have the same block_size")
    if k_head_dim != head_dim or v_head_dim != head_dim:
        raise ValueError("q, k_cache, and v_cache must have the same head_dim")
    if block_tables.shape[0] != num_seqs or seq_lens.numel() != num_seqs:
        raise ValueError("block_tables and seq_lens must have one row/value per sequence")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    max_blocks_per_seq = block_tables.shape[1]
    seq_lens_list = [int(value) for value in seq_lens.detach().cpu().tolist()]
    max_seq_len = max_blocks_per_seq * block_size
    if any(seq_len < 0 or seq_len > max_seq_len for seq_len in seq_lens_list):
        raise ValueError("seq_lens values must be in [0, max_blocks_per_seq * block_size]")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    for seq_idx, seq_len in enumerate(seq_lens_list):
        if seq_len == 0:
            out[seq_idx].zero_()
            continue

        logical_blocks = (seq_len + block_size - 1) // block_size
        block_ids_cpu = [int(value) for value in block_tables[seq_idx, :logical_blocks].detach().cpu().tolist()]
        if any(block_id < 0 or block_id >= num_blocks for block_id in block_ids_cpu):
            raise ValueError("block_tables contains invalid physical block ids for a non-empty sequence")

        block_ids = torch.tensor(block_ids_cpu, device=k_cache.device, dtype=torch.long)
        k_blocks = k_cache.index_select(0, block_ids)
        v_blocks = v_cache.index_select(0, block_ids)
        if kv_layout == "token_major":
            k_seq = k_blocks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
            v_seq = v_blocks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
        else:
            k_seq = k_blocks.permute(0, 3, 1, 2).reshape(-1, num_kv_heads, head_dim)
            v_seq = v_blocks.permute(0, 3, 1, 2).reshape(-1, num_kv_heads, head_dim)
        k_seq = k_seq[:seq_len]
        v_seq = v_seq[:seq_len]

        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            q_vec = q[seq_idx, q_head].to(torch.float32)
            k = k_seq[:, kv_head].to(torch.float32)
            v = v_seq[:, kv_head].to(torch.float32)

            scores = torch.matmul(k, q_vec) * sm_scale
            scores = scores - torch.max(scores)
            probs = torch.exp(scores)
            probs = probs / torch.sum(probs)
            out[seq_idx, q_head] = torch.matmul(probs, v).to(out.dtype)

    return out
