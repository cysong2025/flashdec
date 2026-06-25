"""PyTorch reference implementations used for correctness tests."""

from __future__ import annotations


def vector_add_ref(x, y):
    """Reference for elementwise addition."""
    return x + y


def row_softmax_ref(x):
    """Reference for row-wise softmax over the last dimension."""
    import torch

    return torch.softmax(x, dim=-1)


def rmsnorm_ref(x, weight, eps=1e-6):
    """Reference RMSNorm over the last dimension."""
    import torch

    x_fp32 = x.to(torch.float32)
    variance = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.to(torch.float32)).to(x.dtype)


def matmul_ref(a, b):
    """Reference 2D matrix multiplication."""
    import torch

    return torch.matmul(a, b)


def dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=None):
    """Reference dense single-token decode attention.

    Shapes:
    - q: [num_seqs, num_q_heads, head_dim]
    - k_cache/v_cache: [num_seqs, max_seq_len, num_kv_heads, head_dim]
    - seq_lens: [num_seqs]
    - return: [num_seqs, num_q_heads, head_dim]
    """
    import torch

    if q.dim() != 3:
        raise ValueError("q must have shape [num_seqs, num_q_heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must be a 1D tensor")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if not q.is_floating_point():
        raise ValueError("q, k_cache, and v_cache must be floating point tensors")

    num_seqs, num_q_heads, head_dim = q.shape
    k_num_seqs, max_seq_len, num_kv_heads, k_head_dim = k_cache.shape
    v_num_seqs, v_max_seq_len, v_num_kv_heads, v_head_dim = v_cache.shape
    if k_num_seqs != num_seqs or v_num_seqs != num_seqs:
        raise ValueError("q, k_cache, and v_cache must have the same num_seqs")
    if v_max_seq_len != max_seq_len:
        raise ValueError("k_cache and v_cache must have the same max_seq_len")
    if v_num_kv_heads != num_kv_heads:
        raise ValueError("k_cache and v_cache must have the same num_kv_heads")
    if k_head_dim != head_dim or v_head_dim != head_dim:
        raise ValueError("q, k_cache, and v_cache must have the same head_dim")
    if seq_lens.numel() != num_seqs:
        raise ValueError("seq_lens length must match num_seqs")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    seq_lens_list = [int(value) for value in seq_lens.detach().cpu().tolist()]
    if any(seq_len < 0 or seq_len > max_seq_len for seq_len in seq_lens_list):
        raise ValueError("seq_lens values must be in [0, max_seq_len]")

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    for seq_idx, seq_len in enumerate(seq_lens_list):
        if seq_len == 0:
            out[seq_idx].zero_()
            continue

        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            q_vec = q[seq_idx, q_head].to(torch.float32)
            k = k_cache[seq_idx, :seq_len, kv_head].to(torch.float32)
            v = v_cache[seq_idx, :seq_len, kv_head].to(torch.float32)

            scores = torch.matmul(k, q_vec) * sm_scale
            scores = scores - torch.max(scores)
            probs = torch.exp(scores)
            probs = probs / torch.sum(probs)
            out[seq_idx, q_head] = torch.matmul(probs, v).to(out.dtype)

    return out
