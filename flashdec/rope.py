"""PyTorch reference for decode-time RoPE and paged KV append."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class RopeAppendResult:
    """Outputs required by the next paged decode attention call."""

    q: Any
    positions: Any
    block_tables: Any
    seq_lens: Any


def apply_rope(x, positions, rotary_dim=None, base=10_000.0):
    """Apply split-half rotary position embedding with FP32 math.

    Args:
        x: [batch, num_heads, head_dim] floating point tensor.
        positions: [batch] int32/int64 tensor containing non-negative token positions.
        rotary_dim: Even prefix dimension to rotate. Defaults to head_dim.
        base: RoPE frequency base.

    Returns:
        Tensor with the same shape and dtype as x. Dimensions after rotary_dim
        are copied without modification.
    """
    torch = _torch()
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        raise TypeError("x and positions must be torch tensors")
    if x.dim() != 3:
        raise ValueError("x must have shape [batch, num_heads, head_dim]")
    if positions.dim() != 1 or positions.numel() != x.shape[0]:
        raise ValueError("positions must have shape [batch]")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("apply_rope supports float16, bfloat16, and float32")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must use int32 or int64")
    if positions.device != x.device:
        raise ValueError("x and positions must be on the same device")
    if bool(torch.any(positions < 0).item()):
        raise ValueError("positions must be non-negative")

    head_dim = x.shape[-1]
    if rotary_dim is None:
        rotary_dim = head_dim
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int):
        raise ValueError("rotary_dim must be an even integer")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
    base = float(base)
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("base must be a positive finite number")

    freq_steps = torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = torch.pow(base, -freq_steps / rotary_dim)
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]

    x_rot = x[..., :rotary_dim].to(torch.float32)
    half_dim = rotary_dim // 2
    x_first = x_rot[..., :half_dim]
    x_second = x_rot[..., half_dim:]
    rotated = torch.cat(
        (
            x_first * cos - x_second * sin,
            x_second * cos + x_first * sin,
        ),
        dim=-1,
    ).to(x.dtype)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def rope_paged_kv_append_ref(
    cache,
    layer_idx,
    request_ids,
    q,
    k,
    v,
    rotary_dim=None,
    base=10_000.0,
):
    """Rotate Q/K at pre-append positions, then append rotated K and raw V.

    This is the semantic baseline for the future CUDA data path. It returns
    rotated Q plus the metadata needed by paged decode attention.
    """
    from .cache import PagedKVCache

    if not isinstance(cache, PagedKVCache):
        raise TypeError("cache must be a PagedKVCache")
    ids = cache._normalize_request_ids(request_ids)
    if not ids:
        raise ValueError("request_ids must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("request_ids must be unique")

    if q.dim() != 3:
        raise ValueError("q must have shape [num_requests, num_q_heads, head_dim]")
    if k.dim() == 2 and len(ids) == 1:
        k = k.unsqueeze(0)
    if v.dim() == 2 and len(ids) == 1:
        v = v.unsqueeze(0)
    if k.dim() != 3 or v.dim() != 3:
        raise ValueError("k and v must have shape [num_requests, num_kv_heads, head_dim]")
    if q.shape[0] != len(ids) or q.shape[1] <= 0 or q.shape[2] != cache.head_dim:
        raise ValueError("q shape must match request count and cache head_dim")
    expected_kv_shape = (len(ids), cache.num_kv_heads, cache.head_dim)
    if k.shape != expected_kv_shape or v.shape != expected_kv_shape:
        raise ValueError("k and v shapes must match request count and cache dimensions")
    if q.device != cache.device or k.device != cache.device or v.device != cache.device:
        raise ValueError("q, k, and v must be on the cache device")
    if q.dtype != cache.dtype or k.dtype != cache.dtype or v.dtype != cache.dtype:
        raise ValueError("q, k, and v must use the cache dtype")

    positions = cache.next_positions(ids, device=cache.device)
    q_rotated = apply_rope(q, positions, rotary_dim=rotary_dim, base=base)
    k_rotated = apply_rope(k, positions, rotary_dim=rotary_dim, base=base)
    block_tables = cache.append(layer_idx, ids, k_rotated, v)
    seq_lens = cache.seq_lens_tensor(ids)
    return RopeAppendResult(
        q=q_rotated,
        positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )
