"""Performance estimation helpers for FlashDec benchmarks."""

from __future__ import annotations

import math


def dtype_nbytes(dtype) -> int:
    """Return the byte width for a torch dtype object or dtype name."""
    name = str(dtype).replace("torch.", "")
    if name in {"float16", "bfloat16", "half"}:
        return 2
    if name in {"float32", "float", "int32", "uint32"}:
        return 4
    if name in {"float64", "double", "int64", "uint64"}:
        return 8
    raise ValueError(f"unsupported dtype for byte estimation: {dtype}")


def estimate_paged_decode_bytes(
    *,
    num_seqs: int,
    num_q_heads: int,
    head_dim: int,
    seq_lens,
    max_blocks_per_seq: int,
    block_size: int,
    dtype,
    block_table_entry_bytes: int = 4,
    seq_len_entry_bytes: int = 4,
) -> dict:
    """Estimate bytes touched by the current paged decode kernel.

    The estimate follows the current implementation: one Triton program handles
    one ``(sequence, q_head)`` pair, so K/V bytes are counted once per query
    head. It is a logical traffic estimate for comparing shapes; profiler
    counters remain the source of truth for actual memory transactions.
    """
    seq_lens = [int(value) for value in seq_lens]
    if num_seqs <= 0:
        raise ValueError("num_seqs must be positive")
    if len(seq_lens) != num_seqs:
        raise ValueError("seq_lens must contain one value per sequence")
    for name, value in [
        ("num_q_heads", num_q_heads),
        ("head_dim", head_dim),
        ("max_blocks_per_seq", max_blocks_per_seq),
        ("block_size", block_size),
        ("block_table_entry_bytes", block_table_entry_bytes),
        ("seq_len_entry_bytes", seq_len_entry_bytes),
    ]:
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if any(seq_len < 0 for seq_len in seq_lens):
        raise ValueError("seq_lens values must be non-negative")

    dtype_bytes = dtype_nbytes(dtype)
    num_programs = num_seqs * int(num_q_heads)
    total_context_tokens = sum(seq_lens)
    used_blocks = sum(math.ceil(seq_len / int(block_size)) for seq_len in seq_lens)

    q_read_bytes = num_programs * int(head_dim) * dtype_bytes
    out_write_bytes = q_read_bytes
    kv_read_bytes = total_context_tokens * int(num_q_heads) * 2 * int(head_dim) * dtype_bytes
    block_table_read_bytes = num_programs * int(max_blocks_per_seq) * int(block_table_entry_bytes)
    seq_lens_read_bytes = num_programs * int(seq_len_entry_bytes)
    total_bytes = q_read_bytes + out_write_bytes + kv_read_bytes + block_table_read_bytes + seq_lens_read_bytes

    return {
        "decode_tokens": int(num_seqs),
        "head_outputs": num_programs,
        "total_context_tokens": total_context_tokens,
        "estimated_used_blocks": used_blocks,
        "estimated_q_read_bytes": q_read_bytes,
        "estimated_kv_read_bytes": kv_read_bytes,
        "estimated_block_table_read_bytes": block_table_read_bytes,
        "estimated_seq_lens_read_bytes": seq_lens_read_bytes,
        "estimated_out_write_bytes": out_write_bytes,
        "estimated_total_bytes": total_bytes,
    }


def _per_second(units: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return float(units) * 1000.0 / float(latency_ms)


def _gb_per_second(num_bytes: int, latency_ms: float) -> float:
    return _per_second(num_bytes, latency_ms) / 1_000_000_000.0


def paged_decode_metric_metadata(estimate: dict, *, mean_ms: float, p50_ms: float) -> dict:
    """Return CSV-friendly derived metrics for a paged decode benchmark row."""
    for key in ["decode_tokens", "head_outputs", "estimated_kv_read_bytes", "estimated_total_bytes"]:
        if key not in estimate:
            raise ValueError(f"estimate is missing {key}")

    metadata = {key: estimate[key] for key in estimate}
    metadata.update(
        {
            "decode_tokens_per_s_mean": f"{_per_second(estimate['decode_tokens'], mean_ms):.2f}",
            "head_outputs_per_s_mean": f"{_per_second(estimate['head_outputs'], mean_ms):.2f}",
            "effective_kv_gbps_mean": f"{_gb_per_second(estimate['estimated_kv_read_bytes'], mean_ms):.4f}",
            "effective_total_gbps_mean": f"{_gb_per_second(estimate['estimated_total_bytes'], mean_ms):.4f}",
            "effective_kv_gbps_p50": f"{_gb_per_second(estimate['estimated_kv_read_bytes'], p50_ms):.4f}",
            "effective_total_gbps_p50": f"{_gb_per_second(estimate['estimated_total_bytes'], p50_ms):.4f}",
        }
    )
    return metadata
