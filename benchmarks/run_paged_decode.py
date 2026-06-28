"""Run Week 6 paged decode attention Triton benchmarks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, benchmark_case, write_csv


DEFAULT_SHAPES = [
    (1, 8, 8, 64, 128),
    (4, 8, 8, 64, 512),
    (8, 16, 4, 64, 512),
    (16, 16, 4, 64, 1024),
]


def _parse_shape(text):
    parts = text.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "shape must be formatted as num_seqs,num_q_heads,num_kv_heads,head_dim,max_seq_len"
        )
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("shape values must be positive")
    num_seqs, num_q_heads, num_kv_heads, head_dim, _ = values
    if num_q_heads % num_kv_heads != 0:
        raise argparse.ArgumentTypeError("num_q_heads must be divisible by num_kv_heads")
    if head_dim != 64:
        raise argparse.ArgumentTypeError("Week 6 paged decode v1 requires head_dim 64")
    return values


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _common_metadata(torch, dtype_name, impl, shape, seq_lens, block_tables, cache, block_size):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype_name,
        "impl": impl,
        "op": "paged_decode_attention",
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "min_seq_len": int(seq_lens.min().item()),
        "max_actual_seq_len": int(seq_lens.max().item()),
        "block_size": block_size,
        "max_blocks_per_seq": block_tables.shape[1],
        "used_blocks": cache.num_used_blocks,
    }


def _with_speedup(result, ref_mean_ms):
    metadata = dict(result.metadata)
    metadata["speedup_vs_ref"] = f"{ref_mean_ms / result.mean_ms:.4f}"
    return BenchmarkResult(
        name=result.name,
        mean_ms=result.mean_ms,
        p50_ms=result.p50_ms,
        p90_ms=result.p90_ms,
        min_ms=result.min_ms,
        max_ms=result.max_ms,
        repeats=result.repeats,
        metadata=metadata,
    )


def _make_inputs(torch, shape, dtype, block_size):
    from flashdec.cache import PagedKVCache

    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    request_ids = list(range(num_seqs))
    low = max(1, max_seq_len // 2)
    seq_lens = torch.randint(low, max_seq_len + 1, (num_seqs,), device="cuda", dtype=torch.int32)
    seq_lens_list = [int(value) for value in seq_lens.detach().cpu().tolist()]
    max_blocks = sum((seq_len + block_size - 1) // block_size for seq_len in seq_lens_list)
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )

    token_k = torch.randn((num_seqs, max_seq_len, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    token_v = torch.randn_like(token_k)
    for step in range(max_seq_len):
        active_rows = [row for row, seq_len in enumerate(seq_lens_list) if step < seq_len]
        if not active_rows:
            continue
        active_ids = [request_ids[row] for row in active_rows]
        cache.append(
            layer_idx=0,
            request_ids=active_ids,
            k=token_k[active_rows, step],
            v=token_v[active_rows, step],
        )

    q = torch.randn((num_seqs, num_q_heads, head_dim), device="cuda", dtype=dtype)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    return q, cache.k_cache[0], cache.v_cache[0], block_tables, seq_lens, cache


def _validate(torch, q, k_cache, v_cache, block_tables, seq_lens, block_size, num_warps):
    from flashdec.kernels.paged_decode import paged_decode_attention
    from flashdec.paged_reference import paged_decode_attention_ref

    actual = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size=block_size,
        num_warps=num_warps,
    )
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def run_shape(torch, args, shape, dtype):
    from flashdec.kernels.paged_decode import paged_decode_attention
    from flashdec.paged_reference import paged_decode_attention_ref

    q, k_cache, v_cache, block_tables, seq_lens, cache = _make_inputs(torch, shape, dtype, args.block_size)
    results = []

    ref_result = benchmark_case(
        "paged_decode_attention_ref",
        lambda: paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=_common_metadata(torch, args.dtype, "torch_ref", shape, seq_lens, block_tables, cache, args.block_size),
    )
    results.append(_with_speedup(ref_result, ref_result.mean_ms))

    if args.mode in ("all", "triton"):
        _validate(torch, q, k_cache, v_cache, block_tables, seq_lens, args.block_size, args.num_warps)
        metadata = _common_metadata(torch, args.dtype, "triton", shape, seq_lens, block_tables, cache, args.block_size)
        metadata["num_warps"] = args.num_warps
        triton_result = benchmark_case(
            "triton_paged_decode_attention",
            lambda: paged_decode_attention(
                q,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                block_size=args.block_size,
                num_warps=args.num_warps,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata=metadata,
        )
        results.append(_with_speedup(triton_result, ref_result.mean_ms))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16"], default="float16")
    parser.add_argument(
        "--mode",
        choices=["all", "triton"],
        default="all",
        help="Which implementation to benchmark. Reference is always included as baseline.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Shape formatted as num_seqs,num_q_heads,num_kv_heads,head_dim,max_seq_len.",
    )
    parser.add_argument("--block-size", type=int, choices=[16], default=16)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", default="benchmarks/results/week6_paged_decode.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for paged decode benchmarks")

    dtype = _dtype_from_name(torch, args.dtype)
    shapes = args.shape if args.shape else DEFAULT_SHAPES
    results = []
    for shape in shapes:
        results.extend(run_shape(torch, args, shape, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
