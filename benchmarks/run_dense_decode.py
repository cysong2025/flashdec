"""Run Week 4 dense decode attention Triton benchmarks."""

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
    (16, 16, 4, 128, 1024),
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
    if head_dim not in (64, 128):
        raise argparse.ArgumentTypeError("head_dim must be 64 or 128")
    return values


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _common_metadata(torch, dtype_name, impl, shape, seq_lens):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype_name,
        "impl": impl,
        "op": "dense_decode_attention",
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "min_seq_len": int(seq_lens.min().item()),
        "max_actual_seq_len": int(seq_lens.max().item()),
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


def _make_inputs(torch, shape, dtype):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    q = torch.randn((num_seqs, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.randn(
        (num_seqs, max_seq_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
    )
    low = max(1, max_seq_len // 2)
    seq_lens = torch.randint(low, max_seq_len + 1, (num_seqs,), device="cuda")
    return q, k_cache, v_cache, seq_lens


def _validate(torch, q, k_cache, v_cache, seq_lens, block_seq, num_warps):
    from flashdec.kernels.dense_decode import dense_decode_attention
    from flashdec.reference import dense_decode_attention_ref

    actual = dense_decode_attention(
        q,
        k_cache,
        v_cache,
        seq_lens,
        block_seq=block_seq,
        num_warps=num_warps,
    )
    expected = dense_decode_attention_ref(q, k_cache, v_cache, seq_lens)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def run_shape(torch, args, shape, dtype):
    from flashdec.kernels.dense_decode import dense_decode_attention
    from flashdec.reference import dense_decode_attention_ref

    q, k_cache, v_cache, seq_lens = _make_inputs(torch, shape, dtype)
    results = []

    ref_result = benchmark_case(
        "dense_decode_attention_ref",
        lambda: dense_decode_attention_ref(q, k_cache, v_cache, seq_lens),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=_common_metadata(torch, args.dtype, "torch_ref", shape, seq_lens),
    )
    results.append(_with_speedup(ref_result, ref_result.mean_ms))

    if args.mode in ("all", "triton"):
        _validate(torch, q, k_cache, v_cache, seq_lens, args.block_seq, args.num_warps)
        metadata = _common_metadata(torch, args.dtype, "triton", shape, seq_lens)
        metadata["block_seq"] = args.block_seq
        metadata["num_warps"] = args.num_warps
        triton_result = benchmark_case(
            "triton_dense_decode_attention",
            lambda: dense_decode_attention(
                q,
                k_cache,
                v_cache,
                seq_lens,
                block_seq=args.block_seq,
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
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
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
    parser.add_argument("--block-seq", type=int, choices=[16, 32, 64, 128], default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", default="benchmarks/results/week4_dense_decode.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for dense decode benchmarks")

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
