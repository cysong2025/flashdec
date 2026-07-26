"""Run dense decode attention reference benchmarks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import benchmark_case, write_csv


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
    num_seqs, num_q_heads, num_kv_heads, _, _ = values
    if num_q_heads % num_kv_heads != 0:
        raise argparse.ArgumentTypeError("num_q_heads must be divisible by num_kv_heads")
    return values


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _common_metadata(torch, dtype_name, shape):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype_name,
        "op": "dense_decode_attention_ref",
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
    }


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


def run_shape(torch, args, shape, dtype):
    from flashdec.reference import dense_decode_attention_ref

    q, k_cache, v_cache, seq_lens = _make_inputs(torch, shape, dtype)
    metadata = _common_metadata(torch, args.dtype, shape)
    metadata["min_seq_len"] = int(seq_lens.min().item())
    metadata["max_actual_seq_len"] = int(seq_lens.max().item())
    return benchmark_case(
        "dense_decode_attention_ref",
        lambda: dense_decode_attention_ref(q, k_cache, v_cache, seq_lens),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=metadata,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Shape formatted as num_seqs,num_q_heads,num_kv_heads,head_dim,max_seq_len.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--output",
        default="benchmarks/results/dense_decode_reference.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for decode reference benchmarks")

    dtype = _dtype_from_name(torch, args.dtype)
    shapes = args.shape if args.shape else DEFAULT_SHAPES
    results = [run_shape(torch, args, shape, dtype) for shape in shapes]

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
