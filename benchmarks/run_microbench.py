"""Run Triton microbenchmarks for foundational operators."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import benchmark_case, write_csv


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _common_metadata(torch, dtype_name):
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype_name,
    }


def run_vector_add(torch, args, dtype):
    from flashdec.kernels.vector_add import vector_add

    x = torch.randn(args.size, device="cuda", dtype=dtype)
    y = torch.randn(args.size, device="cuda", dtype=dtype)
    metadata = _common_metadata(torch, args.dtype)
    metadata.update({"op": "vector_add", "size": args.size})
    return benchmark_case(
        "vector_add",
        lambda: vector_add(x, y, block_size=args.block_size),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=metadata,
    )


def run_softmax(torch, args, dtype):
    from flashdec.kernels.softmax import row_softmax

    x = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype)
    metadata = _common_metadata(torch, args.dtype)
    metadata.update({"op": "row_softmax", "rows": args.rows, "cols": args.cols})
    return benchmark_case(
        "row_softmax",
        lambda: row_softmax(x),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=metadata,
    )


def run_rmsnorm(torch, args, dtype):
    from flashdec.kernels.rmsnorm import rmsnorm

    x = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype)
    weight = torch.randn(args.cols, device="cuda", dtype=dtype)
    metadata = _common_metadata(torch, args.dtype)
    metadata.update({"op": "rmsnorm", "rows": args.rows, "cols": args.cols})
    return benchmark_case(
        "rmsnorm",
        lambda: rmsnorm(x, weight),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=metadata,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        choices=["all", "vector_add", "softmax", "rmsnorm"],
        default="all",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/results/microbench.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for microbenchmarks")

    dtype = _dtype_from_name(torch, args.dtype)
    results = []
    if args.op in ("all", "vector_add"):
        results.append(run_vector_add(torch, args, dtype))
    if args.op in ("all", "softmax"):
        results.append(run_softmax(torch, args, dtype))
    if args.op in ("all", "rmsnorm"):
        results.append(run_rmsnorm(torch, args, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
