"""Compare token-major and dim-major paged KV cache layouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import write_csv

from benchmarks.run_paged_decode_warp_sweep import (
    _dtype_from_name,
    _make_shape_matrix,
    run_shape,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--kv-layouts", nargs="+", choices=["token_major", "dim_major"], default=["token_major", "dim_major"])
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=32)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--batch-context", type=int, default=1024)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--num-warps", type=int, nargs="+", default=[2])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/paged_decode_kv_layout_sweep.csv",
    )
    args = parser.parse_args()
    if args.num_q_heads <= 0 or args.num_kv_heads <= 0:
        parser.error("num_q_heads and num_kv_heads must be positive")
    if args.num_q_heads % args.num_kv_heads != 0:
        parser.error("num_q_heads must be divisible by num_kv_heads")
    if any(value not in (1, 2, 4, 8) for value in args.num_warps):
        parser.error("num_warps values must be one of 1, 2, 4, or 8")
    return args


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for paged decode benchmarks")

    if args.dtype == "both":
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        else:
            print("Skipping bfloat16 because torch.cuda.is_bf16_supported() is false.")
    else:
        dtypes = [_dtype_from_name(torch, args.dtype)]
        if dtypes[0] == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("bfloat16 was requested, but this CUDA device does not report BF16 support")

    args.mode = "triton"
    args.experiment = "kv_layout"
    cases = _make_shape_matrix(args)
    results = []
    for dtype in dtypes:
        for kv_layout in args.kv_layouts:
            args.kv_layout = kv_layout
            for case_index, (case_name, sweep, shape) in enumerate(cases):
                results.extend(run_shape(torch, args, case_index, case_name, sweep, shape, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
