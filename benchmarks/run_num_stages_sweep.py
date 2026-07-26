"""Run a bounded Triton num_stages sweep for representative paged decode shapes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import write_csv
from benchmarks.run_paged_decode_warp_sweep import _dtype_from_name, run_shape


CASES = {
    "medium": ("medium_b16_ctx1024", "num_stages", (16, 32, 8, 128, 1024)),
    "large": ("large_b16_ctx8192", "num_stages", (16, 32, 8, 128, 8192)),
    "large_batch": ("large_batch_b64_ctx4096", "num_stages", (64, 32, 8, 128, 4096)),
}


def _parse_num_stages(value):
    if value == "default":
        return None
    try:
        num_stages = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("num_stages must be default or an integer from 1 to 4") from exc
    if num_stages not in (1, 2, 3, 4):
        raise argparse.ArgumentTypeError("num_stages must be default or an integer from 1 to 4")
    return num_stages


def _selected_cases(names):
    return [CASES[name] for name in names]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument(
        "--num-stages",
        type=_parse_num_stages,
        nargs="+",
        default=[None, 1, 2, 3, 4],
        metavar="STAGE",
        help="Triton stages to test: default, 1, 2, 3, or 4.",
    )
    parser.add_argument("--kv-layout", choices=["token_major", "dim_major"], default="token_major")
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=32)
    parser.add_argument("--num-warps", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/paged_decode_staging_sweep.csv",
    )
    return parser.parse_args()


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

    args.num_stages_values = args.num_stages
    args.num_stages = None
    args.mode = "triton"
    args.experiment = "num_stages"
    args.benchmark = "paged_decode_staging_sweep"
    args.num_warps = [args.num_warps]
    results = []
    cases = _selected_cases(args.cases)
    for dtype in dtypes:
        for case_index, (case_name, sweep, shape) in enumerate(cases):
            results.extend(run_shape(torch, args, case_index, case_name, sweep, shape, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
