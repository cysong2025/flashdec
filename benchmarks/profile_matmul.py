"""Profile a matmul case with the PyTorch profiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument(
        "--impl",
        choices=["torch", "fixed", "autotuned"],
        default="fixed",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--output", default="benchmarks/profiles/matmul_profile.txt")
    return parser.parse_args()


def _select_fn(torch, impl):
    if impl == "torch":
        return torch.matmul
    if impl == "fixed":
        from flashdec.kernels.matmul import matmul

        return matmul
    if impl == "autotuned":
        from flashdec.kernels.matmul import matmul_autotuned

        return matmul_autotuned
    raise ValueError(f"unsupported impl: {impl}")


def main():
    args = parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for matmul profiling")

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    fn = _select_fn(torch, args.impl)

    for _ in range(args.warmup):
        fn(a, b)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(args.repeat):
            fn(a, b)
        torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(table)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
