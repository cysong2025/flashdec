"""Run Triton matmul benchmarks."""

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
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (1024, 1024, 256),
    (4096, 1024, 1024),
]


def _parse_shape(text):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be formatted as M,N,K")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if m <= 0 or n <= 0 or k <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return m, n, k


def _common_metadata(torch, dtype_name, impl, m, n, k):
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": dtype_name,
        "impl": impl,
        "op": "matmul",
        "M": m,
        "N": n,
        "K": k,
    }


def _with_speedup(result, torch_mean_ms):
    metadata = dict(result.metadata)
    metadata["speedup_vs_torch"] = f"{torch_mean_ms / result.mean_ms:.4f}"
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


def _make_inputs(torch, m, n, k):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    return a, b


def _validate(torch, a, b, fn):
    actual = fn(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def run_shape(torch, args, shape):
    from flashdec.kernels.matmul import matmul, matmul_autotuned

    m, n, k = shape
    a, b = _make_inputs(torch, m, n, k)
    results = []

    torch_result = benchmark_case(
        "torch_matmul",
        lambda: torch.matmul(a, b),
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=_common_metadata(torch, args.dtype, "torch", m, n, k),
    )
    results.append(_with_speedup(torch_result, torch_result.mean_ms))

    if args.mode in ("all", "fixed"):
        _validate(torch, a, b, matmul)
        fixed_result = benchmark_case(
            "triton_matmul_fixed",
            lambda: matmul(
                a,
                b,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                num_warps=args.num_warps,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata=_common_metadata(torch, args.dtype, "triton_fixed", m, n, k),
        )
        results.append(_with_speedup(fixed_result, torch_result.mean_ms))

    if args.mode in ("all", "autotuned"):
        _validate(torch, a, b, matmul_autotuned)
        autotuned_result = benchmark_case(
            "triton_matmul_autotuned",
            lambda: matmul_autotuned(a, b),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata=_common_metadata(torch, args.dtype, "triton_autotuned", m, n, k),
        )
        results.append(_with_speedup(autotuned_result, torch_result.mean_ms))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16"], default="float16")
    parser.add_argument(
        "--mode",
        choices=["all", "fixed", "autotuned"],
        default="all",
        help="Which Triton implementation to compare against torch.matmul.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Shape formatted as M,N,K. Repeat to benchmark multiple shapes.",
    )
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/results/matmul_benchmark.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for matmul benchmarks")

    shapes = args.shape if args.shape else DEFAULT_SHAPES
    results = []
    for shape in shapes:
        results.extend(run_shape(torch, args, shape))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
