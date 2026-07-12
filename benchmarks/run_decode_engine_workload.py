"""Benchmark dynamic DecodeEngine workloads on the frozen CUDA decode path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, summarize_latencies, write_csv
from flashdec.workload import WorkloadConfig, run_synthetic_workload


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32


@dataclass(frozen=True)
class WorkloadSpec:
    config: WorkloadConfig
    max_blocks: int


WORKLOADS = {
    "short_churn": WorkloadSpec(
        config=WorkloadConfig(
            name="short_churn",
            steps=120,
            max_active=8,
            arrivals_per_step=4,
            decode_tokens_per_request=4,
            initial_context_tokens=8,
            cancel_interval=5,
        ),
        max_blocks=16,
    ),
    "mixed_steady": WorkloadSpec(
        config=WorkloadConfig(
            name="mixed_steady",
            steps=160,
            max_active=16,
            arrivals_per_step=2,
            decode_tokens_per_request=32,
            initial_context_tokens=16,
            context_stagger_tokens=4,
        ),
        max_blocks=96,
    ),
    "long_pressure": WorkloadSpec(
        config=WorkloadConfig(
            name="long_pressure",
            steps=112,
            max_active=16,
            arrivals_per_step=16,
            decode_tokens_per_request=128,
            cancel_on_backpressure=True,
        ),
        # One physical block per active request admits the initial batch. At
        # the token-32 boundary all 16 rows need a second block at once, so
        # the engine must expose backpressure and shed work before retrying.
        max_blocks=16,
    ),
}


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _requested_dtypes(torch, dtype_name):
    names = ("float16", "bfloat16") if dtype_name == "both" else (dtype_name,)
    result = []
    for name in names:
        dtype = _dtype_from_name(torch, name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 workload requested but this CUDA device does not support BF16")
        result.append((name, dtype))
    return result


def _make_engine(torch, dtype, max_blocks, append_backend, decode_backend, num_warps):
    from flashdec.cache import PagedKVCache
    from flashdec.engine import DecodeEngine

    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )
    return DecodeEngine(
        cache,
        append_backend=append_backend,
        decode_backend=decode_backend,
        num_warps=num_warps,
    )


def _preload_native_backends(append_backends):
    if "fused_cuda" in append_backends:
        from flashdec import load_fused_rope_kv_append_extension

        load_fused_rope_kv_append_extension()
    if "cuda" in append_backends:
        from flashdec import load_cuda_kv_append_extension

        load_cuda_kv_append_extension()


def _metadata(
    torch,
    result,
    args,
    dtype_name,
    spec,
    append_backend,
    trial,
    trial_seed,
    backend_order,
):
    metrics = result.engine_metrics
    cache = metrics["cache"]
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "op": "decode_engine_workload",
        "workload": spec.config.name,
        "append_backend": append_backend,
        "decode_backend": args.decode_backend,
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": args.num_warps,
        "steps": spec.config.steps,
        "warmup_steps": args.warmup_steps,
        "max_active": spec.config.max_active,
        "arrivals_per_step": spec.config.arrivals_per_step,
        "decode_tokens_per_request": spec.config.decode_tokens_per_request,
        "initial_context_tokens": spec.config.initial_context_tokens,
        "context_stagger_tokens": spec.config.context_stagger_tokens,
        "cancel_interval": spec.config.cancel_interval,
        "cancel_probability": spec.config.cancel_probability,
        "max_blocks": spec.max_blocks,
        "trial": trial,
        "trial_count": args.trials,
        "backend_order": "->".join(backend_order),
        "p99_ms": f"{result.p99_ms:.6f}",
        "tokens_per_second": f"{result.tokens_per_second:.3f}",
        "mean_active_batch": f"{result.mean_active_batch:.3f}",
        "max_active_batch": result.max_active_batch,
        "successful_steps": result.successful_steps,
        "completed_tokens": result.completed_tokens,
        "admitted_requests": result.admitted_requests,
        "finished_requests": result.finished_requests,
        "cancelled_requests": result.cancelled_requests,
        "prefilled_tokens": result.prefilled_tokens,
        "backpressure_steps": result.backpressure_steps,
        "final_active_requests": metrics["active_requests"],
        "final_used_blocks": cache["used_blocks"],
        "final_free_blocks": cache["free_blocks"],
        "final_block_utilization": f"{cache['block_utilization']:.6f}",
        "final_internal_fragmentation_tokens": cache["internal_fragmentation_tokens"],
        "allocations": cache["allocation_count"],
        "frees": cache["free_count"],
        "reuses": cache["reuse_count"],
        "engine_backpressure_count": metrics["backpressure_count"],
        "validated_invariants": True,
        "timing_scope": (
            "wall-clock submission/admission + Engine.step + finish/cancel; "
            "excludes Q/K/V generation, prompt prefill, warmup, and JIT build"
        ),
        "seed": trial_seed,
    }


def _with_speedup(result, torch_p50_ms):
    metadata = dict(result.metadata)
    metadata["speedup_vs_torch_p50"] = f"{torch_p50_ms / result.p50_ms:.4f}"
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


def _quick_spec(spec):
    steps = {
        "short_churn": 24,
        "mixed_steady": 32,
        # This crosses the common token-32 allocation boundary and recovery.
        "long_pressure": 72,
    }[spec.config.name]
    return replace(spec, config=replace(spec.config, steps=steps))


def _trial_append_backends(append_backends, trial_index):
    """Alternate execution order to reduce fixed backend-order bias."""
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    ordered = list(append_backends)
    if trial_index % 2:
        ordered.reverse()
    return ordered


def _run_workload(
    torch,
    args,
    dtype_name,
    dtype,
    spec,
    append_backend,
    trial,
    trial_seed,
    backend_order,
):
    engine = _make_engine(
        torch,
        dtype,
        spec.max_blocks,
        append_backend,
        args.decode_backend,
        args.num_warps,
    )
    result = run_synthetic_workload(
        engine,
        spec.config,
        num_q_heads=NUM_Q_HEADS,
        warmup_steps=args.warmup_steps,
        seed=trial_seed,
    )
    return summarize_latencies(
        name="decode_engine_workload",
        latencies_ms=result.latencies_ms,
        metadata=_metadata(
            torch,
            result,
            args,
            dtype_name,
            spec,
            append_backend,
            trial,
            trial_seed,
            backend_order,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["all", *WORKLOADS], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument(
        "--append-backends",
        nargs="+",
        choices=["torch", "cuda", "fused_cuda"],
        default=["torch", "fused_cuda"],
        help="Append paths to compare; torch is the complete-step baseline.",
    )
    parser.add_argument("--decode-backend", choices=["triton"], default="triton")
    parser.add_argument("--num-warps", type=int, choices=[2], default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1, help="Repeat each paired workload; adjacent trials reverse backend order.")
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--quick", action="store_true", help="Shorten every workload while preserving pressure boundaries.")
    parser.add_argument("--output", default="benchmarks/results/week12_decode_engine_workload.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup_steps < 0 or args.trials <= 0:
        raise SystemExit("warmup-steps must be non-negative and trials must be positive")
    if len(set(args.append_backends)) != len(args.append_backends):
        raise SystemExit("append-backends must be unique")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DecodeEngine workload benchmarks")
    _preload_native_backends(args.append_backends)
    specs = [WORKLOADS[args.workload]] if args.workload != "all" else list(WORKLOADS.values())
    if args.quick:
        specs = [_quick_spec(spec) for spec in specs]

    results = []
    for dtype_name, dtype in _requested_dtypes(torch, args.dtype):
        for spec in specs:
            for trial_index in range(args.trials):
                trial = trial_index + 1
                trial_seed = args.seed + trial_index
                backend_order = _trial_append_backends(args.append_backends, trial_index)
                per_backend = [
                    _run_workload(
                        torch,
                        args,
                        dtype_name,
                        dtype,
                        spec,
                        append_backend,
                        trial,
                        trial_seed,
                        backend_order,
                    )
                    for append_backend in backend_order
                ]
                torch_result = next(
                    (item for item in per_backend if item.metadata["append_backend"] == "torch"),
                    None,
                )
                if torch_result is None:
                    results.extend(per_backend)
                else:
                    results.extend(
                        _with_speedup(item, torch_result.p50_ms) for item in per_backend
                    )

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
