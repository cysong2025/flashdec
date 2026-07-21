"""Run the R4-C integrated scheduled multi-layer workload on CUDA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, git_commit, percentile, write_csv
from flashdec.integrated_workload import (
    run_integrated_workload,
    standard_integrated_config,
)
from flashdec.perf import dtype_nbytes


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32
NUM_WARPS = 2


@dataclass(frozen=True)
class IntegratedCase:
    name: str
    num_layers: int
    context_tokens: int

    @property
    def prefix_blocks(self):
        return self.context_tokens // BLOCK_SIZE

    @property
    def max_blocks(self):
        return 2 * self.prefix_blocks + 4


CASES = {
    f"l{layers}_c{context}": IntegratedCase(
        name=f"l{layers}_c{context}",
        num_layers=layers,
        context_tokens=context,
    )
    for layers in (2, 4)
    for context in (64, 128)
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
            raise RuntimeError("bfloat16 requested but this CUDA device does not support BF16")
        result.append((name, dtype))
    return result


def _trial_case_order(cases, trial_index):
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    cases = list(cases)
    offset = trial_index % len(cases)
    return cases[offset:] + cases[:offset]


def _quick_case(case):
    context_tokens = 32 if case.context_tokens == 64 else 64
    return IntegratedCase(
        name=f"l{case.num_layers}_c{context_tokens}",
        num_layers=case.num_layers,
        context_tokens=context_tokens,
    )


def _selected_cases(name, quick):
    cases = [CASES[name]] if name != "all" else list(CASES.values())
    return [_quick_case(case) for case in cases] if quick else cases


def _make_engine(torch, case, dtype, seed):
    from flashdec.cache import PagedKVCache
    from flashdec.engine import DecodeEngine

    cache = PagedKVCache(
        num_layers=case.num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=case.max_blocks,
        dtype=dtype,
        device="cuda",
        prefix_cache_capacity_blocks=case.prefix_blocks,
    )
    engine = DecodeEngine(
        cache,
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=NUM_WARPS,
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    shape = (
        case.num_layers,
        case.prefix_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
    )
    prefix_k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    prefix_v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    engine.register_prefix("shared", prefix_k, prefix_v)
    torch.cuda.synchronize()
    return engine


def _run_once(torch, case, dtype, seed):
    engine = _make_engine(torch, case, dtype, seed + 100_000)
    config = standard_integrated_config(
        num_layers=case.num_layers,
        context_tokens=case.context_tokens,
    )
    return run_integrated_workload(
        engine,
        config,
        num_q_heads=NUM_Q_HEADS,
        seed=seed,
    )


def _result_row(torch, args, case, dtype_name, dtype, trial, seed, order, result, commit):
    metrics = result.engine_metrics
    cache = metrics["cache"]
    reference = result.reference
    bytes_per_block = (
        case.num_layers
        * 2
        * NUM_KV_HEADS
        * BLOCK_SIZE
        * HEAD_DIM
        * dtype_nbytes(dtype)
    )
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "op": "integrated_scheduled_multi_layer",
        "case": case.name,
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": commit,
        "append_backend": "fused_cuda",
        "decode_backend": "triton",
        "metadata_policy": "materialized",
        "num_layers": case.num_layers,
        "context_tokens": case.context_tokens,
        "prefix_blocks": case.prefix_blocks,
        "max_blocks": case.max_blocks,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "trial": trial,
        "trial_count": args.trials,
        "case_order": "->".join(item.name for item in order),
        "seed": seed,
        "reference_steps": len(reference.steps),
        "trajectory_digest": result.trajectory_digest,
        "reference_trajectory_digest": reference.digest,
        "trajectory_validated": result.trajectory_digest == reference.digest,
        "completed_request_ids": "|".join(result.completed_request_ids),
        "cancelled_request_ids": "|".join(result.cancelled_request_ids),
        "rejected_request_ids": "|".join(result.rejected_request_ids),
        "successful_steps": result.successful_steps,
        "aborted_steps": result.aborted_steps,
        "completed_tokens": result.completed_tokens,
        "block_reuse_count": result.block_reuse_count,
        "peak_used_blocks": max(step.used_blocks for step in reference.steps),
        "terminal_resident_prefix_blocks": result.terminal_resident_prefix_blocks,
        "final_free_blocks": result.final_free_blocks,
        "bytes_per_block": bytes_per_block,
        "peak_allocated_kv_bytes": (
            max(step.used_blocks for step in reference.steps) * bytes_per_block
        ),
        "scheduler_p50_ms": f"{percentile(result.scheduler_ms, 50):.6f}",
        "context_seed_p50_ms": f"{percentile(result.context_seed_ms, 50):.6f}",
        "engine_p50_ms": f"{percentile(result.engine_ms, 50):.6f}",
        "complete_step_p99_ms": f"{result.p99_ms:.6f}",
        "decode_tokens_per_second": f"{result.tokens_per_second:.3f}",
        "transaction_begin_count": cache["transaction_begin_count"],
        "transaction_commit_count": cache["transaction_commit_count"],
        "transaction_abort_count": cache["transaction_abort_count"],
        "transaction_layer_write_count": cache["transaction_layer_write_count"],
        "transaction_rollback_block_count": cache["transaction_rollback_block_count"],
        "engine_transaction_layer_step_count": metrics["transaction_layer_step_count"],
        "engine_transaction_abort_count": metrics["transaction_abort_count"],
        "prefix_registration_count": cache["prefix_registration_count"],
        "prefix_hit_count": cache["prefix_hit_count"],
        "prefix_eviction_count": cache["prefix_eviction_count"],
        "final_open_transaction_count": cache["open_transaction_count"],
        "final_used_blocks": cache["used_blocks"],
        "validated_invariants": True,
        "timing_scope": (
            "dynamic arrival lifecycle wall including scheduler, private multi-layer "
            "context writes, fused transaction/decode, and finish/cancel; random input "
            "generation, prefix registration, and terminal prefix eviction excluded"
        ),
    }
    return BenchmarkResult(
        name="integrated_scheduled_multi_layer",
        mean_ms=result.mean_ms,
        p50_ms=result.p50_ms,
        p90_ms=result.p90_ms,
        min_ms=min(result.step_latencies_ms),
        max_ms=max(result.step_latencies_ms),
        repeats=len(result.step_latencies_ms),
        metadata=metadata,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/r4_integrated_scheduled_multi_layer.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials <= 0 or args.warmup_runs < 0:
        raise SystemExit("trials must be positive and warmup-runs non-negative")
    if args.quick:
        args.trials = 1
        args.warmup_runs = min(args.warmup_runs, 1)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the R4-C workload")
    from flashdec import load_fused_rope_kv_append_extension

    load_fused_rope_kv_append_extension()
    cases = _selected_cases(args.case, args.quick)
    commit = git_commit(PROJECT_ROOT)
    results = []
    for dtype_name, dtype in _requested_dtypes(torch, args.dtype):
        for trial_index in range(args.trials):
            trial = trial_index + 1
            seed = args.seed + trial_index
            order = _trial_case_order(cases, trial_index)
            for case in order:
                for warmup_index in range(args.warmup_runs):
                    _run_once(torch, case, dtype, seed + 500_000 + warmup_index)
                result = _run_once(torch, case, dtype, seed)
                row = _result_row(
                    torch,
                    args,
                    case,
                    dtype_name,
                    dtype,
                    trial,
                    seed,
                    order,
                    result,
                    commit,
                )
                results.append(row)
                print(row.as_row())
    write_csv(results, args.output)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
