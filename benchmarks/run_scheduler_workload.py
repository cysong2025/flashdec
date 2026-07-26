"""Compare scheduler policies on identical finite request traces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import git_commit, percentile, summarize_latencies, write_csv
from flashdec.scheduled_workload import (
    CANCEL_ON_BACKPRESSURE,
    GREEDY_STEP_ONLY,
    LIFETIME_FIFO_AGING,
    RequestArrival,
    SchedulerWorkloadConfig,
    boundary_deadlock_arrivals,
    run_scheduler_workload,
)
from flashdec.scheduler import RequestSpec


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32
POLICIES = (
    CANCEL_ON_BACKPRESSURE,
    GREEDY_STEP_ONLY,
    LIFETIME_FIFO_AGING,
)


@dataclass(frozen=True)
class SchedulerCase:
    name: str
    arrivals: tuple[RequestArrival, ...]
    max_blocks: int
    max_active_requests: int
    max_batch_requests: int
    max_steps: int
    max_stalled_steps: int


CASES = {
    "boundary_deadlock": SchedulerCase(
        name="boundary_deadlock",
        arrivals=boundary_deadlock_arrivals(num_requests=2, max_new_tokens=64),
        max_blocks=2,
        max_active_requests=2,
        max_batch_requests=2,
        max_steps=140,
        max_stalled_steps=4,
    ),
    "finite_queue": SchedulerCase(
        name="finite_queue",
        arrivals=tuple(
            RequestArrival(
                RequestSpec(
                    request_id=request_id,
                    initial_context_tokens=(request_id % 3) * 8,
                    max_new_tokens=16,
                    submission_order=request_id,
                ),
                arrival_step=request_id // 2,
            )
            for request_id in range(6)
        ),
        max_blocks=4,
        max_active_requests=3,
        max_batch_requests=2,
        max_steps=100,
        max_stalled_steps=4,
    ),
}


def _dtype(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_engine(torch, case, dtype, append_backend, decode_backend):
    from flashdec.cache import PagedKVCache
    from flashdec.engine import DecodeEngine

    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=case.max_blocks,
        dtype=dtype,
        device="cuda",
    )
    return DecodeEngine(
        cache,
        append_backend=append_backend,
        decode_backend=decode_backend,
        num_warps=2,
    )


def _warmup_backend(torch, args, dtype):
    """Compile native/Triton paths on a disposable one-token Engine."""
    case = SchedulerCase(
        name="warmup",
        arrivals=(RequestArrival(RequestSpec("warmup", 0, 1, 0), 0),),
        max_blocks=1,
        max_active_requests=1,
        max_batch_requests=1,
        max_steps=2,
        max_stalled_steps=1,
    )
    engine = _make_engine(
        torch,
        case,
        dtype,
        args.append_backend,
        args.decode_backend,
    )
    run_scheduler_workload(
        engine,
        _config(case, LIFETIME_FIFO_AGING),
        num_q_heads=NUM_Q_HEADS,
        seed=args.seed,
    )
    torch.cuda.synchronize()


def _config(case, policy):
    return SchedulerWorkloadConfig(
        name=case.name,
        arrivals=case.arrivals,
        policy=policy,
        max_active_requests=case.max_active_requests,
        max_batch_requests=case.max_batch_requests,
        aging_threshold_steps=4,
        max_steps=case.max_steps,
        max_stalled_steps=case.max_stalled_steps,
    )


def _trial_policies(policies, trial_index):
    """Rotate policy order across trials without mutating the input."""
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    ordered = tuple(policies)
    if not ordered:
        raise ValueError("at least one policy is required")
    offset = trial_index % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _metadata(torch, args, case, policy, policy_order, result, dtype_name, commit, trial):
    cache = result.engine_metrics["cache"]
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "op": "scheduler_workload",
        "case": case.name,
        "policy": policy,
        "append_backend": args.append_backend,
        "decode_backend": args.decode_backend,
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": commit,
        "trial": trial,
        "policy_order": "->".join(policy_order),
        "seed": args.seed + trial - 1,
        "num_requests": len(case.arrivals),
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "max_blocks": case.max_blocks,
        "max_active_requests": case.max_active_requests,
        "max_batch_requests": case.max_batch_requests,
        "completed_requests": len(result.completed_request_ids),
        "cancelled_requests": len(result.cancelled_request_ids),
        "rejected_requests": len(result.rejected_request_ids),
        "completion_rate": f"{result.completion_rate:.6f}",
        "completed_tokens": result.completed_tokens,
        "useful_tokens": result.useful_tokens,
        "tokens_per_second": f"{result.tokens_per_second:.3f}",
        "useful_tokens_per_second": f"{result.useful_tokens_per_second:.3f}",
        "p99_ms": f"{result.p99_ms:.6f}",
        "successful_steps": result.successful_steps,
        "backpressure_steps": result.backpressure_steps,
        "stalled_steps": result.stalled_steps,
        "resource_deadlocks": result.resource_deadlock_count,
        "forced_cancellations": result.forced_cancellation_count,
        "mean_waiting_depth": f"{result.mean_waiting_depth:.3f}",
        "max_waiting_depth": result.max_waiting_depth,
        "admission_wait_p50": result.admission_wait_p50,
        "admission_wait_p90": result.admission_wait_p90,
        "max_service_wait_steps": result.max_service_wait_steps,
        "scheduler_p50_ms": f"{percentile(result.scheduler_decision_ms, 50):.6f}",
        "max_committed_blocks": max(result.committed_block_samples, default=0),
        "max_physical_blocks": max(result.physical_block_samples, default=0),
        "allocations": cache["allocation_count"],
        "frees": cache["free_count"],
        "reuses": cache["reuse_count"],
        "validated_invariants": True,
        "timing_scope": (
            "scheduler decision wall time + Engine.step wall time; excludes "
            "Q/K/V generation, prompt-context writes, and JIT build"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--append-backend", choices=["torch", "fused_cuda"], default="fused_cuda")
    parser.add_argument("--decode-backend", choices=["reference", "triton"], default="triton")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--output", default="benchmarks/results/scheduler_workload.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("trials must be positive")
    if len(set(args.policy)) != len(args.policy):
        raise SystemExit("policy values must be unique")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for scheduler workload benchmarks")
    if args.append_backend == "fused_cuda":
        from flashdec import load_fused_rope_kv_append_extension

        load_fused_rope_kv_append_extension()

    dtype_names = ("float16", "bfloat16") if args.dtype == "both" else (args.dtype,)
    cases = list(CASES.values()) if args.case == "all" else [CASES[args.case]]
    commit = git_commit(PROJECT_ROOT)
    results = []
    for dtype_name in dtype_names:
        dtype = _dtype(torch, dtype_name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("this CUDA device does not support BF16")
        _warmup_backend(torch, args, dtype)
        for case in cases:
            for trial_index in range(args.trials):
                trial = trial_index + 1
                policy_order = _trial_policies(args.policy, trial_index)
                for policy in policy_order:
                    engine = _make_engine(
                        torch,
                        case,
                        dtype,
                        args.append_backend,
                        args.decode_backend,
                    )
                    result = run_scheduler_workload(
                        engine,
                        _config(case, policy),
                        num_q_heads=NUM_Q_HEADS,
                        seed=args.seed + trial_index,
                    )
                    benchmark = summarize_latencies(
                        name="scheduler_workload",
                        latencies_ms=result.step_latencies_ms,
                        metadata=_metadata(
                            torch,
                            args,
                            case,
                            policy,
                            policy_order,
                            result,
                            dtype_name,
                            commit,
                            trial,
                        ),
                    )
                    results.append(benchmark)
                    print(benchmark.as_row())
    write_csv(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
