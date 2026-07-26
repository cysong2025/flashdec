"""Benchmark complete multi-layer DecodeEngine token transactions on CUDA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import math
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, git_commit, percentile, write_csv
from flashdec.engine import PROFILE_RANGE_APPEND, PROFILE_RANGE_DECODE
from flashdec.perf import dtype_nbytes


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32
NUM_WARPS = 2


@dataclass(frozen=True)
class MultiLayerCase:
    name: str
    num_layers: int
    batch_size: int
    context_tokens: int


CASES = {
    f"l{layers}_b{batch}_c{context}": MultiLayerCase(
        name=f"l{layers}_b{batch}_c{context}",
        num_layers=layers,
        batch_size=batch,
        context_tokens=context,
    )
    for layers in (1, 2, 4)
    for batch in (4, 16)
    for context in (128, 1024)
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


def _trial_backend_order(backends, trial_index):
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    result = list(backends)
    if trial_index % 2:
        result.reverse()
    return result


def _quick_case(case):
    context_tokens = 32 if case.context_tokens == 128 else 64
    return replace(
        case,
        name=f"l{case.num_layers}_b{case.batch_size}_c{context_tokens}",
        context_tokens=context_tokens,
    )


def _selected_cases(name, quick):
    cases = [CASES[name]] if name != "all" else list(CASES.values())
    return [_quick_case(case) for case in cases] if quick else cases


def _max_blocks(case, measured_steps):
    blocks_per_request = math.ceil(
        (case.context_tokens + int(measured_steps) + 1) / BLOCK_SIZE
    )
    return case.batch_size * blocks_per_request


def _make_engine(torch, case, dtype, backend, measured_steps, *, profiled=False):
    from flashdec.cache import PagedKVCache
    from flashdec.engine import DecodeEngine

    cache = PagedKVCache(
        num_layers=case.num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=_max_blocks(case, measured_steps),
        dtype=dtype,
        device="cuda",
    )
    engine = DecodeEngine(
        cache,
        append_backend=backend,
        decode_backend="triton",
        num_warps=NUM_WARPS,
        profile_ranges=profiled,
    )
    request_ids = tuple(range(case.batch_size))
    for request_id in request_ids:
        engine.add_request(request_id)
    engine.admit()
    return engine, request_ids


def _seed_context(torch, engine, request_ids, token_count):
    """Seed committed multi-layer KV state outside every timing boundary."""
    if token_count <= 0:
        return
    k = torch.zeros(
        (len(request_ids), NUM_KV_HEADS, HEAD_DIM),
        device="cuda",
        dtype=engine.cache.dtype,
    )
    v = torch.zeros_like(k)
    for _ in range(token_count):
        transaction = engine.cache.begin_token(request_ids)
        for layer_idx in range(engine.cache.num_layers):
            engine.cache.write_token_layer(transaction, layer_idx, k, v)
        engine.cache.commit_token(transaction)
    torch.cuda.synchronize()
    if not engine.validate_invariants():
        raise RuntimeError("context seeding failed invariant validation")


def _generate_inputs(torch, case, dtype, steps, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    result = []
    for _ in range(steps):
        layers = []
        for _layer_idx in range(case.num_layers):
            q = torch.randn(
                (case.batch_size, NUM_Q_HEADS, HEAD_DIM),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            k = torch.randn(
                (case.batch_size, NUM_KV_HEADS, HEAD_DIM),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            v = torch.randn(
                (case.batch_size, NUM_KV_HEADS, HEAD_DIM),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            layers.append((q, k, v))
        result.append(tuple(layers))
    torch.cuda.synchronize()
    return tuple(result)


def _run_tokens(torch, engine, request_ids, inputs):
    wall_ms = []
    device_ms = []
    layer_device_ms = []
    layer_total_device_ms = []
    begin_host_ms = []
    commit_host_ms = []

    for token_inputs in inputs:
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        layer_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in token_inputs
        ]
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        total_start.record()

        host_start = time.perf_counter()
        transaction = engine.begin_step(request_ids)
        begin_host_ms.append((time.perf_counter() - host_start) * 1_000.0)

        for layer_idx, ((q, k, v), (layer_start, layer_end)) in enumerate(
            zip(token_inputs, layer_events)
        ):
            layer_start.record()
            engine.step_layer(transaction, layer_idx, q, k, v)
            layer_end.record()

        host_start = time.perf_counter()
        engine.commit_step(transaction)
        commit_host_ms.append((time.perf_counter() - host_start) * 1_000.0)
        total_end.record()
        torch.cuda.synchronize()

        wall_ms.append((time.perf_counter() - wall_start) * 1_000.0)
        device_ms.append(total_start.elapsed_time(total_end))
        token_layer_times = [
            start.elapsed_time(end) for start, end in layer_events
        ]
        layer_device_ms.extend(token_layer_times)
        layer_total_device_ms.append(sum(token_layer_times))

    return {
        "wall_ms": wall_ms,
        "device_ms": device_ms,
        "layer_device_ms": layer_device_ms,
        "layer_total_device_ms": layer_total_device_ms,
        "begin_host_ms": begin_host_ms,
        "commit_host_ms": commit_host_ms,
    }


def _event_time_us(event, primary, fallback):
    value = getattr(event, primary, None)
    if value is None:
        value = getattr(event, fallback, 0.0)
    return float(value or 0.0)


def _profiler_kwargs(torch):
    """Keep explicit ranges across profiler cycles triggered by GPU tooling."""
    return {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "acc_events": True,
    }


def _profile_probe(torch, case, dtype, backend, steps, seed):
    if steps <= 0:
        return {
            "profile_steps": 0,
            "profile_token_count": 0,
            "profile_append_count": 0,
            "profile_decode_count": 0,
            "profile_cuda_event_count": 0,
            "profile_append_device_ms_per_layer": 0.0,
            "profile_decode_device_ms_per_layer": 0.0,
        }
    engine, request_ids = _make_engine(
        torch, case, dtype, backend, steps, profiled=True
    )
    _seed_context(torch, engine, request_ids, case.context_tokens)
    inputs = _generate_inputs(torch, case, dtype, steps, seed)
    # PyTorch/Triton may advance profiler cycles while compiling or launching
    # kernels. Keep GPU stage events across those cycles. Pure CPU begin/commit
    # costs come from the separate non-instrumented host timers above.
    completed_tokens = 0
    with torch.profiler.profile(**_profiler_kwargs(torch)) as profiler:
        for token_inputs in inputs:
            transaction = engine.begin_step(request_ids)
            for layer_idx, (q, k, v) in enumerate(token_inputs):
                engine.step_layer(transaction, layer_idx, q, k, v)
            engine.commit_step(transaction)
            completed_tokens += 1
        torch.cuda.synchronize()

    key_averages = profiler.key_averages()
    events = {event.key: event for event in key_averages}
    append = events.get(PROFILE_RANGE_APPEND)
    decode = events.get(PROFILE_RANGE_DECODE)
    expected_layers = steps * case.num_layers
    counts = {
        "profile_token_count": completed_tokens,
        "profile_append_count": int(getattr(append, "count", 0)),
        "profile_decode_count": int(getattr(decode, "count", 0)),
    }
    if counts != {
        "profile_token_count": steps,
        "profile_append_count": expected_layers,
        "profile_decode_count": expected_layers,
    }:
        raise RuntimeError(f"multi-layer profiler range counts are invalid: {counts}")
    cuda_event_count = sum(
        int(event.count)
        for event in key_averages
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    )
    append_device_ms = _event_time_us(
        append, "device_time_total", "cuda_time_total"
    ) / 1_000.0
    decode_device_ms = _event_time_us(
        decode, "device_time_total", "cuda_time_total"
    ) / 1_000.0
    return {
        "profile_steps": steps,
        **counts,
        "profile_cuda_event_count": cuda_event_count,
        "profile_append_device_ms_per_layer": append_device_ms / expected_layers,
        "profile_decode_device_ms_per_layer": decode_device_ms / expected_layers,
    }


def _rollback_probe(torch, case, dtype, backend, repeats, seed):
    if case.num_layers < 2 or repeats <= 0:
        return {
            "rollback_repeats": 0,
            "rollback_p50_ms": 0.0,
            "rollback_blocks": 0,
            "rollback_validated": True,
        }
    probe_case = replace(case, context_tokens=0)
    engine, request_ids = _make_engine(
        torch, probe_case, dtype, backend, 1, profiled=False
    )
    inputs = _generate_inputs(torch, probe_case, dtype, repeats, seed)
    latencies = []
    for token_inputs in inputs:
        q0, k0, v0 = token_inputs[0]
        bad_q = q0[..., :-1]
        torch.cuda.synchronize()
        start = time.perf_counter()
        transaction = engine.begin_step(request_ids)
        engine.step_layer(transaction, 0, q0, k0, v0)
        try:
            q1, k1, v1 = token_inputs[1]
            engine.step_layer(transaction, 1, bad_q, k1, v1)
        except ValueError:
            pass
        else:
            raise RuntimeError("rollback probe expected invalid q failure")
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1_000.0)
        if any(
            engine.cache.request_state(request_id)["seq_len"] != 0
            or engine.cache.request_block_ids(request_id)
            for request_id in request_ids
        ):
            raise RuntimeError("rollback probe left visible request state")
        engine.validate_invariants()
    return {
        "rollback_repeats": repeats,
        "rollback_p50_ms": percentile(latencies, 50),
        "rollback_blocks": engine.cache.metrics()[
            "transaction_rollback_block_count"
        ],
        "rollback_validated": True,
    }


def _summarize_row(
    torch,
    args,
    case,
    dtype_name,
    dtype,
    backend,
    trial,
    trial_seed,
    backend_order,
    timings,
    profile,
    rollback,
    engine,
    commit,
):
    wall = timings["wall_ms"]
    device = timings["device_ms"]
    layer_device = timings["layer_device_ms"]
    layer_total = timings["layer_total_device_ms"]
    metrics = engine.metrics()
    cache = metrics["cache"]
    mean_wall = statistics.fmean(wall)
    dtype_bytes = dtype_nbytes(dtype)
    kv_write_bytes_per_token = (
        case.batch_size
        * case.num_layers
        * 2
        * NUM_KV_HEADS
        * HEAD_DIM
        * dtype_bytes
    )
    cache_capacity_bytes = (
        case.num_layers
        * cache["max_blocks"]
        * 2
        * NUM_KV_HEADS
        * BLOCK_SIZE
        * HEAD_DIM
        * dtype_bytes
    )
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "op": "multi_layer_decode_engine",
        "case": case.name,
        "append_backend": backend,
        "decode_backend": "triton",
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": commit,
        "num_layers": case.num_layers,
        "batch_size": case.batch_size,
        "context_tokens": case.context_tokens,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "warmup": args.warmup,
        "trial": trial,
        "trial_count": args.trials,
        "backend_order": "->".join(backend_order),
        "seed": trial_seed,
        "p99_ms": f"{percentile(wall, 99):.6f}",
        "device_mean_ms": f"{statistics.fmean(device):.6f}",
        "device_p50_ms": f"{percentile(device, 50):.6f}",
        "layer_device_mean_ms": f"{statistics.fmean(layer_device):.6f}",
        "layer_device_p50_ms": f"{percentile(layer_device, 50):.6f}",
        "layer_total_device_p50_ms": f"{percentile(layer_total, 50):.6f}",
        "begin_host_mean_ms": f"{statistics.fmean(timings['begin_host_ms']):.6f}",
        "commit_host_mean_ms": f"{statistics.fmean(timings['commit_host_ms']):.6f}",
        "decode_tokens_per_second": f"{case.batch_size * 1_000.0 / mean_wall:.3f}",
        "layer_steps_per_second": (
            f"{case.batch_size * case.num_layers * 1_000.0 / mean_wall:.3f}"
        ),
        "kv_write_bytes_per_token": kv_write_bytes_per_token,
        "cache_capacity_bytes": cache_capacity_bytes,
        "final_seq_len": case.context_tokens + len(wall),
        "final_used_blocks": cache["used_blocks"],
        "final_free_blocks": cache["free_blocks"],
        "max_blocks": cache["max_blocks"],
        "transaction_begin_count": cache["transaction_begin_count"],
        "transaction_commit_count": cache["transaction_commit_count"],
        "transaction_abort_count": cache["transaction_abort_count"],
        "transaction_layer_write_count": cache["transaction_layer_write_count"],
        "validated_invariants": engine.validate_invariants(),
        "timing_scope": (
            "non-instrumented synchronized complete token wall-clock; inputs, "
            "context seed, JIT build, profiler probe, and rollback probe excluded"
        ),
        "profile_timing_scope": "separate instrumented attribution probe",
        **{
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in profile.items()
        },
        **{
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in rollback.items()
        },
    }
    return BenchmarkResult(
        name="multi_layer_decode_engine",
        mean_ms=mean_wall,
        p50_ms=percentile(wall, 50),
        p90_ms=percentile(wall, 90),
        min_ms=min(wall),
        max_ms=max(wall),
        repeats=len(wall),
        metadata=metadata,
    )


def _run_case(
    torch,
    args,
    case,
    dtype_name,
    dtype,
    backend,
    trial,
    trial_seed,
    backend_order,
    commit,
):
    warmup_engine, warmup_ids = _make_engine(
        torch, case, dtype, backend, args.warmup, profiled=False
    )
    _seed_context(torch, warmup_engine, warmup_ids, case.context_tokens)
    warmup_inputs = _generate_inputs(
        torch, case, dtype, args.warmup, trial_seed + 100_000
    )
    _run_tokens(torch, warmup_engine, warmup_ids, warmup_inputs)

    engine, request_ids = _make_engine(
        torch, case, dtype, backend, args.repeat, profiled=False
    )
    _seed_context(torch, engine, request_ids, case.context_tokens)
    inputs = _generate_inputs(torch, case, dtype, args.repeat, trial_seed)
    timings = _run_tokens(torch, engine, request_ids, inputs)
    profile = _profile_probe(
        torch,
        case,
        dtype,
        backend,
        args.profile_steps,
        trial_seed + 200_000,
    )
    rollback = _rollback_probe(
        torch,
        case,
        dtype,
        backend,
        args.rollback_repeats,
        trial_seed + 300_000,
    )
    return _summarize_row(
        torch,
        args,
        case,
        dtype_name,
        dtype,
        backend,
        trial,
        trial_seed,
        backend_order,
        timings,
        profile,
        rollback,
        engine,
        commit,
    )


def _with_speedup(result, torch_result):
    metadata = dict(result.metadata)
    metadata["speedup_vs_torch_p50"] = (
        f"{torch_result.p50_ms / result.p50_ms:.4f}"
    )
    return replace(result, metadata=metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "both"], default="both"
    )
    parser.add_argument(
        "--append-backends",
        nargs="+",
        choices=["torch", "fused_cuda"],
        default=["torch", "fused_cuda"],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--rollback-repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=541)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/multi_layer_transaction.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0:
        raise SystemExit("warmup must be non-negative")
    if args.repeat <= 0 or args.trials <= 0:
        raise SystemExit("repeat and trials must be positive")
    if args.profile_steps <= 0:
        raise SystemExit("profile-steps must be positive")
    if len(set(args.append_backends)) != len(args.append_backends):
        raise SystemExit("append-backends must be unique")
    if args.quick:
        args.warmup = min(args.warmup, 1)
        args.repeat = min(args.repeat, 5)
        args.profile_steps = min(args.profile_steps, 2)
        args.rollback_repeats = min(args.rollback_repeats, 1)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for multi-layer Engine benchmarks")
    if "fused_cuda" in args.append_backends:
        from flashdec import load_fused_rope_kv_append_extension

        load_fused_rope_kv_append_extension()

    cases = _selected_cases(args.case, args.quick)
    if any(case.num_layers > 1 for case in cases) and args.rollback_repeats <= 0:
        raise SystemExit("rollback-repeats must be positive for multi-layer cases")
    commit = git_commit(PROJECT_ROOT)
    results = []
    for dtype_name, dtype in _requested_dtypes(torch, args.dtype):
        for case in cases:
            for trial_index in range(args.trials):
                trial = trial_index + 1
                trial_seed = args.seed + trial_index
                backend_order = _trial_backend_order(
                    args.append_backends, trial_index
                )
                paired = [
                    _run_case(
                        torch,
                        args,
                        case,
                        dtype_name,
                        dtype,
                        backend,
                        trial,
                        trial_seed,
                        backend_order,
                        commit,
                    )
                    for backend in backend_order
                ]
                torch_result = next(
                    row
                    for row in paired
                    if row.metadata["append_backend"] == "torch"
                )
                results.extend(_with_speedup(row, torch_result) for row in paired)

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
