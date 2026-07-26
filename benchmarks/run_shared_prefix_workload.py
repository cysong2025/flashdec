"""Benchmark shared-prefix capacity, memory, lookup, and decode-step behavior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


HIT_RATES = (0, 25, 50, 75)
NUM_LAYERS = 1
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32
NUM_WARPS = 2


@dataclass(frozen=True)
class SharedPrefixConfig:
    request_count: int = 16
    context_tokens: int = 128
    warmup: int = 2
    repeat: int = 10
    capacity_ratio: float = 0.60

    @property
    def prefix_blocks(self):
        return self.context_tokens // BLOCK_SIZE

    @property
    def decode_tokens(self):
        return self.warmup + self.repeat

    @property
    def tail_blocks(self):
        return math.ceil(self.decode_tokens / BLOCK_SIZE)

    @property
    def no_share_lifetime_blocks(self):
        return self.request_count * (self.prefix_blocks + self.tail_blocks)

    @property
    def capacity_probe_blocks(self):
        return math.ceil(self.no_share_lifetime_blocks * self.capacity_ratio)


def _selected_hit_rates(value):
    if value == "all":
        return list(HIT_RATES)
    rate = int(value)
    if rate not in HIT_RATES:
        raise ValueError(f"unsupported hit rate: {rate}")
    return [rate]


def _trial_hit_rates(hit_rates, trial_index):
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    values = list(hit_rates)
    if not values:
        raise ValueError("hit_rates must be non-empty")
    offset = trial_index % len(values)
    return values[offset:] + values[:offset]


def _quick_config(config):
    return SharedPrefixConfig(
        request_count=4,
        context_tokens=32,
        warmup=min(config.warmup, 1),
        repeat=min(config.repeat, 3),
        capacity_ratio=config.capacity_ratio,
    )


def _validate_config(config, hit_rates):
    if config.request_count <= 0:
        raise ValueError("request_count must be positive")
    if config.context_tokens <= 0 or config.context_tokens % BLOCK_SIZE:
        raise ValueError("context_tokens must be a positive full-block multiple")
    if config.warmup < 0 or config.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if not 0.0 < config.capacity_ratio <= 1.0:
        raise ValueError("capacity_ratio must be in (0, 1]")
    if config.capacity_probe_blocks < config.prefix_blocks:
        raise ValueError("capacity probe must fit one resident prefix")
    for rate in hit_rates:
        if config.request_count * rate % 100:
            raise ValueError("request_count must make every hit rate integral")


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_engine(torch, dtype, max_blocks, prefix_blocks):
    from flashdec.cache import PagedKVCache
    from flashdec.engine import DecodeEngine

    cache = PagedKVCache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
        prefix_cache_capacity_blocks=prefix_blocks,
    )
    return DecodeEngine(
        cache,
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=NUM_WARPS,
    )


def _prefix_tensors(torch, config, dtype, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (
        NUM_LAYERS,
        config.prefix_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
    )
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    return k, v


def _request_specs(config, hit_count):
    from flashdec.scheduler import RequestSpec

    result = []
    for index in range(config.request_count):
        result.append(
            RequestSpec(
                request_id=f"request-{index}",
                initial_context_tokens=config.context_tokens,
                max_new_tokens=config.decode_tokens,
                submission_order=index,
                prefix_id="shared" if index < hit_count else None,
            )
        )
    return tuple(result)


def _scheduler(config):
    from flashdec.scheduler import BlockAwareScheduler, SchedulerConfig

    return BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=config.request_count,
            max_batch_requests=config.request_count,
        )
    )


def _register(engine, prefix_k, prefix_v):
    engine.register_prefix("shared", prefix_k, prefix_v)


def _capacity_probe(torch, config, dtype, hit_count, prefix_k, prefix_v):
    prefix_blocks = config.prefix_blocks if hit_count else 0
    engine = _make_engine(
        torch,
        dtype,
        config.capacity_probe_blocks,
        prefix_blocks,
    )
    if hit_count:
        _register(engine, prefix_k, prefix_v)
    for spec in _request_specs(config, hit_count):
        engine.submit_request(spec)
    decision = _scheduler(config).plan(engine.scheduling_snapshot(logical_step=0))
    engine.apply_scheduler_decision(decision)
    if not engine.validate_invariants():
        raise RuntimeError("capacity probe invariant validation failed")
    metrics = engine.metrics()
    result = {
        "capacity_probe_blocks": config.capacity_probe_blocks,
        "capacity_admitted_requests": len(decision.admit_ids),
        "capacity_waiting_requests": len(decision.waiting_ids),
        "capacity_rejected_requests": len(decision.rejected_ids),
        "capacity_admission_rate": len(decision.admit_ids) / config.request_count,
        "capacity_committed_blocks": decision.committed_blocks_after,
        "capacity_physical_blocks": metrics["cache"]["used_blocks"],
    }
    for request_id in engine.active_request_ids():
        engine.finish_request(request_id)
    if not engine.validate_invariants():
        raise RuntimeError("capacity probe cleanup invariant validation failed")
    return result


def _attach_latency_probe(torch, config, dtype, hit_count, prefix_k, prefix_v):
    if not hit_count:
        return {"attach_mean_us": 0.0, "attach_p50_us": 0.0, "attach_p90_us": 0.0}
    from flashdec.cache import PagedKVCache

    cache = PagedKVCache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=config.prefix_blocks,
        dtype=dtype,
        device="cuda",
        prefix_cache_capacity_blocks=config.prefix_blocks,
    )
    cache.register_prefix("shared", prefix_k, prefix_v)
    request_ids = tuple(f"attach-{index}" for index in range(hit_count))
    for request_id in request_ids:
        cache.add_request(request_id)
    torch.cuda.synchronize()
    latencies = []
    for request_id in request_ids:
        start = time.perf_counter_ns()
        cache.attach_prefix(request_id, "shared")
        latencies.append((time.perf_counter_ns() - start) / 1_000.0)
    for request_id in request_ids:
        cache.finish_request(request_id)
    cache.evict_prefix("shared")
    if not cache.validate_invariants():
        raise RuntimeError("attach probe invariant validation failed")
    return {
        "attach_mean_us": statistics.fmean(latencies),
        "attach_p50_us": percentile(latencies, 50),
        "attach_p90_us": percentile(latencies, 90),
    }


def _dense_prefix(prefix):
    return (
        prefix[0]
        .permute(0, 2, 1, 3)
        .reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        .contiguous()
    )


def _prefill_misses(engine, request_ids, dense_k, dense_v):
    for request_id in request_ids:
        for token_index in range(dense_k.shape[0]):
            engine.prefill_request(
                request_id,
                dense_k[token_index],
                dense_v[token_index],
            )


def _validate_context(torch, engine, request_ids, dense_k, dense_v):
    actual_k, actual_v, seq_lens = engine.cache.to_dense(0, request_ids)
    expected_k = dense_k.unsqueeze(0).expand(len(request_ids), -1, -1, -1)
    expected_v = dense_v.unsqueeze(0).expand(len(request_ids), -1, -1, -1)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    if seq_lens.tolist() != [dense_k.shape[0]] * len(request_ids):
        raise RuntimeError("materialized context lengths do not match")


def _decode_inputs(torch, config, dtype, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    result = []
    for _ in range(config.decode_tokens):
        q = torch.randn(
            (config.request_count, NUM_Q_HEADS, HEAD_DIM),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        k = torch.randn(
            (config.request_count, NUM_KV_HEADS, HEAD_DIM),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        v = torch.randn(
            (config.request_count, NUM_KV_HEADS, HEAD_DIM),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        result.append((q, k, v))
    torch.cuda.synchronize()
    return tuple(result)


def _run_decode_steps(torch, engine, scheduler, request_ids, inputs, warmup):
    scheduler_ms = []
    engine_ms = []
    complete_ms = []
    last_output = None
    for step_index, (q, k, v) in enumerate(inputs):
        host_start = time.perf_counter_ns()
        snapshot = engine.scheduling_snapshot(logical_step=step_index)
        decision = scheduler.plan(snapshot)
        engine.apply_scheduler_decision(decision)
        host_ms = (time.perf_counter_ns() - host_start) / 1_000_000.0
        if decision.runnable_ids != request_ids:
            raise RuntimeError("latency probe did not preserve the full request batch")

        torch.cuda.synchronize()
        device_start = time.perf_counter_ns()
        result = engine.step(q, k, v, request_ids=request_ids)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter_ns() - device_start) / 1_000_000.0
        if result.status != engine.STEP_OK:
            raise RuntimeError("latency probe decode step failed")
        last_output = result.output
        if step_index >= warmup:
            scheduler_ms.append(host_ms)
            engine_ms.append(step_ms)
            complete_ms.append(host_ms + step_ms)
    if last_output is None or not bool(torch.isfinite(last_output).all().item()):
        raise RuntimeError("latency probe produced non-finite output")
    return scheduler_ms, engine_ms, complete_ms


def _latency_probe(torch, config, dtype, hit_count, prefix_k, prefix_v, seed):
    prefix_blocks = config.prefix_blocks if hit_count else 0
    engine = _make_engine(
        torch,
        dtype,
        config.no_share_lifetime_blocks,
        prefix_blocks,
    )
    registration_ms = 0.0
    if hit_count:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        _register(engine, prefix_k, prefix_v)
        torch.cuda.synchronize()
        registration_ms = (time.perf_counter_ns() - start) / 1_000_000.0

    specs = _request_specs(config, hit_count)
    request_ids = tuple(spec.request_id for spec in specs)
    for spec in specs:
        engine.submit_request(spec)
    scheduler = _scheduler(config)
    admission = scheduler.plan(engine.scheduling_snapshot(logical_step=0))
    engine.apply_scheduler_decision(admission)
    if admission.admit_ids != request_ids or admission.waiting_ids or admission.rejected_ids:
        raise RuntimeError("latency probe requires complete admission")

    dense_k = _dense_prefix(prefix_k)
    dense_v = _dense_prefix(prefix_v)
    miss_ids = request_ids[hit_count:]
    _prefill_misses(engine, miss_ids, dense_k, dense_v)
    torch.cuda.synchronize()
    _validate_context(torch, engine, request_ids, dense_k, dense_v)

    inputs = _decode_inputs(torch, config, dtype, seed + 100_000)
    scheduler_ms, engine_ms, complete_ms = _run_decode_steps(
        torch,
        engine,
        scheduler,
        request_ids,
        inputs,
        config.warmup,
    )
    metrics = engine.metrics()["cache"]
    if not engine.validate_invariants():
        raise RuntimeError("latency probe invariant validation failed")
    if hit_count:
        prefix_state = engine.cache.prefix_state("shared")
        block_index = torch.tensor(prefix_state["block_ids"], device="cuda")
        torch.testing.assert_close(engine.cache.k_cache.index_select(1, block_index), prefix_k)
        torch.testing.assert_close(engine.cache.v_cache.index_select(1, block_index), prefix_v)

    for request_id in request_ids:
        engine.finish_request(request_id)
    eviction_us = 0.0
    if hit_count:
        start = time.perf_counter_ns()
        engine.cache.evict_prefix("shared")
        eviction_us = (time.perf_counter_ns() - start) / 1_000.0
    if not engine.cache.validate_invariants():
        raise RuntimeError("latency probe cleanup invariant validation failed")
    cleanup_metrics = engine.cache.metrics()
    return {
        "registration_ms": registration_ms,
        "scheduler_ms": scheduler_ms,
        "engine_ms": engine_ms,
        "complete_ms": complete_ms,
        "cache_metrics": metrics,
        "eviction_us": eviction_us,
        "prefix_eviction_count": cleanup_metrics["prefix_eviction_count"],
        "final_free_blocks": engine.cache.num_free_blocks,
        "validated_invariants": True,
    }


def _result(torch, args, config, dtype_name, dtype, hit_rate, order, trial, seed, commit):
    hit_count = config.request_count * hit_rate // 100
    miss_count = config.request_count - hit_count
    prefix_k, prefix_v = _prefix_tensors(torch, config, dtype, seed)
    capacity = _capacity_probe(
        torch, config, dtype, hit_count, prefix_k, prefix_v
    )
    attach = _attach_latency_probe(
        torch, config, dtype, hit_count, prefix_k, prefix_v
    )
    latency = _latency_probe(
        torch, config, dtype, hit_count, prefix_k, prefix_v, seed
    )
    complete = latency["complete_ms"]
    engine_ms = latency["engine_ms"]
    scheduler_ms = latency["scheduler_ms"]
    cache = latency["cache_metrics"]
    logical_context_blocks = config.request_count * config.prefix_blocks
    physical_context_blocks = (
        logical_context_blocks
        if hit_count == 0
        else (miss_count + 1) * config.prefix_blocks
    )
    memory_saving_ratio = 1.0 - physical_context_blocks / logical_context_blocks
    elapsed_seconds = sum(complete) / 1_000.0
    metadata = {
        "op": "shared_prefix_decode_workload",
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": commit,
        "append_backend": "fused_cuda",
        "decode_backend": "triton",
        "num_layers": NUM_LAYERS,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "request_count": config.request_count,
        "context_tokens": config.context_tokens,
        "prefix_blocks": config.prefix_blocks,
        "decode_tokens": config.decode_tokens,
        "tail_blocks": config.tail_blocks,
        "warmup": config.warmup,
        "hit_rate_percent": hit_rate,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "trial": trial,
        "trial_count": args.trials,
        "hit_rate_order": "->".join(str(value) for value in order),
        "seed": seed,
        "capacity_ratio": f"{config.capacity_ratio:.4f}",
        **capacity,
        "latency_max_blocks": config.no_share_lifetime_blocks,
        "logical_context_blocks": logical_context_blocks,
        "physical_context_blocks": physical_context_blocks,
        "context_memory_saving_ratio": f"{memory_saving_ratio:.6f}",
        "bytes_per_block": cache["bytes_per_block"],
        "physical_context_bytes": physical_context_blocks * cache["bytes_per_block"],
        "peak_used_blocks": cache["used_blocks"],
        "peak_allocated_kv_bytes": cache["allocated_kv_bytes"],
        "resident_prefix_blocks": cache["resident_prefix_blocks"],
        "active_prefix_references": cache["active_prefix_references"],
        "saved_prefix_blocks": cache["saved_prefix_blocks"],
        "saved_prefix_bytes": cache["saved_prefix_bytes"],
        "prefix_hit_count": cache["prefix_hit_count"],
        "prefix_miss_count": cache["prefix_miss_count"],
        "prefix_eviction_count": latency["prefix_eviction_count"],
        "registration_ms": f"{latency['registration_ms']:.6f}",
        "attach_mean_us": f"{attach['attach_mean_us']:.6f}",
        "attach_p50_us": f"{attach['attach_p50_us']:.6f}",
        "attach_p90_us": f"{attach['attach_p90_us']:.6f}",
        "eviction_us": f"{latency['eviction_us']:.6f}",
        "scheduler_p50_ms": f"{percentile(scheduler_ms, 50):.6f}",
        "engine_step_p50_ms": f"{percentile(engine_ms, 50):.6f}",
        "engine_step_p90_ms": f"{percentile(engine_ms, 90):.6f}",
        "engine_step_p99_ms": f"{percentile(engine_ms, 99):.6f}",
        "complete_step_p99_ms": f"{percentile(complete, 99):.6f}",
        "decode_tokens_per_second": f"{config.request_count * config.repeat / elapsed_seconds:.3f}",
        "final_free_blocks": latency["final_free_blocks"],
        "validated_invariants": latency["validated_invariants"],
        "timing_scope": "non-instrumented scheduler host plus synchronized engine step wall time",
    }
    return BenchmarkResult(
        name="shared_prefix_workload",
        mean_ms=statistics.fmean(complete),
        p50_ms=percentile(complete, 50),
        p90_ms=percentile(complete, 90),
        min_ms=min(complete),
        max_ms=max(complete),
        repeats=len(complete),
        metadata=metadata,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hit-rate", choices=["all", *(str(v) for v in HIT_RATES)], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--request-count", type=int, default=16)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--capacity-ratio", type=float, default=0.60)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/shared_prefix_workload.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials <= 0:
        raise SystemExit("trials must be positive")
    hit_rates = _selected_hit_rates(args.hit_rate)
    config = SharedPrefixConfig(
        request_count=args.request_count,
        context_tokens=args.context_tokens,
        warmup=args.warmup,
        repeat=args.repeat,
        capacity_ratio=args.capacity_ratio,
    )
    if args.quick:
        config = _quick_config(config)
    try:
        _validate_config(config, hit_rates)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for shared-prefix workload benchmarks")
    from flashdec import load_fused_rope_kv_append_extension

    load_fused_rope_kv_append_extension()
    dtype_names = ("float16", "bfloat16") if args.dtype == "both" else (args.dtype,)
    commit = git_commit(PROJECT_ROOT)
    results = []
    for dtype_name in dtype_names:
        dtype = _dtype_from_name(torch, dtype_name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("this CUDA device does not support BF16")
        for trial_index in range(args.trials):
            order = _trial_hit_rates(hit_rates, trial_index)
            trial = trial_index + 1
            seed = args.seed + trial_index
            for hit_rate in order:
                result = _result(
                    torch,
                    args,
                    config,
                    dtype_name,
                    dtype,
                    hit_rate,
                    order,
                    trial,
                    seed,
                    commit,
                )
                results.append(result)
                print(result.as_row())
    write_csv(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
