"""Benchmark materialized versus persistent Cache transaction metadata.

Both paths execute the same R4-A trusted fused CUDA append and Triton decode
math.  The ``materialized`` path exists only inside this benchmark: private
instance methods are patched to reproduce the pre-R4-B ``2L + 2`` transaction
view materializations.  No benchmark policy is added to the public API.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from flashdec.engine import (
    DecodeLayerResult,
    DecodeStepResult,
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
)
from flashdec.perf import dtype_nbytes


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 32
NUM_WARPS = 2
METADATA_PATHS = ("materialized", "persistent")
METADATA_COUNTERS = (
    "transaction_metadata_build_count",
    "transaction_metadata_materialization_count",
    "transaction_metadata_reuse_count",
    "transaction_metadata_release_count",
)
WALL_TIMING_SCOPE = (
    "pure synchronized complete-token wall; no CUDA events or profiler in interval; "
    "inputs, context seed, JIT build, and attribution probe excluded"
)
PROFILE_TIMING_SCOPE = (
    "separate CPU-only profiler attribution probe; device attribution excluded"
)
MAX_PROFILE_ATTEMPTS = 3


class IncompleteProfilerTrace(RuntimeError):
    """Raised when Kineto did not capture a complete CPU attribution trace."""


@dataclass(frozen=True)
class MetadataCase:
    name: str
    num_layers: int
    batch_size: int
    context_tokens: int


CASES = {
    f"l{layers}_b{batch}_c{context}": MetadataCase(
        name=f"l{layers}_b{batch}_c{context}",
        num_layers=layers,
        batch_size=batch,
        context_tokens=context,
    )
    for layers in (2, 4)
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
            raise RuntimeError("bfloat16 requested but this CUDA device lacks BF16")
        result.append((name, dtype))
    return result


def _trial_path_order(paths, trial_index):
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError("trial_index must be non-negative")
    result = list(paths)
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
    selected = [CASES[name]] if name != "all" else list(CASES.values())
    return [_quick_case(case) for case in selected] if quick else selected


def _max_blocks(case, measured_steps):
    per_request = math.ceil(
        (case.context_tokens + int(measured_steps) + 1) / BLOCK_SIZE
    )
    return case.batch_size * per_request


@contextmanager
def _metadata_path_context(engine, metadata_path):
    """Apply the benchmark-only legacy materialization policy to one Cache.

    Production ``persistent`` execution is untouched.  ``materialized``
    replaces only package-private instance hooks and restores them afterwards.
    The patched path still enters ``_write_token_layer_fused_cuda_for_engine``,
    whose core always launches R4-A's trusted raw CUDA function.
    """
    if metadata_path not in METADATA_PATHS:
        raise ValueError(f"unsupported metadata path: {metadata_path}")
    if metadata_path == "persistent":
        yield
        return

    cache = engine.cache
    required_cache = (
        "_transaction_device_metadata_for_layer",
        "_materialize_transaction_device_metadata_for_layer",
        "_abort_token_for_engine",
    )
    required_engine = (
        "_write_transaction_layer",
        "_prepare_layer_step_result",
        "_commit_cache_transaction_and_prepare_result",
    )
    missing = [
        f"Cache.{name}" for name in required_cache if not hasattr(cache, name)
    ] + [
        f"Engine.{name}" for name in required_engine if not hasattr(engine, name)
    ]
    if missing:
        raise RuntimeError(
            "R4-B private benchmark hooks are unavailable: " + ", ".join(missing)
        )

    patched_cache_names = (
        "_transaction_device_metadata_for_layer",
        "_abort_token_for_engine",
    )
    missing_instance_attribute = object()
    cache_instance_originals = {
        name: cache.__dict__.get(name, missing_instance_attribute)
        for name in patched_cache_names
    }
    engine_instance_originals = {
        name: engine.__dict__.get(name, missing_instance_attribute)
        for name in required_engine
    }

    def materialized_write_transaction_layer(
        state,
        layer_idx,
        q,
        k,
        v,
        *,
        rotary_dim=None,
        base=10_000.0,
    ):
        return cache.write_token_layer_fused_cuda(
            state.cache_transaction,
            layer_idx,
            q,
            k,
            v,
            rotary_dim=rotary_dim,
            base=base,
        )

    def materialized_layer_result(state, layer_idx, output, metadata):
        return DecodeLayerResult(
            transaction_id=state.handle.transaction_id,
            layer_idx=layer_idx,
            request_ids=state.handle.request_ids,
            output=output,
            positions=metadata.positions,
            block_tables=metadata.block_tables,
            effective_seq_lens=metadata.effective_seq_lens,
        )

    def materialized_commit_result(state):
        committed = cache.commit_token(state.cache_transaction)
        return DecodeStepResult(
            status=engine.STEP_OK,
            request_ids=state.handle.request_ids,
            output=state.last_output,
            positions=committed.positions,
            block_tables=committed.block_tables,
            seq_lens=cache.seq_lens_tensor(state.handle.request_ids),
            needed_new_blocks=state.needed_new_blocks,
            free_blocks=cache.num_free_blocks,
        )

    def materialized_abort(transaction):
        return cache.abort_token(transaction)

    cache._transaction_device_metadata_for_layer = (
        cache._materialize_transaction_device_metadata_for_layer
    )
    cache._abort_token_for_engine = materialized_abort
    engine._write_transaction_layer = materialized_write_transaction_layer
    engine._prepare_layer_step_result = materialized_layer_result
    engine._commit_cache_transaction_and_prepare_result = (
        materialized_commit_result
    )
    try:
        yield
    finally:
        for name, original in cache_instance_originals.items():
            if original is missing_instance_attribute:
                delattr(cache, name)
            else:
                setattr(cache, name, original)
        for name, original in engine_instance_originals.items():
            if original is missing_instance_attribute:
                delattr(engine, name)
            else:
                setattr(engine, name, original)


def _make_engine(torch, case, dtype, measured_steps, *, profiled):
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
        append_backend="fused_cuda",
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
    tokens = []
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
        tokens.append(tuple(layers))
    torch.cuda.synchronize()
    return tuple(tokens)


def _counter_snapshot(cache):
    metrics = cache.metrics()
    return {
        **{name: int(metrics[name]) for name in METADATA_COUNTERS},
        "resident_transaction_metadata_count": int(
            metrics["resident_transaction_metadata_count"]
        ),
    }


def _counter_delta(before, after, token_count):
    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    deltas = {name: after[name] - before[name] for name in METADATA_COUNTERS}
    if any(value < 0 for value in deltas.values()):
        raise RuntimeError(f"metadata counters moved backwards: {deltas}")
    return {
        "metadata_build_delta": deltas["transaction_metadata_build_count"],
        "metadata_materialization_delta": deltas[
            "transaction_metadata_materialization_count"
        ],
        "metadata_reuse_delta": deltas["transaction_metadata_reuse_count"],
        "metadata_release_delta": deltas[
            "transaction_metadata_release_count"
        ],
        "metadata_resident_before": before[
            "resident_transaction_metadata_count"
        ],
        "metadata_resident_after": after[
            "resident_transaction_metadata_count"
        ],
        "metadata_builds_per_token": deltas[
            "transaction_metadata_build_count"
        ]
        / token_count,
        "metadata_materializations_per_token": deltas[
            "transaction_metadata_materialization_count"
        ]
        / token_count,
        "metadata_reuses_per_token": deltas[
            "transaction_metadata_reuse_count"
        ]
        / token_count,
        "metadata_releases_per_token": deltas[
            "transaction_metadata_release_count"
        ]
        / token_count,
    }


def _run_tokens(torch, engine, request_ids, inputs):
    """Measure complete tokens with synchronized wall time and no CUDA Events."""
    wall_ms = []
    begin_host_ms = []
    commit_host_ms = []
    counters_before = _counter_snapshot(engine.cache)
    for token_inputs in inputs:
        torch.cuda.synchronize()
        wall_start = time.perf_counter()

        host_start = time.perf_counter()
        transaction = engine.begin_step(request_ids)
        begin_host_ms.append((time.perf_counter() - host_start) * 1_000.0)

        for layer_idx, (q, k, v) in enumerate(token_inputs):
            engine.step_layer(transaction, layer_idx, q, k, v)

        host_start = time.perf_counter()
        engine.commit_step(transaction)
        commit_host_ms.append((time.perf_counter() - host_start) * 1_000.0)

        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1_000.0)
    counters_after = _counter_snapshot(engine.cache)
    return {
        "wall_ms": wall_ms,
        "begin_host_ms": begin_host_ms,
        "commit_host_ms": commit_host_ms,
        "metadata": _counter_delta(counters_before, counters_after, len(inputs)),
    }


def _execute_for_parity(torch, engine, request_ids, inputs):
    outputs = []
    for token_inputs in inputs:
        transaction = engine.begin_step(request_ids)
        layer_outputs = []
        for layer_idx, (q, k, v) in enumerate(token_inputs):
            layer_outputs.append(
                engine.step_layer(transaction, layer_idx, q, k, v).output
            )
        engine.commit_step(transaction)
        outputs.append(tuple(layer_outputs))
    torch.cuda.synchronize()
    return tuple(outputs)


def _functional_cache_metrics(metrics):
    ignored = set(METADATA_COUNTERS) | {"resident_transaction_metadata_count"}
    return {key: value for key, value in metrics.items() if key not in ignored}


def _parity_probe(torch, case, dtype, steps, seed):
    inputs = _generate_inputs(torch, case, dtype, steps, seed)
    request_ids = tuple(range(case.batch_size))
    engines = {}
    outputs = {}
    for metadata_path in METADATA_PATHS:
        engine, actual_ids = _make_engine(
            torch, case, dtype, steps, profiled=False
        )
        if actual_ids != request_ids:
            raise RuntimeError("parity request row order changed")
        _seed_context(torch, engine, request_ids, case.context_tokens)
        with _metadata_path_context(engine, metadata_path):
            outputs[metadata_path] = _execute_for_parity(
                torch, engine, request_ids, inputs
            )
        engines[metadata_path] = engine

    materialized = engines["materialized"]
    persistent = engines["persistent"]
    output_equal = all(
        torch.equal(lhs, rhs)
        for lhs_token, rhs_token in zip(
            outputs["materialized"], outputs["persistent"]
        )
        for lhs, rhs in zip(lhs_token, rhs_token)
    )
    cache_equal = torch.equal(
        materialized.cache.k_cache, persistent.cache.k_cache
    ) and torch.equal(materialized.cache.v_cache, persistent.cache.v_cache)
    state_equal = (
        _functional_cache_metrics(materialized.cache.metrics())
        == _functional_cache_metrics(persistent.cache.metrics())
        and all(
            materialized.cache.request_state(request_id)
            == persistent.cache.request_state(request_id)
            for request_id in request_ids
        )
        and materialized.metrics()["completed_step_count"]
        == persistent.metrics()["completed_step_count"]
        and materialized.metrics()["appended_token_count"]
        == persistent.metrics()["appended_token_count"]
    )
    validated = (
        output_equal
        and cache_equal
        and state_equal
        and materialized.validate_invariants()
        and persistent.validate_invariants()
    )
    if not validated:
        raise RuntimeError("materialized/persistent parity probe failed")
    return {
        "parity_steps": steps,
        "parity_output_equal": output_equal,
        "parity_cache_equal": cache_equal,
        "parity_state_equal": state_equal,
        "parity_validated": validated,
    }


def _event_is_cpu(event):
    return "cpu" in str(getattr(event, "device_type", "")).lower()


def _event_time_us(event):
    return float(getattr(event, "cpu_time_total", 0.0) or 0.0)


def _raw_profile_range_totals(raw_events, key, expected_count):
    ranges = tuple(
        event
        for event in raw_events
        if getattr(event, "key", None) == key
        and _event_is_cpu(event)
        and getattr(event, "is_user_annotation", False) is True
    )
    if len(ranges) > expected_count:
        raise RuntimeError(
            f"profiler range {key!r} exceeded contract: "
            f"expected {expected_count}, found {len(ranges)}"
        )
    if len(ranges) < expected_count:
        raise IncompleteProfilerTrace(
            f"profiler range {key!r} incomplete: expected {expected_count}, "
            f"found {len(ranges)}"
        )
    times = tuple(_event_time_us(event) for event in ranges)
    if any(value <= 0.0 or not math.isfinite(value) for value in times):
        raise IncompleteProfilerTrace(
            f"profiler range {key!r} inclusive CPU time must be positive"
        )
    return {"count": len(ranges), "cpu_time_us": sum(times)}


def _raw_cpu_operator_count(raw_events, key):
    return sum(
        1
        for event in raw_events
        if getattr(event, "key", None) == key and _event_is_cpu(event)
    )


def _profile_scalar_counts(raw_events):
    counts = {
        "profile_item_count": _raw_cpu_operator_count(raw_events, "aten::item"),
        "profile_local_scalar_dense_count": _raw_cpu_operator_count(
            raw_events, "aten::_local_scalar_dense"
        ),
    }
    if any(counts.values()):
        raise RuntimeError(
            "R4-B paths must not extract device scalars during append: "
            f"{counts}"
        )
    return counts


def _profiler_kwargs(torch):
    return {
        "activities": [torch.profiler.ProfilerActivity.CPU],
        "schedule": torch.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1
        ),
        "acc_events": True,
    }


def _run_profile_token(engine, request_ids, token_inputs, *, commit):
    transaction = engine.begin_step(request_ids)
    for layer_idx, (q, k, v) in enumerate(token_inputs):
        engine.step_layer(transaction, layer_idx, q, k, v)
    if commit:
        engine.commit_step(transaction)
    else:
        engine.abort_step(transaction)


def _validate_profile_warmup_abort(engine, request_ids, expected_seq_len):
    metrics = engine.metrics()
    if metrics["open_step_transaction_count"] != 0:
        raise RuntimeError("profile warmup left an open Engine transaction")
    if metrics["cache"]["open_transaction_count"] != 0:
        raise RuntimeError("profile warmup left an open Cache transaction")
    if metrics["cache"]["resident_transaction_metadata_count"] != 0:
        raise RuntimeError("profile warmup retained transaction metadata")
    if any(
        engine.cache.request_state(request_id)["seq_len"] != expected_seq_len
        for request_id in request_ids
    ):
        raise RuntimeError("profile warmup changed committed request length")
    if not engine.validate_invariants():
        raise RuntimeError("profile warmup violated Engine invariants")


def _profile_probe_once(torch, case, dtype, steps, seed, metadata_path):
    engine, request_ids = _make_engine(
        torch, case, dtype, steps, profiled=True
    )
    _seed_context(torch, engine, request_ids, case.context_tokens)
    inputs = _generate_inputs(torch, case, dtype, steps, seed)
    warmup = _generate_inputs(torch, case, dtype, 1, seed + 1_000_000)[0]
    completed = 0
    with _metadata_path_context(engine, metadata_path):
        with torch.profiler.profile(**_profiler_kwargs(torch)) as profiler:
            _run_profile_token(engine, request_ids, warmup, commit=False)
            torch.cuda.synchronize()
            _validate_profile_warmup_abort(
                engine, request_ids, case.context_tokens
            )
            profiler.step()
            for token_inputs in inputs:
                _run_profile_token(engine, request_ids, token_inputs, commit=True)
                completed += 1
            torch.cuda.synchronize()
            profiler.step()

    raw_events = tuple(profiler.events())
    expected_layers = steps * case.num_layers
    append = _raw_profile_range_totals(
        raw_events, PROFILE_RANGE_APPEND, expected_layers
    )
    decode = _raw_profile_range_totals(
        raw_events, PROFILE_RANGE_DECODE, expected_layers
    )
    counts = {
        "profile_token_count": completed,
        "profile_append_count": append["count"],
        "profile_decode_count": decode["count"],
    }
    if counts != {
        "profile_token_count": steps,
        "profile_append_count": expected_layers,
        "profile_decode_count": expected_layers,
    }:
        raise IncompleteProfilerTrace(
            f"persistent metadata profiler counts are invalid: {counts}"
        )
    return {
        "profile_steps": steps,
        **counts,
        "profile_append_cpu_ms_per_layer": (
            append["cpu_time_us"] / 1_000.0 / expected_layers
        ),
        **_profile_scalar_counts(raw_events),
    }


def _profile_probe(
    torch,
    case,
    dtype,
    steps,
    seed,
    metadata_path,
    *,
    max_attempts=MAX_PROFILE_ATTEMPTS,
):
    max_attempts = int(max_attempts)
    if max_attempts <= 0 or max_attempts > MAX_PROFILE_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1, {MAX_PROFILE_ATTEMPTS}]")
    failures = []
    for attempt in range(1, max_attempts + 1):
        try:
            result = _profile_probe_once(
                torch, case, dtype, steps, seed, metadata_path
            )
        except IncompleteProfilerTrace as exc:
            failures.append(str(exc))
            if attempt == max_attempts:
                raise IncompleteProfilerTrace(
                    "profiler attribution remained incomplete after "
                    f"{max_attempts} attempts: {' | '.join(failures)}"
                ) from exc
            print(
                f"Profiler attempt {attempt}/{max_attempts} was incomplete; "
                f"rebuilding probe: {exc}",
                file=sys.stderr,
            )
            continue
        return {**result, "profile_attempt_count": attempt}
    raise AssertionError("unreachable profiler retry state")


def _rollback_probe(torch, case, dtype, repeats, seed, metadata_path):
    if case.num_layers < 2:
        raise ValueError("rollback probe requires at least two layers")
    probe_case = replace(case, context_tokens=0)
    engine, request_ids = _make_engine(
        torch, probe_case, dtype, 1, profiled=False
    )
    inputs = _generate_inputs(torch, probe_case, dtype, repeats, seed)
    latencies = []
    before = _counter_snapshot(engine.cache)
    with _metadata_path_context(engine, metadata_path):
        for token_inputs in inputs:
            q0, k0, v0 = token_inputs[0]
            bad_q = token_inputs[1][0][..., :-1]
            torch.cuda.synchronize()
            start = time.perf_counter()
            transaction = engine.begin_step(request_ids)
            engine.step_layer(transaction, 0, q0, k0, v0)
            try:
                _q1, k1, v1 = token_inputs[1]
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
    after = _counter_snapshot(engine.cache)
    delta = _counter_delta(before, after, repeats)
    return {
        "rollback_repeats": repeats,
        "rollback_p50_ms": percentile(latencies, 50),
        "rollback_blocks": engine.cache.metrics()[
            "transaction_rollback_block_count"
        ],
        "rollback_metadata_releases": delta["metadata_release_delta"],
        "rollback_metadata_resident_after": delta["metadata_resident_after"],
        "rollback_validated": True,
    }


def _run_wall_case(torch, args, case, dtype, metadata_path, trial_seed):
    warmup_engine, warmup_ids = _make_engine(
        torch, case, dtype, args.warmup, profiled=False
    )
    _seed_context(torch, warmup_engine, warmup_ids, case.context_tokens)
    warmup_inputs = _generate_inputs(
        torch, case, dtype, args.warmup, trial_seed + 100_000
    )
    with _metadata_path_context(warmup_engine, metadata_path):
        _run_tokens(torch, warmup_engine, warmup_ids, warmup_inputs)

    engine, request_ids = _make_engine(
        torch, case, dtype, args.repeat, profiled=False
    )
    _seed_context(torch, engine, request_ids, case.context_tokens)
    inputs = _generate_inputs(torch, case, dtype, args.repeat, trial_seed)
    with _metadata_path_context(engine, metadata_path):
        timings = _run_tokens(torch, engine, request_ids, inputs)
    return {"timings": timings, "engine": engine}


def _summarize_row(
    torch,
    args,
    case,
    dtype_name,
    dtype,
    metadata_path,
    trial,
    trial_seed,
    path_order,
    wall_evidence,
    profile,
    rollback,
    parity,
    commit,
    run_id,
):
    timings = wall_evidence["timings"]
    engine = wall_evidence["engine"]
    wall = timings["wall_ms"]
    metrics = engine.metrics()
    cache = metrics["cache"]
    expected_seq_len = case.context_tokens + len(wall)
    if any(
        engine.cache.request_state(request_id)["seq_len"] != expected_seq_len
        for request_id in range(case.batch_size)
    ):
        raise RuntimeError("measured requests have an unexpected final seq_len")
    mean_wall = statistics.fmean(wall)
    dtype_bytes = dtype_nbytes(dtype)
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "op": "persistent_transaction_metadata",
        "case": case.name,
        "metadata_path": metadata_path,
        "append_backend": "fused_cuda",
        "raw_dispatch": "trusted",
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
        "path_order": "->".join(path_order),
        "seed": trial_seed,
        "p99_ms": f"{percentile(wall, 99):.6f}",
        "begin_host_p50_ms": f"{percentile(timings['begin_host_ms'], 50):.6f}",
        "commit_host_p50_ms": f"{percentile(timings['commit_host_ms'], 50):.6f}",
        "decode_tokens_per_second": f"{case.batch_size * 1_000.0 / mean_wall:.3f}",
        "layer_steps_per_second": (
            f"{case.batch_size * case.num_layers * 1_000.0 / mean_wall:.3f}"
        ),
        "kv_write_bytes_per_token": (
            case.batch_size
            * case.num_layers
            * 2
            * NUM_KV_HEADS
            * HEAD_DIM
            * dtype_bytes
        ),
        "cache_capacity_bytes": (
            case.num_layers
            * cache["max_blocks"]
            * 2
            * NUM_KV_HEADS
            * BLOCK_SIZE
            * HEAD_DIM
            * dtype_bytes
        ),
        "final_seq_len": expected_seq_len,
        "final_used_blocks": cache["used_blocks"],
        "final_free_blocks": cache["free_blocks"],
        "final_request_blocks": sum(
            len(engine.cache.request_block_ids(request_id))
            for request_id in range(case.batch_size)
        ),
        "max_blocks": cache["max_blocks"],
        "allocation_count": cache["allocation_count"],
        "fresh_allocation_count": cache["fresh_allocation_count"],
        "reuse_count": cache["reuse_count"],
        "capacity_failure_count": cache["capacity_failure_count"],
        "transaction_begin_count": cache["transaction_begin_count"],
        "transaction_commit_count": cache["transaction_commit_count"],
        "transaction_abort_count": cache["transaction_abort_count"],
        "transaction_layer_write_count": cache["transaction_layer_write_count"],
        "engine_completed_step_count": metrics["completed_step_count"],
        "engine_appended_token_count": metrics["appended_token_count"],
        "validated_invariants": engine.validate_invariants(),
        "timing_scope": WALL_TIMING_SCOPE,
        "wall_timer_cuda_events": False,
        "profile_timing_scope": PROFILE_TIMING_SCOPE,
        **{
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in timings["metadata"].items()
        },
        **{
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in profile.items()
        },
        **{
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in rollback.items()
        },
        **parity,
    }
    return BenchmarkResult(
        name="persistent_transaction_metadata",
        mean_ms=mean_wall,
        p50_ms=percentile(wall, 50),
        p90_ms=percentile(wall, 90),
        min_ms=min(wall),
        max_ms=max(wall),
        repeats=len(wall),
        metadata=metadata,
    )


def _run_paired_trial(
    torch,
    args,
    case,
    dtype_name,
    dtype,
    trial_index,
    commit,
    run_id,
):
    trial = trial_index + 1
    trial_seed = args.seed + trial_index
    path_order = _trial_path_order(args.metadata_paths, trial_index)
    parity = _parity_probe(
        torch, case, dtype, args.parity_steps, trial_seed + 400_000
    )

    # Complete both pure-wall paths before profiler/rollback attribution so a
    # Kineto retry can never asymmetrically perturb the paired latency order.
    wall = {
        path: _run_wall_case(torch, args, case, dtype, path, trial_seed)
        for path in path_order
    }
    rows = []
    for path in path_order:
        profile = _profile_probe(
            torch,
            case,
            dtype,
            args.profile_steps,
            trial_seed + 200_000,
            path,
        )
        rollback = _rollback_probe(
            torch,
            case,
            dtype,
            args.rollback_repeats,
            trial_seed + 300_000,
            path,
        )
        rows.append(
            _summarize_row(
                torch,
                args,
                case,
                dtype_name,
                dtype,
                path,
                trial,
                trial_seed,
                path_order,
                wall[path],
                profile,
                rollback,
                parity,
                commit,
                run_id,
            )
        )
    return rows


def _with_speedup(result, materialized):
    metadata = dict(result.metadata)
    metadata["speedup_vs_materialized_p50"] = (
        f"{materialized.p50_ms / result.p50_ms:.4f}"
    )
    return replace(result, metadata=metadata)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "both"], default="both"
    )
    parser.add_argument(
        "--metadata-paths",
        nargs="+",
        choices=list(METADATA_PATHS),
        default=list(METADATA_PATHS),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--parity-steps", type=int, default=2)
    parser.add_argument("--rollback-repeats", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/results/r4_persistent_transaction_metadata.csv",
    )
    return parser.parse_args(argv)


def _validate_args(args):
    if args.warmup <= 0:
        raise SystemExit("warmup must be positive")
    names = (
        "repeat",
        "profile_steps",
        "parity_steps",
        "rollback_repeats",
        "trials",
    )
    if any(getattr(args, name) <= 0 for name in names):
        raise SystemExit(
            "repeat, profile-steps, parity-steps, rollback-repeats, and trials "
            "must be positive"
        )
    if tuple(args.metadata_paths) != METADATA_PATHS:
        raise SystemExit(
            "metadata-paths must be exactly: materialized persistent"
        )
    if args.quick:
        args.warmup = min(args.warmup, 1)
        args.repeat = min(args.repeat, 5)
        args.profile_steps = min(args.profile_steps, 2)
        args.parity_steps = min(args.parity_steps, 1)
        args.rollback_repeats = min(args.rollback_repeats, 1)
    return args


def main(argv=None):
    args = _validate_args(parse_args(argv))

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for persistent metadata benchmarks")
    from flashdec import load_fused_rope_kv_append_extension

    load_fused_rope_kv_append_extension()
    cases = _selected_cases(args.case, args.quick)
    commit = git_commit(PROJECT_ROOT)
    run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S%z')}-{commit}"
    results = []
    for dtype_name, dtype in _requested_dtypes(torch, args.dtype):
        for case in cases:
            for trial_index in range(args.trials):
                paired = _run_paired_trial(
                    torch,
                    args,
                    case,
                    dtype_name,
                    dtype,
                    trial_index,
                    commit,
                    run_id,
                )
                materialized = next(
                    row
                    for row in paired
                    if row.metadata["metadata_path"] == "materialized"
                )
                results.extend(
                    _with_speedup(row, materialized) for row in paired
                )

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
