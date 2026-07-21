"""R4-C scheduled multi-layer workload with fixed shared-prefix residency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
import time
from typing import Any, Hashable

from .benchmark import percentile
from .scheduled_workload import RequestArrival
from .scheduler import (
    ActiveRequestMetadata,
    BlockAwareScheduler,
    SchedulerConfig,
    SchedulingSnapshot,
    WaitingRequestMetadata,
)


def _require_int(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _blocks(token_count, block_size):
    return (int(token_count) + int(block_size) - 1) // int(block_size)


@dataclass(frozen=True)
class RequestCancellation:
    """Cancel one active request before scheduling ``logical_step``."""

    request_id: Hashable
    logical_step: int

    def __post_init__(self):
        try:
            hash(self.request_id)
        except TypeError as exc:
            raise ValueError("request_id must be hashable") from exc
        _require_int("logical_step", self.logical_step)


@dataclass(frozen=True)
class LayerFailure:
    """Inject one invalid layer input and require whole-token rollback."""

    logical_step: int
    layer_idx: int

    def __post_init__(self):
        _require_int("logical_step", self.logical_step)
        _require_int("layer_idx", self.layer_idx)


@dataclass(frozen=True)
class IntegratedWorkloadConfig:
    """Finite R4-C trace and frozen runtime geometry."""

    name: str
    num_layers: int
    arrivals: tuple[RequestArrival, ...]
    cancellations: tuple[RequestCancellation, ...] = ()
    failures: tuple[LayerFailure, ...] = ()
    max_active_requests: int = 3
    max_batch_requests: int = 2
    reserve_blocks: int = 0
    aging_threshold_steps: int = 4
    max_steps: int = 100

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        _require_int("num_layers", self.num_layers, minimum=2)
        arrivals = tuple(self.arrivals)
        cancellations = tuple(self.cancellations)
        failures = tuple(self.failures)
        if not arrivals or not all(isinstance(item, RequestArrival) for item in arrivals):
            raise ValueError("arrivals must contain RequestArrival values")
        if not all(isinstance(item, RequestCancellation) for item in cancellations):
            raise ValueError("cancellations must contain RequestCancellation values")
        if not all(isinstance(item, LayerFailure) for item in failures):
            raise ValueError("failures must contain LayerFailure values")
        object.__setattr__(self, "arrivals", arrivals)
        object.__setattr__(self, "cancellations", cancellations)
        object.__setattr__(self, "failures", failures)
        _require_int("max_active_requests", self.max_active_requests, minimum=1)
        _require_int("max_batch_requests", self.max_batch_requests, minimum=1)
        _require_int("reserve_blocks", self.reserve_blocks)
        _require_int("aging_threshold_steps", self.aging_threshold_steps, minimum=1)
        _require_int("max_steps", self.max_steps, minimum=1)
        if self.max_batch_requests > self.max_active_requests:
            raise ValueError("max_batch_requests must not exceed max_active_requests")

        request_ids = [item.spec.request_id for item in arrivals]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request ids must be unique")
        known = set(request_ids)
        cancellation_ids = [item.request_id for item in cancellations]
        if len(cancellation_ids) != len(set(cancellation_ids)):
            raise ValueError("a request may only be cancelled once")
        cancellation_steps = [item.logical_step for item in cancellations]
        if len(cancellation_steps) != len(set(cancellation_steps)):
            raise ValueError("at most one cancellation is allowed per logical step")
        if not set(cancellation_ids).issubset(known):
            raise ValueError("cancellations contain unknown request ids")
        failure_steps = [item.logical_step for item in failures]
        if len(failure_steps) != len(set(failure_steps)):
            raise ValueError("at most one layer failure is allowed per logical step")
        if any(item.layer_idx >= self.num_layers for item in failures):
            raise ValueError("failure layer_idx must be smaller than num_layers")
        if any(item.arrival_step >= self.max_steps for item in arrivals):
            raise ValueError("arrival steps must be smaller than max_steps")
        if any(item.logical_step >= self.max_steps for item in cancellations):
            raise ValueError("cancellation steps must be smaller than max_steps")
        if any(item.logical_step >= self.max_steps for item in failures):
            raise ValueError("failure steps must be smaller than max_steps")
        if not any(item.spec.prefix_id is None for item in arrivals):
            raise ValueError("R4-C trace requires at least one private-prefix miss")
        if not any(item.spec.prefix_id is not None for item in arrivals):
            raise ValueError("R4-C trace requires at least one shared-prefix hit")


@dataclass(frozen=True)
class IntegratedReferenceStep:
    """Dependency-free expected state transition for one logical step."""

    logical_step: int
    submitted_ids: tuple[Hashable, ...]
    cancelled_ids: tuple[Hashable, ...]
    admitted_ids: tuple[Hashable, ...]
    rejected_ids: tuple[Hashable, ...]
    runnable_ids: tuple[Hashable, ...]
    deferred_ids: tuple[Hashable, ...]
    completed_ids: tuple[Hashable, ...]
    positions: tuple[int, ...]
    aborted: bool
    committed_blocks: int
    used_blocks: int
    free_blocks: int

    def canonical(self):
        return {
            "logical_step": self.logical_step,
            "submitted_ids": list(self.submitted_ids),
            "cancelled_ids": list(self.cancelled_ids),
            "admitted_ids": list(self.admitted_ids),
            "rejected_ids": list(self.rejected_ids),
            "runnable_ids": list(self.runnable_ids),
            "deferred_ids": list(self.deferred_ids),
            "completed_ids": list(self.completed_ids),
            "positions": list(self.positions),
            "aborted": self.aborted,
            "committed_blocks": self.committed_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
        }


@dataclass(frozen=True)
class IntegratedReferenceTrajectory:
    """Pure scheduler/lifecycle reference used by CPU and CUDA executions."""

    steps: tuple[IntegratedReferenceStep, ...]
    completed_request_ids: tuple[Hashable, ...]
    cancelled_request_ids: tuple[Hashable, ...]
    rejected_request_ids: tuple[Hashable, ...]
    successful_steps: int
    aborted_steps: int
    completed_tokens: int
    digest: str


@dataclass(frozen=True)
class IntegratedWorkloadResult:
    """Measured R4-C execution plus audited lifecycle evidence."""

    config: IntegratedWorkloadConfig
    reference: IntegratedReferenceTrajectory
    step_latencies_ms: tuple[float, ...]
    scheduler_ms: tuple[float, ...]
    context_seed_ms: tuple[float, ...]
    engine_ms: tuple[float, ...]
    completed_request_ids: tuple[Hashable, ...]
    cancelled_request_ids: tuple[Hashable, ...]
    rejected_request_ids: tuple[Hashable, ...]
    successful_steps: int
    aborted_steps: int
    completed_tokens: int
    block_reuse_count: int
    terminal_resident_prefix_blocks: int
    final_free_blocks: int
    trajectory_digest: str
    engine_metrics: dict[str, Any]

    @property
    def mean_ms(self):
        return statistics.fmean(self.step_latencies_ms)

    @property
    def p50_ms(self):
        return percentile(self.step_latencies_ms, 50)

    @property
    def p90_ms(self):
        return percentile(self.step_latencies_ms, 90)

    @property
    def p99_ms(self):
        return percentile(self.step_latencies_ms, 99)

    @property
    def tokens_per_second(self):
        seconds = sum(self.step_latencies_ms) / 1_000.0
        return self.completed_tokens / seconds if seconds else 0.0


def standard_integrated_arrivals(*, context_tokens, prefix_id="shared"):
    """Return the frozen mixed hit/miss, staggered-arrival R4-C trace."""
    from .scheduler import RequestSpec

    _require_int("context_tokens", context_tokens, minimum=1)
    return (
        RequestArrival(RequestSpec("hit-a", context_tokens, 4, 0, prefix_id), 0),
        RequestArrival(RequestSpec("miss-a", context_tokens, 3, 1), 0),
        RequestArrival(RequestSpec("hit-cancel", context_tokens, 5, 2, prefix_id), 1),
        RequestArrival(RequestSpec("miss-b", context_tokens, 5, 3), 2),
    )


def standard_integrated_config(*, num_layers, context_tokens):
    """Build the pre-registered R4-C correctness/performance trace."""
    return IntegratedWorkloadConfig(
        name=f"l{int(num_layers)}_c{int(context_tokens)}",
        num_layers=int(num_layers),
        arrivals=standard_integrated_arrivals(context_tokens=context_tokens),
        cancellations=(RequestCancellation("hit-cancel", 3),),
        failures=(LayerFailure(4, 1),),
        max_active_requests=3,
        max_batch_requests=2,
        aging_threshold_steps=3,
        max_steps=40,
    )


def _trajectory_digest(steps):
    payload = [step.canonical() for step in steps]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_integrated_reference(config, *, block_size, max_blocks, resident_prefix_blocks):
    """Simulate the R4-C scheduler and lifecycle without importing torch."""
    if not isinstance(config, IntegratedWorkloadConfig):
        raise TypeError("config must be an IntegratedWorkloadConfig")
    _require_int("block_size", block_size, minimum=1)
    _require_int("max_blocks", max_blocks, minimum=1)
    _require_int("resident_prefix_blocks", resident_prefix_blocks, minimum=1)
    if config.reserve_blocks >= max_blocks:
        raise ValueError("reserve_blocks must be smaller than max_blocks")

    specs = {item.spec.request_id: item.spec for item in config.arrivals}
    arrivals_by_step = {}
    for item in config.arrivals:
        arrivals_by_step.setdefault(item.arrival_step, []).append(item.spec.request_id)
    cancellations_by_step = {
        item.logical_step: item.request_id for item in config.cancellations
    }
    failures_by_step = {item.logical_step: item for item in config.failures}
    shared_blocks = {}
    for request_id, spec in specs.items():
        if spec.prefix_id is None:
            shared_blocks[request_id] = 0
        else:
            if spec.initial_context_tokens % block_size:
                raise ValueError("shared prefix contexts must contain full blocks")
            shared_blocks[request_id] = spec.initial_context_tokens // block_size
            if shared_blocks[request_id] > resident_prefix_blocks:
                raise ValueError("request shared prefix exceeds fixed residency")

    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=config.max_active_requests,
            max_batch_requests=config.max_batch_requests,
            reserve_blocks=config.reserve_blocks,
            aging_threshold_steps=config.aging_threshold_steps,
        )
    )
    waiting = []
    active = {}
    wait_steps = {}
    skip_counts = {}
    completed = []
    cancelled = []
    rejected = []
    steps = []
    successful_steps = 0
    aborted_steps = 0
    completed_tokens = 0

    def used_blocks():
        return resident_prefix_blocks + sum(
            _blocks(state["seq_len"], block_size) - shared_blocks[request_id]
            for request_id, state in active.items()
        )

    def committed_blocks():
        return resident_prefix_blocks + sum(
            specs[request_id].commitment_blocks(block_size) - shared_blocks[request_id]
            for request_id in active
        )

    for logical_step in range(config.max_steps):
        submitted_ids = tuple(arrivals_by_step.get(logical_step, ()))
        for request_id in submitted_ids:
            waiting.append(request_id)
            wait_steps[request_id] = 0
            skip_counts[request_id] = 0

        cancelled_ids = ()
        request_id = cancellations_by_step.get(logical_step)
        if request_id is not None:
            if request_id not in active:
                raise ValueError("cancellation must target an active request")
            del active[request_id]
            cancelled.append(request_id)
            cancelled_ids = (request_id,)

        snapshot = SchedulingSnapshot(
            state_version=logical_step,
            logical_step=logical_step,
            block_size=block_size,
            max_blocks=max_blocks,
            free_blocks=max_blocks - used_blocks(),
            resident_prefix_blocks=resident_prefix_blocks,
            waiting=tuple(
                WaitingRequestMetadata(
                    specs[item],
                    wait_steps=wait_steps[item],
                    skip_count=skip_counts[item],
                    shared_prefix_blocks=shared_blocks[item],
                )
                for item in waiting
            ),
            active=tuple(
                ActiveRequestMetadata(
                    spec=specs[item],
                    seq_len=state["seq_len"],
                    remaining_tokens=specs[item].max_new_tokens - state["decoded"],
                    physical_blocks=_blocks(state["seq_len"], block_size),
                    committed_blocks=(
                        specs[item].commitment_blocks(block_size) - shared_blocks[item]
                    ),
                    service_wait_steps=state["service_wait"],
                    shared_prefix_blocks=shared_blocks[item],
                )
                for item, state in active.items()
            ),
        )
        decision = scheduler.plan(snapshot)
        for item in decision.admit_ids:
            waiting.remove(item)
            wait_steps.pop(item)
            skip_counts.pop(item)
            active[item] = {
                "seq_len": specs[item].initial_context_tokens,
                "decoded": 0,
                "service_wait": 0,
            }
        for item in decision.rejected_ids:
            waiting.remove(item)
            wait_steps.pop(item)
            skip_counts.pop(item)
            rejected.append(item)
        for item in decision.waiting_ids:
            wait_steps[item] += 1
            skip_counts[item] += 1

        positions = tuple(active[item]["seq_len"] for item in decision.runnable_ids)
        failure = failures_by_step.get(logical_step)
        aborted = bool(failure is not None and decision.runnable_ids)
        completed_ids = []
        if decision.runnable_ids:
            if aborted:
                aborted_steps += 1
                for item in (*decision.runnable_ids, *decision.deferred_ids):
                    active[item]["service_wait"] += 1
            else:
                successful_steps += 1
                completed_tokens += len(decision.runnable_ids)
                for item in decision.runnable_ids:
                    active[item]["seq_len"] += 1
                    active[item]["decoded"] += 1
                    active[item]["service_wait"] = 0
                for item in decision.deferred_ids:
                    active[item]["service_wait"] += 1
                for item in decision.runnable_ids:
                    if active[item]["decoded"] == specs[item].max_new_tokens:
                        completed_ids.append(item)
                for item in completed_ids:
                    del active[item]
                    completed.append(item)
        elif failure is not None:
            raise ValueError("failure step must have a runnable batch")

        used = used_blocks()
        committed = committed_blocks()
        steps.append(
            IntegratedReferenceStep(
                logical_step=logical_step,
                submitted_ids=submitted_ids,
                cancelled_ids=cancelled_ids,
                admitted_ids=tuple(decision.admit_ids),
                rejected_ids=tuple(decision.rejected_ids),
                runnable_ids=tuple(decision.runnable_ids),
                deferred_ids=tuple(decision.deferred_ids),
                completed_ids=tuple(completed_ids),
                positions=positions,
                aborted=aborted,
                committed_blocks=committed,
                used_blocks=used,
                free_blocks=max_blocks - used,
            )
        )
        if len(completed) + len(cancelled) + len(rejected) == len(specs):
            break
    else:
        raise RuntimeError("integrated reference did not resolve before max_steps")

    return IntegratedReferenceTrajectory(
        steps=tuple(steps),
        completed_request_ids=tuple(completed),
        cancelled_request_ids=tuple(cancelled),
        rejected_request_ids=tuple(rejected),
        successful_steps=successful_steps,
        aborted_steps=aborted_steps,
        completed_tokens=completed_tokens,
        digest=_trajectory_digest(steps),
    )


def run_integrated_workload(engine, config, *, num_q_heads, seed=0):
    """Execute and audit one dynamic R4-C trace on a fresh Engine."""
    from .engine import DecodeEngine

    if not isinstance(engine, DecodeEngine):
        raise TypeError("engine must be a DecodeEngine")
    if not isinstance(config, IntegratedWorkloadConfig):
        raise TypeError("config must be an IntegratedWorkloadConfig")
    _require_int("num_q_heads", num_q_heads, minimum=1)
    if engine.cache.num_layers != config.num_layers:
        raise ValueError("config num_layers must match the cache")
    if num_q_heads % engine.cache.num_kv_heads:
        raise ValueError("num_q_heads must be divisible by cache num_kv_heads")
    metrics = engine.metrics()
    if any(metrics[name] for name in (
        "waiting_requests", "active_requests", "finished_requests",
        "cancelled_requests", "rejected_requests",
    )):
        raise ValueError("run_integrated_workload requires a fresh DecodeEngine")
    cache_metrics = metrics["cache"]
    resident_prefix_blocks = cache_metrics["resident_prefix_blocks"]
    if resident_prefix_blocks <= 0:
        raise ValueError("R4-C requires a fixed resident shared prefix")
    if cache_metrics["used_blocks"] != resident_prefix_blocks:
        raise ValueError("fresh R4-C cache may only contain resident prefix blocks")

    prefix_ids = tuple(dict.fromkeys(
        item.spec.prefix_id
        for item in config.arrivals
        if item.spec.prefix_id is not None
    ))
    for prefix_id in prefix_ids:
        engine.cache.prefix_state(prefix_id)
    reference = build_integrated_reference(
        config,
        block_size=engine.cache.block_size,
        max_blocks=engine.cache.max_blocks,
        resident_prefix_blocks=resident_prefix_blocks,
    )

    torch = __import__("torch")
    generator = torch.Generator(device=engine.cache.device.type)
    generator.manual_seed(int(seed))
    specs = {item.spec.request_id: item.spec for item in config.arrivals}
    arrivals_by_step = {}
    for item in config.arrivals:
        arrivals_by_step.setdefault(item.arrival_step, []).append(item.spec)
    cancellations_by_step = {
        item.logical_step: item.request_id for item in config.cancellations
    }
    failures_by_step = {item.logical_step: item for item in config.failures}
    wait_steps = {}
    skip_counts = {}
    service_wait = {}
    decoded = {request_id: 0 for request_id in specs}
    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=config.max_active_requests,
            max_batch_requests=config.max_batch_requests,
            reserve_blocks=config.reserve_blocks,
            aging_threshold_steps=config.aging_threshold_steps,
        )
    )
    released_blocks = set()
    reused_blocks = set()
    observed_steps = []
    step_latencies = []
    scheduler_times = []
    context_times = []
    engine_times = []

    def synchronize():
        if engine.cache.device.type == "cuda":
            torch.cuda.synchronize(engine.cache.device)

    def random_kv_layers():
        shape = (
            engine.cache.num_layers,
            engine.cache.num_kv_heads,
            engine.cache.head_dim,
        )
        k = torch.randn(
            shape,
            device=engine.cache.device,
            dtype=engine.cache.dtype,
            generator=generator,
        )
        return k, torch.randn(
            shape,
            device=engine.cache.device,
            dtype=engine.cache.dtype,
            generator=generator,
        )

    def random_decode_inputs(batch_size):
        values = []
        for _ in range(engine.cache.num_layers):
            q = torch.randn(
                (batch_size, num_q_heads, engine.cache.head_dim),
                device=engine.cache.device,
                dtype=engine.cache.dtype,
                generator=generator,
            )
            k = torch.randn(
                (batch_size, engine.cache.num_kv_heads, engine.cache.head_dim),
                device=engine.cache.device,
                dtype=engine.cache.dtype,
                generator=generator,
            )
            v = torch.randn(
                (batch_size, engine.cache.num_kv_heads, engine.cache.head_dim),
                device=engine.cache.device,
                dtype=engine.cache.dtype,
                generator=generator,
            )
            values.append((q, k, v))
        return tuple(values)

    private_contexts = {}
    for request_id, spec in specs.items():
        if spec.prefix_id is None:
            private_contexts[request_id] = tuple(
                random_kv_layers() for _ in range(spec.initial_context_tokens)
            )
    decode_inputs = {
        step.logical_step: random_decode_inputs(len(step.runnable_ids))
        for step in reference.steps
        if step.runnable_ids
    }
    synchronize()

    def seed_context(request_id):
        spec = specs[request_id]
        if spec.prefix_id is not None:
            return
        for k_layers, v_layers in private_contexts[request_id]:
            engine.prefill_request_layers(request_id, k_layers, v_layers)

    for expected in reference.steps:
        logical_step = expected.logical_step
        for spec in arrivals_by_step.get(logical_step, ()):
            engine.submit_request(spec)
            wait_steps[spec.request_id] = 0
            skip_counts[spec.request_id] = 0

        cancelled_id = cancellations_by_step.get(logical_step)
        if cancelled_id is not None:
            released_blocks.update(engine.cancel_request(cancelled_id))
            service_wait.pop(cancelled_id)

        decision_start = time.perf_counter()
        snapshot = engine.scheduling_snapshot(
            logical_step,
            waiting_wait_steps=wait_steps,
            waiting_skip_counts=skip_counts,
            active_service_wait_steps=service_wait,
        )
        decision = scheduler.plan(snapshot)
        engine.apply_scheduler_decision(decision)
        scheduler_ms = (time.perf_counter() - decision_start) * 1_000.0
        if tuple(decision.admit_ids) != expected.admitted_ids:
            raise RuntimeError("Engine admission diverged from R4-C reference")
        if tuple(decision.rejected_ids) != expected.rejected_ids:
            raise RuntimeError("Engine rejection diverged from R4-C reference")
        for request_id in decision.admit_ids:
            wait_steps.pop(request_id)
            skip_counts.pop(request_id)
            service_wait[request_id] = 0
        for request_id in decision.rejected_ids:
            wait_steps.pop(request_id)
            skip_counts.pop(request_id)

        synchronize()
        context_start = time.perf_counter()
        for request_id in decision.admit_ids:
            seed_context(request_id)
            reused_blocks.update(
                set(engine.cache.request_block_ids(request_id)) & released_blocks
            )
        synchronize()
        context_ms = (time.perf_counter() - context_start) * 1_000.0

        execution_decision = decision
        seeded_private_context = any(
            specs[item].prefix_id is None
            and specs[item].initial_context_tokens > 0
            for item in decision.admit_ids
        )
        if seeded_private_context:
            decision_start = time.perf_counter()
            snapshot = engine.scheduling_snapshot(
                logical_step,
                waiting_wait_steps=wait_steps,
                waiting_skip_counts=skip_counts,
                active_service_wait_steps=service_wait,
            )
            execution_decision = scheduler.plan(snapshot)
            if execution_decision.admit_ids or execution_decision.rejected_ids:
                raise RuntimeError("context replan unexpectedly changed admission")
            engine.apply_scheduler_decision(execution_decision)
            scheduler_ms += (time.perf_counter() - decision_start) * 1_000.0
        if tuple(execution_decision.runnable_ids) != expected.runnable_ids:
            raise RuntimeError("Engine runnable batch diverged from R4-C reference")
        if tuple(execution_decision.deferred_ids) != expected.deferred_ids:
            raise RuntimeError("Engine deferred batch diverged from R4-C reference")
        for request_id in execution_decision.waiting_ids:
            wait_steps[request_id] += 1
            skip_counts[request_id] += 1

        inputs = decode_inputs.get(logical_step, ())
        synchronize()
        engine_start = time.perf_counter()
        observed_positions = ()
        observed_completed_ids = ()
        if expected.runnable_ids:
            before = {
                request_id: (
                    engine.cache.request_state(request_id)["seq_len"],
                    engine.cache.request_block_ids(request_id),
                )
                for request_id in expected.runnable_ids
            }
            transaction = engine.begin_step(expected.runnable_ids)
            observed_positions = tuple(transaction.positions.tolist())
            if observed_positions != expected.positions:
                raise RuntimeError("transaction positions diverged from R4-C reference")
            reused_blocks.update(
                set(transaction.physical_block_ids.tolist()) & released_blocks
            )
            failure = failures_by_step.get(logical_step)
            if failure is not None:
                for layer_idx, (q, k, v) in enumerate(inputs):
                    if layer_idx == failure.layer_idx:
                        q = q[..., :-1]
                    try:
                        engine.step_layer(transaction, layer_idx, q, k, v)
                    except ValueError:
                        if layer_idx != failure.layer_idx:
                            raise
                        break
                else:
                    raise RuntimeError("R4-C rollback injection did not fail")
                for request_id, state in before.items():
                    if (
                        engine.cache.request_state(request_id)["seq_len"],
                        engine.cache.request_block_ids(request_id),
                    ) != state:
                        raise RuntimeError("R4-C rollback changed visible request state")
                for request_id in (*expected.runnable_ids, *expected.deferred_ids):
                    service_wait[request_id] += 1
            else:
                for layer_idx, (q, k, v) in enumerate(inputs):
                    engine.step_layer(transaction, layer_idx, q, k, v)
                engine.commit_step(transaction)
                for request_id in expected.runnable_ids:
                    decoded[request_id] += 1
                    service_wait[request_id] = 0
                for request_id in expected.deferred_ids:
                    service_wait[request_id] += 1
                observed_completed_ids = tuple(
                    request_id
                    for request_id in expected.runnable_ids
                    if decoded[request_id] == specs[request_id].max_new_tokens
                )
                if observed_completed_ids != expected.completed_ids:
                    raise RuntimeError("completion trajectory diverged from R4-C reference")
                for request_id in observed_completed_ids:
                    released_blocks.update(engine.finish_request(request_id))
                    service_wait.pop(request_id)
        elif failures_by_step.get(logical_step) is not None:
            raise RuntimeError("R4-C failure step had no runnable batch")
        synchronize()
        engine_ms = (time.perf_counter() - engine_start) * 1_000.0

        current = engine.metrics()
        if current["committed_blocks"] != expected.committed_blocks:
            raise RuntimeError("committed-block trajectory diverged from R4-C reference")
        if engine.cache.num_used_blocks != expected.used_blocks:
            raise RuntimeError("physical-block trajectory diverged from R4-C reference")
        if engine.cache.num_free_blocks != expected.free_blocks:
            raise RuntimeError("free-block trajectory diverged from R4-C reference")
        engine.validate_invariants()
        observed_steps.append(
            IntegratedReferenceStep(
                logical_step=logical_step,
                submitted_ids=tuple(
                    spec.request_id for spec in arrivals_by_step.get(logical_step, ())
                ),
                cancelled_ids=(cancelled_id,) if cancelled_id is not None else (),
                admitted_ids=tuple(decision.admit_ids),
                rejected_ids=tuple(decision.rejected_ids),
                runnable_ids=tuple(execution_decision.runnable_ids),
                deferred_ids=tuple(execution_decision.deferred_ids),
                completed_ids=observed_completed_ids,
                positions=observed_positions,
                aborted=failures_by_step.get(logical_step) is not None,
                committed_blocks=current["committed_blocks"],
                used_blocks=engine.cache.num_used_blocks,
                free_blocks=engine.cache.num_free_blocks,
            )
        )
        scheduler_times.append(scheduler_ms)
        context_times.append(context_ms)
        engine_times.append(engine_ms)
        step_latencies.append(max(scheduler_ms + context_ms + engine_ms, 1e-9))

    terminal_metrics = engine.metrics()
    terminal_cache = terminal_metrics["cache"]
    if terminal_cache["active_prefix_references"] != 0:
        raise RuntimeError("terminal R4-C state retained active prefix references")
    if terminal_cache["used_blocks"] != resident_prefix_blocks:
        raise RuntimeError("terminal R4-C state retained request-private blocks")
    if terminal_cache["open_transaction_count"] != 0:
        raise RuntimeError("terminal R4-C state retained an open transaction")
    for prefix_id in prefix_ids:
        released_blocks.update(engine.evict_prefix(prefix_id))
    engine.validate_invariants()
    if engine.cache.num_free_blocks != engine.cache.max_blocks:
        raise RuntimeError("R4-C terminal cleanup did not restore the full block pool")
    if not reused_blocks:
        raise RuntimeError("R4-C trace did not demonstrate released-block reuse")
    observed_digest = _trajectory_digest(observed_steps)
    if observed_digest != reference.digest:
        raise RuntimeError("observed R4-C trajectory digest differs from reference")

    return IntegratedWorkloadResult(
        config=config,
        reference=reference,
        step_latencies_ms=tuple(step_latencies),
        scheduler_ms=tuple(scheduler_times),
        context_seed_ms=tuple(context_times),
        engine_ms=tuple(engine_times),
        completed_request_ids=reference.completed_request_ids,
        cancelled_request_ids=reference.cancelled_request_ids,
        rejected_request_ids=reference.rejected_request_ids,
        successful_steps=reference.successful_steps,
        aborted_steps=reference.aborted_steps,
        completed_tokens=reference.completed_tokens,
        block_reuse_count=len(reused_blocks),
        terminal_resident_prefix_blocks=resident_prefix_blocks,
        final_free_blocks=engine.cache.num_free_blocks,
        trajectory_digest=observed_digest,
        engine_metrics=engine.metrics(),
    )
