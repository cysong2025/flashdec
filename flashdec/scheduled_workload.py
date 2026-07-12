"""Trace-driven scheduler policy workloads for the FlashDec decode runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any

from .benchmark import percentile
from .scheduler import (
    LIFETIME_FIFO_AGING,
    BlockAwareScheduler,
    RequestSpec,
    SchedulerConfig,
)


CANCEL_ON_BACKPRESSURE = "cancel_on_backpressure"
GREEDY_STEP_ONLY = "greedy_step_only"
SCHEDULER_POLICIES = (
    CANCEL_ON_BACKPRESSURE,
    GREEDY_STEP_ONLY,
    LIFETIME_FIFO_AGING,
)


def _require_int(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class RequestArrival:
    """One immutable request specification and its logical arrival step."""

    spec: RequestSpec
    arrival_step: int

    def __post_init__(self):
        if not isinstance(self.spec, RequestSpec):
            raise TypeError("spec must be a RequestSpec")
        _require_int("arrival_step", self.arrival_step)


@dataclass(frozen=True)
class SchedulerWorkloadConfig:
    """Finite request trace and one scheduler policy configuration."""

    name: str
    arrivals: tuple[RequestArrival, ...]
    policy: str
    max_active_requests: int
    max_batch_requests: int
    reserve_blocks: int = 0
    aging_threshold_steps: int = 8
    max_steps: int = 1_000
    max_stalled_steps: int = 8

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        arrivals = tuple(self.arrivals)
        if not arrivals or not all(isinstance(item, RequestArrival) for item in arrivals):
            raise ValueError("arrivals must contain at least one RequestArrival")
        object.__setattr__(self, "arrivals", arrivals)
        if self.policy not in SCHEDULER_POLICIES:
            raise ValueError(f"policy must be one of {SCHEDULER_POLICIES}")
        _require_int("max_active_requests", self.max_active_requests, minimum=1)
        _require_int("max_batch_requests", self.max_batch_requests, minimum=1)
        _require_int("reserve_blocks", self.reserve_blocks)
        _require_int("aging_threshold_steps", self.aging_threshold_steps, minimum=1)
        _require_int("max_steps", self.max_steps, minimum=1)
        _require_int("max_stalled_steps", self.max_stalled_steps, minimum=1)
        if self.max_batch_requests > self.max_active_requests:
            raise ValueError("max_batch_requests must not exceed max_active_requests")
        request_ids = [item.spec.request_id for item in arrivals]
        orders = [item.spec.submission_order for item in arrivals]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request ids must be unique across arrivals")
        if len(orders) != len(set(orders)):
            raise ValueError("submission_order values must be unique across arrivals")
        if max(item.arrival_step for item in arrivals) >= self.max_steps:
            raise ValueError("every arrival_step must be smaller than max_steps")


@dataclass(frozen=True)
class SchedulerWorkloadResult:
    """Correctness, progress, fairness, memory, and latency observations."""

    config: SchedulerWorkloadConfig
    step_latencies_ms: tuple[float, ...]
    scheduler_decision_ms: tuple[float, ...]
    completed_request_ids: tuple[Any, ...]
    cancelled_request_ids: tuple[Any, ...]
    rejected_request_ids: tuple[Any, ...]
    completed_tokens: int
    useful_tokens: int
    successful_steps: int
    backpressure_steps: int
    stalled_steps: int
    resource_deadlock_count: int
    forced_cancellation_count: int
    waiting_depth_samples: tuple[int, ...]
    active_batch_samples: tuple[int, ...]
    admission_wait_steps: tuple[int, ...]
    max_service_wait_steps: int
    committed_block_samples: tuple[int, ...]
    physical_block_samples: tuple[int, ...]
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

    @property
    def useful_tokens_per_second(self):
        seconds = sum(self.step_latencies_ms) / 1_000.0
        return self.useful_tokens / seconds if seconds else 0.0

    @property
    def completion_rate(self):
        return len(self.completed_request_ids) / len(self.config.arrivals)

    @property
    def mean_waiting_depth(self):
        return statistics.fmean(self.waiting_depth_samples)

    @property
    def max_waiting_depth(self):
        return max(self.waiting_depth_samples, default=0)

    @property
    def admission_wait_p50(self):
        return percentile(self.admission_wait_steps, 50) if self.admission_wait_steps else 0

    @property
    def admission_wait_p90(self):
        return percentile(self.admission_wait_steps, 90) if self.admission_wait_steps else 0


def boundary_deadlock_arrivals(*, num_requests=2, max_new_tokens=4):
    """Build simultaneous requests that cross the same block boundary."""
    _require_int("num_requests", num_requests, minimum=1)
    _require_int("max_new_tokens", max_new_tokens, minimum=1)
    return tuple(
        RequestArrival(
            RequestSpec(
                request_id=request_id,
                initial_context_tokens=0,
                max_new_tokens=max_new_tokens,
                submission_order=request_id,
            ),
            arrival_step=0,
        )
        for request_id in range(num_requests)
    )


def run_scheduler_workload(engine, config, num_q_heads, seed=0):
    """Run one finite trace through cancel, greedy, or lifetime scheduling.

    Q/K/V generation and prompt-context writes are excluded from timing.
    Each recorded logical-step latency is scheduler decision wall time plus
    Engine step wall time, so scheduler overhead remains visible without
    mixing random input generation or prefill into the decode boundary.
    """
    from .engine import DecodeEngine

    if not isinstance(engine, DecodeEngine):
        raise TypeError("engine must be a DecodeEngine")
    if not isinstance(config, SchedulerWorkloadConfig):
        raise TypeError("config must be a SchedulerWorkloadConfig")
    _require_int("num_q_heads", num_q_heads, minimum=1)
    if num_q_heads % engine.cache.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by cache num_kv_heads")
    if config.reserve_blocks >= engine.cache.max_blocks:
        raise ValueError("reserve_blocks must be smaller than cache max_blocks")
    metrics = engine.metrics()
    if any(
        metrics[name]
        for name in (
            "waiting_requests",
            "active_requests",
            "finished_requests",
            "cancelled_requests",
            "rejected_requests",
        )
    ) or engine.cache.num_used_blocks:
        raise ValueError("run_scheduler_workload requires a fresh DecodeEngine")

    torch = _torch()
    generator = torch.Generator(device=engine.cache.device.type)
    generator.manual_seed(int(seed))
    arrivals = sorted(
        config.arrivals,
        key=lambda item: (item.arrival_step, item.spec.submission_order),
    )
    arrivals_by_step = {}
    specs = {}
    arrival_steps = {}
    for arrival in arrivals:
        arrivals_by_step.setdefault(arrival.arrival_step, []).append(arrival.spec)
        specs[arrival.spec.request_id] = arrival.spec
        arrival_steps[arrival.spec.request_id] = arrival.arrival_step

    waiting_ids = []
    active_ids = []
    wait_steps = {}
    skip_counts = {}
    service_wait_steps = {}
    decoded_tokens = {request_id: 0 for request_id in specs}
    completed_ids = []
    cancelled_ids = []
    rejected_ids = []
    admission_wait = []
    step_latencies = []
    decision_latencies = []
    waiting_samples = []
    active_samples = []
    committed_samples = []
    physical_samples = []
    completed_tokens = 0
    successful_steps = 0
    backpressure_steps = 0
    stalled_steps = 0
    resource_deadlocks = 0
    forced_cancellations = 0
    consecutive_stalls = 0
    max_service_wait = 0

    scheduler = None
    if config.policy == LIFETIME_FIFO_AGING:
        scheduler = BlockAwareScheduler(
            SchedulerConfig(
                max_active_requests=config.max_active_requests,
                max_batch_requests=config.max_batch_requests,
                reserve_blocks=config.reserve_blocks,
                aging_threshold_steps=config.aging_threshold_steps,
            )
        )

    def synchronize():
        if engine.cache.device.type == "cuda":
            torch.cuda.synchronize(engine.cache.device)

    def zero_kv():
        k = torch.zeros(
            (1, engine.cache.num_kv_heads, engine.cache.head_dim),
            device=engine.cache.device,
            dtype=engine.cache.dtype,
        )
        return k, torch.zeros_like(k)

    def seed_lifetime_context(request_id):
        k, v = zero_kv()
        for _ in range(specs[request_id].initial_context_tokens):
            engine.prefill_request(request_id, k, v)

    def seed_legacy_context(request_id):
        k, v = zero_kv()
        for _ in range(specs[request_id].initial_context_tokens):
            engine.cache.append(0, [request_id], k, v)

    def tensor_inputs(request_ids):
        batch_size = len(request_ids)
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
        return q, k, v

    def apply_lifetime_policy(logical_step):
        nonlocal max_service_wait
        decision_start = time.perf_counter()
        snapshot = engine.scheduling_snapshot(
            logical_step,
            waiting_wait_steps=wait_steps,
            waiting_skip_counts=skip_counts,
            active_service_wait_steps=service_wait_steps,
        )
        decision = scheduler.plan(snapshot)
        decision_ms = (time.perf_counter() - decision_start) * 1_000.0
        engine.apply_scheduler_decision(decision)

        admitted = tuple(decision.admit_ids)
        rejected = tuple(decision.rejected_ids)
        for request_id in admitted:
            waiting_ids.remove(request_id)
            wait_steps.pop(request_id, None)
            skip_counts.pop(request_id, None)
            active_ids.append(request_id)
            service_wait_steps[request_id] = 0
            admission_wait.append(logical_step - arrival_steps[request_id])
        for request_id in rejected:
            waiting_ids.remove(request_id)
            wait_steps.pop(request_id, None)
            skip_counts.pop(request_id, None)
            rejected_ids.append(request_id)

        needs_replan = any(specs[item].initial_context_tokens for item in admitted)
        for request_id in admitted:
            seed_lifetime_context(request_id)
        if needs_replan:
            decision_start = time.perf_counter()
            snapshot = engine.scheduling_snapshot(
                logical_step,
                waiting_wait_steps=wait_steps,
                waiting_skip_counts=skip_counts,
                active_service_wait_steps=service_wait_steps,
            )
            decision = scheduler.plan(snapshot)
            decision_ms += (time.perf_counter() - decision_start) * 1_000.0
            if decision.admit_ids or decision.rejected_ids:
                raise RuntimeError("context replan unexpectedly changed admission")
            engine.apply_scheduler_decision(decision)

        for request_id in decision.waiting_ids:
            wait_steps[request_id] += 1
            skip_counts[request_id] += 1
        max_service_wait = max(
            max_service_wait,
            max(service_wait_steps.values(), default=0),
        )
        return tuple(decision.runnable_ids), tuple(decision.deferred_ids), decision_ms

    def apply_legacy_policy(logical_step):
        nonlocal max_service_wait
        decision_start = time.perf_counter()
        capacity = engine.cache.max_blocks - config.reserve_blocks
        permanently_rejected = [
            request_id
            for request_id in waiting_ids
            if specs[request_id].commitment_blocks(engine.cache.block_size) > capacity
        ]
        for request_id in permanently_rejected:
            waiting_ids.remove(request_id)
            wait_steps.pop(request_id, None)
            rejected_ids.append(request_id)

        slots = config.max_active_requests - len(active_ids)
        block_budget = engine.cache.num_free_blocks - config.reserve_blocks
        admitted = []
        for request_id in list(waiting_ids):
            if slots == 0:
                break
            spec = specs[request_id]
            initial_blocks = math.ceil(spec.initial_context_tokens / engine.cache.block_size)
            needs_decode_block = spec.initial_context_tokens % engine.cache.block_size == 0
            needed_now = initial_blocks + int(needs_decode_block)
            if needed_now <= block_budget:
                admitted.append(request_id)
                block_budget -= needed_now
                slots -= 1
        for request_id in admitted:
            waiting_ids.remove(request_id)
            wait_steps.pop(request_id, None)
            engine.add_request(request_id)
        if admitted:
            engine.admit(admitted)
        for request_id in admitted:
            active_ids.append(request_id)
            service_wait_steps[request_id] = 0
            admission_wait.append(logical_step - arrival_steps[request_id])
            seed_legacy_context(request_id)

        for request_id in waiting_ids:
            wait_steps[request_id] += 1
        ordered_active = sorted(
            active_ids,
            key=lambda item: (-service_wait_steps[item], specs[item].submission_order),
        )
        runnable = tuple(ordered_active[: config.max_batch_requests])
        deferred = tuple(ordered_active[config.max_batch_requests :])
        max_service_wait = max(
            max_service_wait,
            max(service_wait_steps.values(), default=0),
        )
        decision_ms = (time.perf_counter() - decision_start) * 1_000.0
        return runnable, deferred, decision_ms

    for logical_step in range(config.max_steps):
        for spec in arrivals_by_step.get(logical_step, ()):
            request_id = spec.request_id
            waiting_ids.append(request_id)
            wait_steps[request_id] = 0
            skip_counts[request_id] = 0
            if config.policy == LIFETIME_FIFO_AGING:
                engine.submit_request(spec)

        if (
            config.policy == LIFETIME_FIFO_AGING
            and engine.metrics()["scheduler_managed"]
        ):
            runnable, deferred, decision_ms = apply_lifetime_policy(logical_step)
        elif config.policy == LIFETIME_FIFO_AGING:
            runnable, deferred, decision_ms = (), (), 0.0
        else:
            runnable, deferred, decision_ms = apply_legacy_policy(logical_step)
        decision_latencies.append(decision_ms)

        engine_elapsed_ms = 0.0
        made_progress = False
        if runnable:
            q, k, v = tensor_inputs(runnable)
            synchronize()
            engine_start = time.perf_counter()
            result = engine.step(q, k, v, request_ids=runnable)
            synchronize()
            engine_elapsed_ms = (time.perf_counter() - engine_start) * 1_000.0
            if result.status == engine.STEP_BACKPRESSURE:
                backpressure_steps += 1
                stalled_steps += 1
                consecutive_stalls += 1
                if config.policy == CANCEL_ON_BACKPRESSURE:
                    victim = min(active_ids, key=lambda item: specs[item].submission_order)
                    engine.cancel_request(victim)
                    active_ids.remove(victim)
                    service_wait_steps.pop(victim, None)
                    cancelled_ids.append(victim)
                    forced_cancellations += 1
                    consecutive_stalls = 0
                for request_id in active_ids:
                    service_wait_steps[request_id] += 1
            else:
                made_progress = True
                consecutive_stalls = 0
                successful_steps += 1
                completed_tokens += len(runnable)
                for request_id in runnable:
                    decoded_tokens[request_id] += 1
                for request_id in tuple(runnable):
                    if decoded_tokens[request_id] == specs[request_id].max_new_tokens:
                        engine.finish_request(request_id)
                        active_ids.remove(request_id)
                        service_wait_steps.pop(request_id, None)
                        completed_ids.append(request_id)
                for request_id in runnable:
                    if request_id in service_wait_steps:
                        service_wait_steps[request_id] = 0
                for request_id in deferred:
                    if request_id in service_wait_steps:
                        service_wait_steps[request_id] += 1
        else:
            arrived_unresolved = bool(waiting_ids or active_ids)
            if arrived_unresolved:
                stalled_steps += 1
                consecutive_stalls += 1
                for request_id in active_ids:
                    service_wait_steps[request_id] += 1

        if (
            config.policy == GREEDY_STEP_ONLY
            and consecutive_stalls >= config.max_stalled_steps
        ):
            resource_deadlocks = 1

        step_latencies.append(max(decision_ms + engine_elapsed_ms, 1e-9))
        waiting_samples.append(len(waiting_ids))
        active_samples.append(len(runnable))
        if config.policy == LIFETIME_FIFO_AGING:
            committed_samples.append(engine.metrics()["committed_blocks"])
        else:
            committed_samples.append(0)
        physical_samples.append(engine.cache.num_used_blocks)
        max_service_wait = max(
            max_service_wait,
            max(service_wait_steps.values(), default=0),
        )

        resolved = len(completed_ids) + len(cancelled_ids) + len(rejected_ids)
        if resolved == len(arrivals):
            break
        if resource_deadlocks:
            break
        if not made_progress and logical_step == config.max_steps - 1:
            if config.policy != GREEDY_STEP_ONLY:
                raise RuntimeError("finite scheduler workload did not resolve before max_steps")
            resource_deadlocks = 1

    engine.validate_invariants()
    return SchedulerWorkloadResult(
        config=config,
        step_latencies_ms=tuple(step_latencies),
        scheduler_decision_ms=tuple(decision_latencies),
        completed_request_ids=tuple(completed_ids),
        cancelled_request_ids=tuple(cancelled_ids),
        rejected_request_ids=tuple(rejected_ids),
        completed_tokens=completed_tokens,
        useful_tokens=sum(specs[item].max_new_tokens for item in completed_ids),
        successful_steps=successful_steps,
        backpressure_steps=backpressure_steps,
        stalled_steps=stalled_steps,
        resource_deadlock_count=resource_deadlocks,
        forced_cancellation_count=forced_cancellations,
        waiting_depth_samples=tuple(waiting_samples),
        active_batch_samples=tuple(active_samples),
        admission_wait_steps=tuple(admission_wait),
        max_service_wait_steps=max_service_wait,
        committed_block_samples=tuple(committed_samples),
        physical_block_samples=tuple(physical_samples),
        engine_metrics=engine.metrics(),
    )
