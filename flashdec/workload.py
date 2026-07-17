"""Synthetic dynamic workloads for the DecodeEngine single-layer API."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
import time
from typing import Any

from .benchmark import percentile


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class WorkloadConfig:
    """Deterministic request-arrival and lifecycle configuration."""

    name: str
    steps: int
    max_active: int
    arrivals_per_step: int
    decode_tokens_per_request: int
    initial_context_tokens: int = 0
    context_stagger_tokens: int = 0
    cancel_interval: int = 0
    cancel_probability: float = 0.0
    cancel_on_backpressure: bool = True

    def __post_init__(self):
        for name, value in [
            ("steps", self.steps),
            ("max_active", self.max_active),
            ("arrivals_per_step", self.arrivals_per_step),
            ("decode_tokens_per_request", self.decode_tokens_per_request),
        ]:
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("initial_context_tokens", self.initial_context_tokens),
            ("context_stagger_tokens", self.context_stagger_tokens),
        ]:
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.cancel_interval) < 0:
            raise ValueError("cancel_interval must be non-negative")
        cancel_probability = float(self.cancel_probability)
        if not math.isfinite(cancel_probability) or not 0.0 <= cancel_probability <= 1.0:
            raise ValueError("cancel_probability must be finite and in [0, 1]")


@dataclass(frozen=True)
class WorkloadResult:
    """Measured complete engine-step metrics for one synthetic workload."""

    config: WorkloadConfig
    latencies_ms: tuple[float, ...]
    successful_steps: int
    completed_tokens: int
    admitted_requests: int
    finished_requests: int
    cancelled_requests: int
    prefilled_tokens: int
    backpressure_steps: int
    active_batch_samples: tuple[int, ...]
    engine_metrics: dict[str, Any]

    @property
    def mean_ms(self):
        """Return the arithmetic mean complete-step latency."""
        return statistics.fmean(self.latencies_ms)

    @property
    def p50_ms(self):
        """Return median complete-step latency."""
        return percentile(self.latencies_ms, 50)

    @property
    def p90_ms(self):
        """Return p90 complete-step latency."""
        return percentile(self.latencies_ms, 90)

    @property
    def p99_ms(self):
        """Return p99 complete-step latency."""
        return percentile(self.latencies_ms, 99)

    @property
    def tokens_per_second(self):
        """Return completed decode tokens per measured second."""
        total_seconds = sum(self.latencies_ms) / 1_000.0
        return self.completed_tokens / total_seconds if total_seconds else 0.0

    @property
    def mean_active_batch(self):
        """Return the mean active batch size over measured steps."""
        return statistics.fmean(self.active_batch_samples) if self.active_batch_samples else 0.0

    @property
    def max_active_batch(self):
        """Return the maximum active batch size over measured steps."""
        return max(self.active_batch_samples, default=0)


def run_synthetic_workload(
    engine,
    config,
    num_q_heads,
    warmup_steps=5,
    seed=0,
):
    """Run a deterministic dynamic workload against a DecodeEngine.

    Wall-clock timing includes request submission/admission, ``DecodeEngine.step``,
    and the post-step finish/cancel policy, with CUDA synchronization before
    and after each measured step. It deliberately excludes external model
    projection, prompt prefill, and random Q/K/V input generation: those are
    outside the v1 decode-step engine boundary.

    A backpressured step cancels the oldest active request when configured,
    then retries naturally on a later logical step. This makes memory-pressure
    behavior observable without silently dropping allocator errors.
    """
    from .engine import DecodeEngine

    if not isinstance(engine, DecodeEngine):
        raise TypeError("engine must be a DecodeEngine")
    if not isinstance(config, WorkloadConfig):
        raise TypeError("config must be a WorkloadConfig")
    if int(num_q_heads) <= 0 or int(num_q_heads) % engine.cache.num_kv_heads != 0:
        raise ValueError("num_q_heads must be positive and divisible by cache num_kv_heads")
    if int(warmup_steps) < 0:
        raise ValueError("warmup_steps must be non-negative")

    torch = _torch()
    device = engine.cache.device
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    cancel_rng = random.Random(int(seed))
    remaining_tokens: dict[int, int] = {}
    next_request_id = 0
    latencies_ms = []
    active_batch_samples = []
    successful_steps = 0
    completed_tokens = 0
    admitted_requests = 0
    finished_requests = 0
    cancelled_requests = 0
    prefilled_tokens = 0
    backpressure_steps = 0

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def prefill_request(request_id, token_count):
        """Seed a request context outside the measured decode-step boundary."""
        if token_count == 0:
            return
        k = torch.zeros(
            (1, engine.cache.num_kv_heads, engine.cache.head_dim),
            device=device,
            dtype=engine.cache.dtype,
        )
        v = torch.zeros_like(k)
        for _ in range(token_count):
            engine.cache.append(0, [request_id], k, v)

    total_steps = int(warmup_steps) + config.steps
    for logical_step in range(total_steps):
        measured = logical_step >= warmup_steps
        free_slots = config.max_active - len(engine.active_request_ids())
        arrivals = min(config.arrivals_per_step, max(0, free_slots))
        admission_elapsed_ms = 0.0
        admitted_this_step = 0
        prefilled_this_step = 0
        for _ in range(arrivals):
            request_id = next_request_id
            next_request_id += 1
            admission_start = time.perf_counter() if measured else None
            engine.add_request(request_id)
            engine.admit([request_id])
            if measured:
                admission_elapsed_ms += (time.perf_counter() - admission_start) * 1_000.0
            remaining_tokens[request_id] = config.decode_tokens_per_request
            admitted_this_step += 1
            prefill_tokens = config.initial_context_tokens + (
                request_id % config.max_active
            ) * config.context_stagger_tokens
            prefill_request(request_id, prefill_tokens)
            prefilled_this_step += prefill_tokens

        request_ids = engine.active_request_ids()
        if not request_ids:
            continue
        batch_size = len(request_ids)
        q = torch.randn(
            (batch_size, int(num_q_heads), engine.cache.head_dim),
            device=device,
            dtype=engine.cache.dtype,
            generator=generator,
        )
        k = torch.randn(
            (batch_size, engine.cache.num_kv_heads, engine.cache.head_dim),
            device=device,
            dtype=engine.cache.dtype,
            generator=generator,
        )
        v = torch.randn(
            (batch_size, engine.cache.num_kv_heads, engine.cache.head_dim),
            device=device,
            dtype=engine.cache.dtype,
            generator=generator,
        )
        synchronize()

        start = time.perf_counter()
        step_result = engine.step(q, k, v, request_ids=request_ids)
        successful_this_step = 0
        completed_this_step = 0
        finished_this_step = 0
        cancelled_this_step = 0
        backpressure_this_step = 0
        if step_result.status == engine.STEP_BACKPRESSURE:
            backpressure_this_step = 1
            if config.cancel_on_backpressure and request_ids:
                victim = request_ids[0]
                engine.cancel_request(victim)
                remaining_tokens.pop(victim, None)
                cancelled_this_step += 1
        else:
            successful_this_step = 1
            completed_this_step = batch_size
            for request_id in request_ids:
                remaining_tokens[request_id] -= 1
            for request_id in request_ids:
                if remaining_tokens[request_id] == 0:
                    engine.finish_request(request_id)
                    remaining_tokens.pop(request_id)
                    finished_this_step += 1
            if config.cancel_interval and (logical_step + 1) % config.cancel_interval == 0:
                remaining = engine.active_request_ids()
                if remaining:
                    victim = remaining[0]
                    engine.cancel_request(victim)
                    remaining_tokens.pop(victim)
                    cancelled_this_step += 1
            if config.cancel_probability:
                for victim in engine.active_request_ids():
                    if cancel_rng.random() < config.cancel_probability:
                        engine.cancel_request(victim)
                        remaining_tokens.pop(victim)
                        cancelled_this_step += 1
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1_000.0

        if measured:
            latencies_ms.append(admission_elapsed_ms + elapsed_ms)
            active_batch_samples.append(batch_size)
            successful_steps += successful_this_step
            completed_tokens += completed_this_step
            admitted_requests += admitted_this_step
            finished_requests += finished_this_step
            cancelled_requests += cancelled_this_step
            prefilled_tokens += prefilled_this_step
            backpressure_steps += backpressure_this_step

    if not latencies_ms:
        raise RuntimeError("workload produced no measured decode steps")
    engine.validate_invariants()
    return WorkloadResult(
        config=config,
        latencies_ms=tuple(latencies_ms),
        successful_steps=successful_steps,
        completed_tokens=completed_tokens,
        admitted_requests=admitted_requests,
        finished_requests=finished_requests,
        cancelled_requests=cancelled_requests,
        prefilled_tokens=prefilled_tokens,
        backpressure_steps=backpressure_steps,
        active_batch_samples=tuple(active_batch_samples),
        engine_metrics=engine.metrics(),
    )
