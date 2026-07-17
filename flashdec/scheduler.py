"""Pure block-aware scheduling policy for the FlashDec decode runtime.

This module intentionally has no torch dependency. It plans logical request
admission and runnable rows from immutable metadata; DecodeEngine and
PagedKVCache integration remain separate so planning cannot mutate KV state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


LIFETIME_FIFO_AGING = "lifetime_fifo_aging"


def _require_non_bool_int(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")


def _require_request_id(request_id):
    if request_id is None:
        raise ValueError("request_id must not be None")
    try:
        hash(request_id)
    except TypeError as exc:
        raise ValueError("request_id must be hashable") from exc


def _blocks_for_tokens(token_count, block_size):
    _require_non_bool_int("token_count", token_count)
    _require_non_bool_int("block_size", block_size, minimum=1)
    return (token_count + block_size - 1) // block_size


@dataclass(frozen=True)
class RequestSpec:
    """Immutable lifetime information supplied when a request is submitted."""

    request_id: Hashable
    initial_context_tokens: int
    max_new_tokens: int
    submission_order: int

    def __post_init__(self):
        _require_request_id(self.request_id)
        _require_non_bool_int("initial_context_tokens", self.initial_context_tokens)
        _require_non_bool_int("max_new_tokens", self.max_new_tokens, minimum=1)
        _require_non_bool_int("submission_order", self.submission_order)

    def commitment_blocks(self, block_size):
        """Return the maximum physical blocks needed over this request lifetime."""
        return _blocks_for_tokens(
            self.initial_context_tokens + self.max_new_tokens,
            block_size,
        )


@dataclass(frozen=True)
class WaitingRequestMetadata:
    """Scheduler-owned waiting counters paired with an immutable request spec."""

    spec: RequestSpec
    wait_steps: int = 0
    skip_count: int = 0

    def __post_init__(self):
        if not isinstance(self.spec, RequestSpec):
            raise TypeError("spec must be a RequestSpec")
        _require_non_bool_int("wait_steps", self.wait_steps)
        _require_non_bool_int("skip_count", self.skip_count)

    @property
    def request_id(self):
        """Return the request id carried by the immutable specification."""
        return self.spec.request_id


@dataclass(frozen=True)
class ActiveRequestMetadata:
    """Scheduling metadata for an active request, without K/V tensors or ids."""

    spec: RequestSpec
    seq_len: int
    remaining_tokens: int
    physical_blocks: int
    committed_blocks: int
    service_wait_steps: int = 0

    def __post_init__(self):
        if not isinstance(self.spec, RequestSpec):
            raise TypeError("spec must be a RequestSpec")
        _require_non_bool_int("seq_len", self.seq_len)
        _require_non_bool_int("remaining_tokens", self.remaining_tokens, minimum=1)
        _require_non_bool_int("physical_blocks", self.physical_blocks)
        _require_non_bool_int("committed_blocks", self.committed_blocks, minimum=1)
        _require_non_bool_int("service_wait_steps", self.service_wait_steps)
        if self.remaining_tokens > self.spec.max_new_tokens:
            raise ValueError("remaining_tokens must not exceed max_new_tokens")
        expected_seq_len = self.spec.initial_context_tokens + (
            self.spec.max_new_tokens - self.remaining_tokens
        )
        if self.seq_len != expected_seq_len:
            raise ValueError(
                "seq_len must equal initial_context_tokens plus completed decode tokens"
            )
        if self.physical_blocks > self.committed_blocks:
            raise ValueError("physical_blocks must not exceed committed_blocks")

    @property
    def request_id(self):
        """Return the request id carried by the immutable specification."""
        return self.spec.request_id


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the deterministic lifetime-reservation policy."""

    max_active_requests: int
    max_batch_requests: int
    reserve_blocks: int = 0
    aging_threshold_steps: int = 8
    policy: str = LIFETIME_FIFO_AGING

    def __post_init__(self):
        _require_non_bool_int("max_active_requests", self.max_active_requests, minimum=1)
        _require_non_bool_int("max_batch_requests", self.max_batch_requests, minimum=1)
        _require_non_bool_int("reserve_blocks", self.reserve_blocks)
        _require_non_bool_int("aging_threshold_steps", self.aging_threshold_steps, minimum=1)
        if self.max_batch_requests > self.max_active_requests:
            raise ValueError("max_batch_requests must not exceed max_active_requests")
        if self.policy != LIFETIME_FIFO_AGING:
            raise ValueError(f"policy must be {LIFETIME_FIFO_AGING!r}")


@dataclass(frozen=True)
class SchedulingSnapshot:
    """Immutable Engine/Cache metadata consumed by one scheduler decision."""

    state_version: int
    logical_step: int
    block_size: int
    max_blocks: int
    free_blocks: int
    waiting: tuple[WaitingRequestMetadata, ...] = ()
    active: tuple[ActiveRequestMetadata, ...] = ()

    def __post_init__(self):
        _require_non_bool_int("state_version", self.state_version)
        _require_non_bool_int("logical_step", self.logical_step)
        _require_non_bool_int("block_size", self.block_size, minimum=1)
        _require_non_bool_int("max_blocks", self.max_blocks, minimum=1)
        _require_non_bool_int("free_blocks", self.free_blocks)
        if self.free_blocks > self.max_blocks:
            raise ValueError("free_blocks must not exceed max_blocks")

        waiting = tuple(self.waiting)
        active = tuple(self.active)
        if not all(isinstance(item, WaitingRequestMetadata) for item in waiting):
            raise TypeError("waiting must contain WaitingRequestMetadata values")
        if not all(isinstance(item, ActiveRequestMetadata) for item in active):
            raise TypeError("active must contain ActiveRequestMetadata values")
        object.__setattr__(self, "waiting", waiting)
        object.__setattr__(self, "active", active)

        request_ids = [item.request_id for item in waiting] + [item.request_id for item in active]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("waiting and active request_ids must be globally unique")
        submission_orders = [item.spec.submission_order for item in waiting] + [
            item.spec.submission_order for item in active
        ]
        if len(submission_orders) != len(set(submission_orders)):
            raise ValueError("submission_order values must be globally unique")

        for item in active:
            expected_commitment = item.spec.commitment_blocks(self.block_size)
            if item.committed_blocks != expected_commitment:
                raise ValueError("active committed_blocks must match the request lifetime")
            expected_physical = _blocks_for_tokens(item.seq_len, self.block_size)
            if item.physical_blocks != expected_physical:
                raise ValueError("active physical_blocks must match seq_len")

        used_blocks = self.max_blocks - self.free_blocks
        active_physical_blocks = sum(item.physical_blocks for item in active)
        if active_physical_blocks != used_blocks:
            raise ValueError(
                "active physical_blocks must exactly match max_blocks - free_blocks"
            )


@dataclass(frozen=True)
class SchedulerDecision:
    """Pure scheduling output; applying it is a separate Engine operation."""

    snapshot_version: int
    admit_ids: tuple[Hashable, ...]
    runnable_ids: tuple[Hashable, ...]
    deferred_ids: tuple[Hashable, ...]
    waiting_ids: tuple[Hashable, ...]
    rejected_ids: tuple[Hashable, ...]
    committed_blocks_before: int
    committed_blocks_after: int
    needed_physical_blocks_now: int
    free_blocks_before_step: int
    drain_for_request_id: Hashable | None = None
    reasons: tuple[str, ...] = ()


class BlockAwareScheduler:
    """Stateless planner using lifetime block commitments and FIFO + aging."""

    _REASON_ORDER = (
        "request_exceeds_capacity",
        "aging_drain",
        "active_limit",
        "capacity_deferred",
        "admitted",
        "batch_limit",
        "idle",
    )

    def __init__(self, config):
        if not isinstance(config, SchedulerConfig):
            raise TypeError("config must be a SchedulerConfig")
        self.config = config

    def plan(self, snapshot):
        """Plan one admission/runnable decision without mutating runtime state."""
        if not isinstance(snapshot, SchedulingSnapshot):
            raise TypeError("snapshot must be a SchedulingSnapshot")

        config = self.config
        if config.reserve_blocks >= snapshot.max_blocks:
            raise ValueError("reserve_blocks must be smaller than snapshot.max_blocks")
        if len(snapshot.active) > config.max_active_requests:
            raise ValueError("snapshot active requests exceed max_active_requests")

        schedulable_blocks = snapshot.max_blocks - config.reserve_blocks
        committed_before = sum(item.committed_blocks for item in snapshot.active)
        if committed_before > schedulable_blocks:
            raise ValueError("active commitments exceed schedulable block capacity")

        reasons = set()
        ordered_waiting = sorted(
            snapshot.waiting,
            key=lambda item: item.spec.submission_order,
        )
        rejected = [
            item
            for item in ordered_waiting
            if item.spec.commitment_blocks(snapshot.block_size) > schedulable_blocks
        ]
        rejected_ids = {item.request_id for item in rejected}
        eligible = [item for item in ordered_waiting if item.request_id not in rejected_ids]
        if rejected:
            reasons.add("request_exceeds_capacity")

        remaining_capacity = schedulable_blocks - committed_before
        remaining_slots = config.max_active_requests - len(snapshot.active)
        admitted = []
        drain_for_request_id = None

        aged = [
            item
            for item in eligible
            if item.wait_steps >= config.aging_threshold_steps
        ]
        barrier = aged[0] if aged else None
        if barrier is not None:
            barrier_blocks = barrier.spec.commitment_blocks(snapshot.block_size)
            if remaining_slots == 0 or barrier_blocks > remaining_capacity:
                drain_for_request_id = barrier.request_id
                reasons.add("aging_drain")
            else:
                admitted.append(barrier)
                remaining_slots -= 1
                remaining_capacity -= barrier_blocks

        if drain_for_request_id is None:
            for item in eligible:
                if item is barrier:
                    continue
                if remaining_slots == 0:
                    reasons.add("active_limit")
                    break
                item_blocks = item.spec.commitment_blocks(snapshot.block_size)
                if item_blocks <= remaining_capacity:
                    admitted.append(item)
                    remaining_slots -= 1
                    remaining_capacity -= item_blocks
                else:
                    reasons.add("capacity_deferred")

        admitted_ids = {item.request_id for item in admitted}
        waiting_after = [
            item
            for item in eligible
            if item.request_id not in admitted_ids
        ]
        if admitted:
            reasons.add("admitted")

        admitted_active = [
            ActiveRequestMetadata(
                spec=item.spec,
                seq_len=item.spec.initial_context_tokens,
                remaining_tokens=item.spec.max_new_tokens,
                physical_blocks=_blocks_for_tokens(
                    item.spec.initial_context_tokens,
                    snapshot.block_size,
                ),
                committed_blocks=item.spec.commitment_blocks(snapshot.block_size),
                service_wait_steps=0,
            )
            for item in admitted
        ]
        runnable_candidates = list(snapshot.active) + admitted_active
        runnable_candidates.sort(
            key=lambda item: (-item.service_wait_steps, item.spec.submission_order)
        )
        runnable = runnable_candidates[: config.max_batch_requests]
        deferred = runnable_candidates[config.max_batch_requests :]
        if deferred:
            reasons.add("batch_limit")
        if not runnable:
            reasons.add("idle")

        needed_physical_blocks_now = sum(
            item.physical_blocks for item in admitted_active
        )
        for item in runnable:
            seq_len_before_append = item.seq_len
            if seq_len_before_append % snapshot.block_size == 0:
                needed_physical_blocks_now += 1

        committed_after = committed_before + sum(
            item.committed_blocks for item in admitted_active
        )
        if committed_after > schedulable_blocks:
            raise RuntimeError("scheduler produced an overcommitted decision")
        if needed_physical_blocks_now > snapshot.free_blocks:
            raise RuntimeError(
                "lifetime commitments did not cover immediate physical allocation"
            )

        ordered_reasons = tuple(
            reason for reason in self._REASON_ORDER if reason in reasons
        )
        return SchedulerDecision(
            snapshot_version=snapshot.state_version,
            admit_ids=tuple(item.request_id for item in admitted),
            runnable_ids=tuple(item.request_id for item in runnable),
            deferred_ids=tuple(item.request_id for item in deferred),
            waiting_ids=tuple(item.request_id for item in waiting_after),
            rejected_ids=tuple(item.request_id for item in rejected),
            committed_blocks_before=committed_before,
            committed_blocks_after=committed_after,
            needed_physical_blocks_now=needed_physical_blocks_now,
            free_blocks_before_step=snapshot.free_blocks,
            drain_for_request_id=drain_for_request_id,
            reasons=ordered_reasons,
        )
