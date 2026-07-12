"""Pure-Python correctness coverage for Block-aware Scheduler v2 R1-A."""

import pytest

import flashdec
from flashdec.scheduler import (
    ActiveRequestMetadata,
    BlockAwareScheduler,
    RequestSpec,
    SchedulerConfig,
    SchedulerDecision,
    SchedulingSnapshot,
    WaitingRequestMetadata,
)


def _waiting(request_id, order, initial=0, max_new=4, wait_steps=0, skip_count=0):
    return WaitingRequestMetadata(
        RequestSpec(request_id, initial, max_new, order),
        wait_steps=wait_steps,
        skip_count=skip_count,
    )


def _active(
    request_id,
    order,
    *,
    block_size=2,
    initial=0,
    max_new=4,
    completed=1,
    service_wait_steps=0,
):
    spec = RequestSpec(request_id, initial, max_new, order)
    seq_len = initial + completed
    return ActiveRequestMetadata(
        spec=spec,
        seq_len=seq_len,
        remaining_tokens=max_new - completed,
        physical_blocks=(seq_len + block_size - 1) // block_size,
        committed_blocks=spec.commitment_blocks(block_size),
        service_wait_steps=service_wait_steps,
    )


def _snapshot(*, waiting=(), active=(), block_size=2, max_blocks=8, version=0, step=0):
    used_blocks = sum(item.physical_blocks for item in active)
    return SchedulingSnapshot(
        state_version=version,
        logical_step=step,
        block_size=block_size,
        max_blocks=max_blocks,
        free_blocks=max_blocks - used_blocks,
        waiting=waiting,
        active=active,
    )


def test_scheduler_symbols_are_public_api():
    assert flashdec.RequestSpec is RequestSpec
    assert flashdec.SchedulerConfig is SchedulerConfig
    assert flashdec.SchedulingSnapshot is SchedulingSnapshot
    assert flashdec.SchedulerDecision is SchedulerDecision
    assert flashdec.BlockAwareScheduler is BlockAwareScheduler
    assert flashdec.WaitingRequestMetadata is WaitingRequestMetadata
    assert flashdec.ActiveRequestMetadata is ActiveRequestMetadata


@pytest.mark.parametrize(
    ("initial", "max_new", "block_size", "expected"),
    [
        (0, 1, 32, 1),
        (31, 1, 32, 1),
        (32, 1, 32, 2),
        (1024, 64, 32, 34),
    ],
)
def test_request_lifetime_commitment_rounds_up_total_tokens(
    initial,
    max_new,
    block_size,
    expected,
):
    spec = RequestSpec("r", initial, max_new, 0)
    assert spec.commitment_blocks(block_size) == expected


def test_scheduler_admits_only_requests_whose_full_lifetime_fits():
    snapshot = _snapshot(
        waiting=(
            _waiting("context", 0, initial=2, max_new=2),
            _waiting("decode", 1, initial=0, max_new=4),
            _waiting("waiting", 2, initial=0, max_new=6),
        ),
        max_blocks=8,
    )
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=2)
    )

    decision = scheduler.plan(snapshot)

    assert decision.admit_ids == ("context", "decode")
    assert decision.runnable_ids == ("context", "decode")
    assert decision.waiting_ids == ("waiting",)
    assert decision.rejected_ids == ()
    assert decision.committed_blocks_before == 0
    assert decision.committed_blocks_after == 4
    assert decision.needed_physical_blocks_now == 3
    assert decision.free_blocks_before_step == 8
    assert decision.reasons == ("active_limit", "admitted")


def test_scheduler_rejects_request_that_can_never_fit_schedulable_capacity():
    snapshot = _snapshot(
        waiting=(
            _waiting("too-large", 0, max_new=7),
            _waiting("small", 1, max_new=2),
        ),
        max_blocks=4,
    )
    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=2,
            max_batch_requests=2,
            reserve_blocks=1,
        )
    )

    decision = scheduler.plan(snapshot)

    assert decision.rejected_ids == ("too-large",)
    assert decision.admit_ids == ("small",)
    assert decision.waiting_ids == ()
    assert "request_exceeds_capacity" in decision.reasons


def test_scheduler_allows_small_request_to_bypass_unaged_large_request():
    active = (_active("active", 0, max_new=8, completed=3),)
    snapshot = _snapshot(
        active=active,
        waiting=(
            _waiting("large", 1, max_new=6),
            _waiting("small", 2, max_new=4),
        ),
        max_blocks=6,
    )
    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=3,
            max_batch_requests=3,
            aging_threshold_steps=4,
        )
    )

    decision = scheduler.plan(snapshot)

    assert decision.admit_ids == ("small",)
    assert decision.waiting_ids == ("large",)
    assert decision.drain_for_request_id is None
    assert "capacity_deferred" in decision.reasons


def test_aged_large_request_creates_drain_barrier_against_younger_admission():
    active = (_active("active", 0, max_new=8, completed=3),)
    snapshot = _snapshot(
        active=active,
        waiting=(
            _waiting("large", 1, max_new=6, wait_steps=4),
            _waiting("small", 2, max_new=4),
        ),
        max_blocks=6,
    )
    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=3,
            max_batch_requests=3,
            aging_threshold_steps=4,
        )
    )

    decision = scheduler.plan(snapshot)

    assert decision.admit_ids == ()
    assert decision.waiting_ids == ("large", "small")
    assert decision.drain_for_request_id == "large"
    assert decision.reasons == ("aging_drain",)
    assert decision.runnable_ids == ("active",)


def test_scheduler_rotates_active_rows_by_service_wait_then_submission_order():
    active = (
        _active("newly-served", 0, max_new=2, service_wait_steps=0),
        _active("oldest-wait", 1, max_new=2, service_wait_steps=3),
        _active("next-wait", 2, max_new=2, service_wait_steps=1),
    )
    snapshot = _snapshot(active=active, max_blocks=3)
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=3, max_batch_requests=2)
    )

    decision = scheduler.plan(snapshot)

    assert decision.runnable_ids == ("oldest-wait", "next-wait")
    assert decision.deferred_ids == ("newly-served",)
    assert decision.needed_physical_blocks_now == 0
    assert decision.reasons == ("batch_limit",)


def test_lifetime_commitment_leaves_blocks_for_simultaneous_boundary_crossing():
    active = (
        _active("a", 0, max_new=4, completed=2),
        _active("b", 1, max_new=4, completed=2),
    )
    snapshot = _snapshot(active=active, max_blocks=4)
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=2)
    )

    decision = scheduler.plan(snapshot)

    assert decision.committed_blocks_before == 4
    assert decision.free_blocks_before_step == 2
    assert decision.runnable_ids == ("a", "b")
    assert decision.needed_physical_blocks_now == 2


def test_scheduler_rejects_snapshot_that_is_already_overcommitted_after_reserve():
    active = (
        _active("a", 0, max_new=4, completed=2),
        _active("b", 1, max_new=4, completed=2),
    )
    snapshot = _snapshot(active=active, max_blocks=4)
    scheduler = BlockAwareScheduler(
        SchedulerConfig(
            max_active_requests=2,
            max_batch_requests=2,
            reserve_blocks=1,
        )
    )

    with pytest.raises(ValueError, match="commitments exceed"):
        scheduler.plan(snapshot)


def test_scheduler_plan_is_deterministic_and_does_not_mutate_snapshot():
    snapshot = _snapshot(
        waiting=(
            _waiting("a", 0, max_new=2),
            _waiting("b", 1, max_new=2),
        ),
        version=7,
        step=11,
    )
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=2, max_batch_requests=1)
    )
    before = repr(snapshot)

    first = scheduler.plan(snapshot)
    second = scheduler.plan(snapshot)

    assert first == second
    assert repr(snapshot) == before
    assert first.snapshot_version == 7
    assert first.runnable_ids == ("a",)
    assert first.deferred_ids == ("b",)


def test_snapshot_rejects_inconsistent_physical_block_accounting():
    active = (_active("a", 0),)

    with pytest.raises(ValueError, match="exactly match"):
        SchedulingSnapshot(
            state_version=0,
            logical_step=0,
            block_size=2,
            max_blocks=4,
            free_blocks=4,
            active=active,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RequestSpec("r", -1, 1, 0), "initial_context_tokens"),
        (lambda: RequestSpec("r", 0, 0, 0), "max_new_tokens"),
        (
            lambda: SchedulerConfig(max_active_requests=1, max_batch_requests=2),
            "max_batch_requests",
        ),
        (
            lambda: ActiveRequestMetadata(
                RequestSpec("r", 0, 2, 0),
                seq_len=0,
                remaining_tokens=1,
                physical_blocks=0,
                committed_blocks=1,
            ),
            "seq_len",
        ),
    ],
)
def test_scheduler_metadata_rejects_invalid_values(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
