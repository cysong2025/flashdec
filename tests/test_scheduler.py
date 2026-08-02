"""Dependency-free correctness coverage for the block-aware scheduler."""

import unittest

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


def _snapshot(
    *,
    waiting=(),
    active=(),
    block_size=2,
    max_blocks=8,
    resident_prefix_blocks=0,
    version=0,
    step=0,
):
    used_blocks = resident_prefix_blocks + sum(
        item.private_physical_blocks for item in active
    )
    return SchedulingSnapshot(
        state_version=version,
        logical_step=step,
        block_size=block_size,
        max_blocks=max_blocks,
        free_blocks=max_blocks - used_blocks,
        resident_prefix_blocks=resident_prefix_blocks,
        waiting=waiting,
        active=active,
    )


class SchedulerPolicyTests(unittest.TestCase):
    def test_scheduler_symbols_are_public_api(self):
        self.assertIs(flashdec.RequestSpec, RequestSpec)
        self.assertIs(flashdec.SchedulerConfig, SchedulerConfig)
        self.assertIs(flashdec.SchedulingSnapshot, SchedulingSnapshot)
        self.assertIs(flashdec.SchedulerDecision, SchedulerDecision)
        self.assertIs(flashdec.BlockAwareScheduler, BlockAwareScheduler)
        self.assertIs(flashdec.WaitingRequestMetadata, WaitingRequestMetadata)
        self.assertIs(flashdec.ActiveRequestMetadata, ActiveRequestMetadata)

    def test_request_lifetime_commitment_rounds_up_total_tokens(self):
        cases = (
            (0, 1, 32, 1),
            (31, 1, 32, 1),
            (32, 1, 32, 2),
            (1024, 64, 32, 34),
        )
        for initial, max_new, block_size, expected in cases:
            with self.subTest(
                initial=initial,
                max_new=max_new,
                block_size=block_size,
            ):
                spec = RequestSpec("r", initial, max_new, 0)
                self.assertEqual(spec.commitment_blocks(block_size), expected)

    def test_shared_prefix_admission_counts_residency_once_and_commits_private_tail(self):
        spec = RequestSpec(
            "shared",
            initial_context_tokens=4,
            max_new_tokens=3,
            submission_order=0,
            prefix_id="system",
        )
        waiting = WaitingRequestMetadata(spec, shared_prefix_blocks=2)
        snapshot = _snapshot(
            waiting=(waiting,),
            block_size=2,
            max_blocks=8,
            resident_prefix_blocks=2,
        )
        scheduler = BlockAwareScheduler(
            SchedulerConfig(max_active_requests=1, max_batch_requests=1)
        )

        decision = scheduler.plan(snapshot)

        self.assertEqual(waiting.private_commitment_blocks(2), 2)
        self.assertEqual(decision.admit_ids, ("shared",))
        self.assertEqual(decision.runnable_ids, ("shared",))
        self.assertEqual(decision.committed_blocks_before, 2)
        self.assertEqual(decision.committed_blocks_after, 4)
        self.assertEqual(decision.needed_physical_blocks_now, 1)

    def test_two_active_requests_can_reference_same_resident_prefix(self):
        spec_a = RequestSpec("a", 4, 2, 0, prefix_id="system")
        spec_b = RequestSpec("b", 4, 2, 1, prefix_id="system")
        active = (
            ActiveRequestMetadata(
                spec_a,
                seq_len=4,
                remaining_tokens=2,
                physical_blocks=2,
                committed_blocks=1,
                shared_prefix_blocks=2,
            ),
            ActiveRequestMetadata(
                spec_b,
                seq_len=4,
                remaining_tokens=2,
                physical_blocks=2,
                committed_blocks=1,
                shared_prefix_blocks=2,
            ),
        )
        snapshot = _snapshot(
            active=active,
            block_size=2,
            max_blocks=8,
            resident_prefix_blocks=2,
        )
        scheduler = BlockAwareScheduler(
            SchedulerConfig(max_active_requests=2, max_batch_requests=2)
        )

        decision = scheduler.plan(snapshot)

        self.assertEqual(snapshot.free_blocks, 6)
        self.assertEqual(decision.committed_blocks_before, 4)
        self.assertEqual(decision.needed_physical_blocks_now, 2)
        self.assertEqual(decision.runnable_ids, ("a", "b"))

    def test_snapshot_rejects_prefix_that_does_not_cover_initial_context(self):
        waiting = WaitingRequestMetadata(
            RequestSpec("r", 6, 1, 0, prefix_id="system"),
            shared_prefix_blocks=2,
        )

        with self.assertRaisesRegex(ValueError, "full initial context"):
            _snapshot(
                waiting=(waiting,),
                block_size=2,
                resident_prefix_blocks=2,
            )

    def test_snapshot_rejects_shared_prefix_larger_than_global_residency(self):
        waiting = WaitingRequestMetadata(
            RequestSpec("r", 4, 1, 0, prefix_id="system"),
            shared_prefix_blocks=2,
        )

        with self.assertRaisesRegex(ValueError, "exceed global residency"):
            _snapshot(
                waiting=(waiting,),
                block_size=2,
                resident_prefix_blocks=1,
            )

    def test_scheduler_admits_only_requests_whose_full_lifetime_fits(self):
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

        self.assertEqual(decision.admit_ids, ("context", "decode"))
        self.assertEqual(decision.runnable_ids, ("context", "decode"))
        self.assertEqual(decision.waiting_ids, ("waiting",))
        self.assertEqual(decision.rejected_ids, ())
        self.assertEqual(decision.committed_blocks_before, 0)
        self.assertEqual(decision.committed_blocks_after, 4)
        self.assertEqual(decision.needed_physical_blocks_now, 3)
        self.assertEqual(decision.free_blocks_before_step, 8)
        self.assertEqual(decision.reasons, ("active_limit", "admitted"))

    def test_scheduler_rejects_request_that_can_never_fit_schedulable_capacity(self):
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

        self.assertEqual(decision.rejected_ids, ("too-large",))
        self.assertEqual(decision.admit_ids, ("small",))
        self.assertEqual(decision.waiting_ids, ())
        self.assertIn("request_exceeds_capacity", decision.reasons)

    def test_scheduler_allows_small_request_to_bypass_unaged_large_request(self):
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

        self.assertEqual(decision.admit_ids, ("small",))
        self.assertEqual(decision.waiting_ids, ("large",))
        self.assertIsNone(decision.drain_for_request_id)
        self.assertIn("capacity_deferred", decision.reasons)

    def test_aged_large_request_creates_drain_barrier(self):
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

        self.assertEqual(decision.admit_ids, ())
        self.assertEqual(decision.waiting_ids, ("large", "small"))
        self.assertEqual(decision.drain_for_request_id, "large")
        self.assertEqual(decision.reasons, ("aging_drain",))
        self.assertEqual(decision.runnable_ids, ("active",))

    def test_scheduler_rotates_active_rows_by_service_wait(self):
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

        self.assertEqual(decision.runnable_ids, ("oldest-wait", "next-wait"))
        self.assertEqual(decision.deferred_ids, ("newly-served",))
        self.assertEqual(decision.needed_physical_blocks_now, 0)
        self.assertEqual(decision.reasons, ("batch_limit",))

    def test_lifetime_commitment_covers_simultaneous_boundary_crossing(self):
        active = (
            _active("a", 0, max_new=4, completed=2),
            _active("b", 1, max_new=4, completed=2),
        )
        snapshot = _snapshot(active=active, max_blocks=4)
        scheduler = BlockAwareScheduler(
            SchedulerConfig(max_active_requests=2, max_batch_requests=2)
        )

        decision = scheduler.plan(snapshot)

        self.assertEqual(decision.committed_blocks_before, 4)
        self.assertEqual(decision.free_blocks_before_step, 2)
        self.assertEqual(decision.runnable_ids, ("a", "b"))
        self.assertEqual(decision.needed_physical_blocks_now, 2)

    def test_scheduler_rejects_already_overcommitted_snapshot(self):
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

        with self.assertRaisesRegex(ValueError, "commitments exceed"):
            scheduler.plan(snapshot)

    def test_scheduler_plan_is_deterministic_and_pure(self):
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

        self.assertEqual(first, second)
        self.assertEqual(repr(snapshot), before)
        self.assertEqual(first.snapshot, snapshot)
        self.assertEqual(first.scheduler_config, scheduler.config)
        self.assertEqual(first.snapshot_version, 7)
        self.assertEqual(first.runnable_ids, ("a",))
        self.assertEqual(first.deferred_ids, ("b",))

    def test_snapshot_rejects_inconsistent_physical_block_accounting(self):
        active = (_active("a", 0),)

        with self.assertRaisesRegex(ValueError, "exactly match"):
            SchedulingSnapshot(
                state_version=0,
                logical_step=0,
                block_size=2,
                max_blocks=4,
                free_blocks=4,
                active=active,
            )

    def test_scheduler_metadata_rejects_invalid_values(self):
        cases = (
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
        )
        for factory, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    factory()


if __name__ == "__main__":
    unittest.main()
