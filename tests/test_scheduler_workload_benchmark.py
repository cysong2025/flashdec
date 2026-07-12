"""Pure configuration coverage for the scheduler workload benchmark."""

import pytest

from benchmarks.run_scheduler_workload import CASES, POLICIES, _config, _trial_policies
from flashdec.scheduled_workload import (
    GREEDY_STEP_ONLY,
    LIFETIME_FIFO_AGING,
)


def test_boundary_case_preserves_the_common_block_boundary_deadlock():
    case = CASES["boundary_deadlock"]

    assert case.max_blocks == 2
    assert len(case.arrivals) == 2
    assert {item.spec.max_new_tokens for item in case.arrivals} == {64}
    assert all(item.spec.commitment_blocks(32) == 2 for item in case.arrivals)
    assert _config(case, GREEDY_STEP_ONLY).max_stalled_steps == 4
    assert _config(case, LIFETIME_FIFO_AGING).max_batch_requests == 2


def test_trial_policy_order_rotates_without_mutating_input():
    policies = list(POLICIES)

    assert _trial_policies(policies, 0) == tuple(POLICIES)
    assert _trial_policies(policies, 1) == tuple(POLICIES[1:] + POLICIES[:1])
    assert _trial_policies(policies, 2) == tuple(POLICIES[2:] + POLICIES[:2])
    assert policies == list(POLICIES)

    with pytest.raises(ValueError, match="non-negative"):
        _trial_policies(policies, -1)
