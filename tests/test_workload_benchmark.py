"""Pure-Python configuration coverage for the dynamic workload benchmark."""

import pytest

from benchmarks.run_decode_engine_workload import (
    WORKLOADS,
    _quick_spec,
    _trial_append_backends,
)


def test_trial_append_backends_alternates_order_without_mutating_input():
    backends = ["torch", "fused_cuda"]

    assert _trial_append_backends(backends, 0) == ["torch", "fused_cuda"]
    assert _trial_append_backends(backends, 1) == ["fused_cuda", "torch"]
    assert _trial_append_backends(backends, 2) == ["torch", "fused_cuda"]
    assert backends == ["torch", "fused_cuda"]


def test_trial_append_backends_rejects_negative_index():
    with pytest.raises(ValueError, match="non-negative"):
        _trial_append_backends(["torch", "fused_cuda"], -1)


@pytest.mark.parametrize(
    ("workload", "steps"),
    [("short_churn", 24), ("mixed_steady", 32), ("long_pressure", 72)],
)
def test_quick_spec_preserves_pressure_boundaries(workload, steps):
    original = WORKLOADS[workload]
    quick = _quick_spec(original)

    assert quick.config.steps == steps
    assert quick.max_blocks == original.max_blocks
    assert original.config.steps > quick.config.steps
