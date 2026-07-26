"""Dependency-free validation coverage for scheduler workload summaries."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.summarize_scheduler_workload import (
    CASES,
    DTYPES,
    POLICIES,
    validate_scheduler_rows,
    write_summary,
)


def _rows(trials=3):
    result = []
    for case in CASES:
        for dtype in DTYPES:
            for policy in POLICIES:
                for trial in range(1, trials + 1):
                    completion = 1.0
                    deadlocks = 0
                    forced = 0
                    cancelled = 0
                    useful = 128
                    completed = 128
                    if case == "boundary_deadlock" and policy == "greedy_step_only":
                        completion = 0.0
                        deadlocks = 1
                        useful = 0
                        completed = 64
                    elif case == "boundary_deadlock" and policy == "cancel_on_backpressure":
                        completion = 0.5
                        forced = 1
                        cancelled = 1
                        useful = 64
                        completed = 96
                    result.append(
                        {
                            "case": case,
                            "dtype": dtype,
                            "policy": policy,
                            "trial": str(trial),
                            "policy_order": "->".join(
                                POLICIES[(trial - 1) % len(POLICIES) :]
                                + POLICIES[: (trial - 1) % len(POLICIES)]
                            ),
                            "seed": str(502 + trial),
                            "git_commit": "abc1234",
                            "device": "test-gpu",
                            "validated_invariants": "True",
                            "completion_rate": str(completion),
                            "resource_deadlocks": str(deadlocks),
                            "forced_cancellations": str(forced),
                            "cancelled_requests": str(cancelled),
                            "useful_tokens": str(useful),
                            "completed_tokens": str(completed),
                            "p50_ms": "1.0",
                            "p99_ms": "2.0",
                            "useful_tokens_per_second": "100.0",
                            "admission_wait_p90": "3",
                            "scheduler_p50_ms": "0.01",
                            "max_committed_blocks": "2",
                            "max_physical_blocks": "2",
                        }
                    )
    return result


class SchedulerWorkloadSummaryTests(unittest.TestCase):
    def test_validator_accepts_complete_policy_matrix(self):
        rows = _rows()
        self.assertEqual(validate_scheduler_rows(rows), rows)

    def test_validator_rejects_missing_duplicate_and_bad_policy_semantics(self):
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_scheduler_rows(_rows()[:-1])

        duplicate = _rows()
        duplicate.append(dict(duplicate[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_scheduler_rows(duplicate)

        invalid = _rows()
        target = next(
            row
            for row in invalid
            if row["case"] == "boundary_deadlock"
            and row["policy"] == "lifetime_fifo_aging"
        )
        target["completion_rate"] = "0.5"
        with self.assertRaisesRegex(ValueError, "lifetime boundary"):
            validate_scheduler_rows(invalid)

    def test_summary_writes_auditable_cross_trial_table(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            write_summary(_rows(), output, "input.csv", expected_trials=3)
            text = output.read_text()

        self.assertTrue(text.startswith("# Scheduler Capacity and Progress Summary\n"))
        self.assertIn("Rows: 36", text)
        self.assertIn("`abc1234`", text)
        self.assertIn("boundary_deadlock", text)
        self.assertIn("useful TPS", text)


if __name__ == "__main__":
    unittest.main()
