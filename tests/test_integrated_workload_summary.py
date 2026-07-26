"""Dependency-free strict validation coverage for integrated-workload evidence."""

import unittest

from benchmarks.run_integrated_scheduled_multi_layer import CASES
from benchmarks.summarize_integrated_scheduled_multi_layer import (
    IntegratedValidationError,
    REQUIRED_FIELDS,
    aggregate,
    render_markdown,
    validate_rows,
)
from flashdec.integrated_workload import (
    build_integrated_reference,
    standard_integrated_config,
)


def _row(dtype, case_name, trial, trials=3):
    case = CASES[case_name]
    reference = build_integrated_reference(
        standard_integrated_config(
            num_layers=case.num_layers,
            context_tokens=case.context_tokens,
        ),
        block_size=32,
        max_blocks=case.max_blocks,
        resident_prefix_blocks=case.prefix_blocks,
    )
    order_values = list(CASES)
    offset = (trial - 1) % len(order_values)
    order = order_values[offset:] + order_values[:offset]
    bytes_per_block = case.num_layers * 1_048_576
    peak = max(step.used_blocks for step in reference.steps)
    row = {field: "1" for field in REQUIRED_FIELDS}
    row.update(
        {
            "name": "integrated_scheduled_multi_layer",
            "op": "integrated_scheduled_multi_layer",
            "case": case.name,
            "dtype": dtype,
            "device": "NVIDIA GeForce RTX 5070",
            "torch": "2.11.0+cu128",
            "cuda": "12.8",
            "git_commit": "abc1234",
            "append_backend": "fused_cuda",
            "decode_backend": "triton",
            "metadata_policy": "materialized",
            "num_layers": str(case.num_layers),
            "context_tokens": str(case.context_tokens),
            "prefix_blocks": str(case.prefix_blocks),
            "max_blocks": str(case.max_blocks),
            "num_q_heads": "32",
            "num_kv_heads": "8",
            "head_dim": "128",
            "block_size": "32",
            "num_warps": "2",
            "trial": str(trial),
            "trial_count": str(trials),
            "case_order": "->".join(order),
            "seed": str(1700 + trial),
            "repeats": str(len(reference.steps)),
            "reference_steps": str(len(reference.steps)),
            "trajectory_digest": reference.digest,
            "reference_trajectory_digest": reference.digest,
            "trajectory_validated": "True",
            "completed_request_ids": "|".join(reference.completed_request_ids),
            "cancelled_request_ids": "|".join(reference.cancelled_request_ids),
            "rejected_request_ids": "|".join(reference.rejected_request_ids),
            "successful_steps": str(reference.successful_steps),
            "aborted_steps": str(reference.aborted_steps),
            "completed_tokens": str(reference.completed_tokens),
            "block_reuse_count": "3",
            "peak_used_blocks": str(peak),
            "terminal_resident_prefix_blocks": str(case.prefix_blocks),
            "final_free_blocks": str(case.max_blocks),
            "bytes_per_block": str(bytes_per_block),
            "peak_allocated_kv_bytes": str(peak * bytes_per_block),
            "mean_ms": "3.0",
            "p50_ms": "2.5",
            "p90_ms": "4.0",
            "complete_step_p99_ms": "5.0",
            "scheduler_p50_ms": "0.1",
            "context_seed_p50_ms": "0.2",
            "engine_p50_ms": "2.0",
            "decode_tokens_per_second": "500.0",
            "transaction_begin_count": str(
                2 * case.context_tokens
                + reference.successful_steps
                + reference.aborted_steps
            ),
            "transaction_commit_count": str(
                2 * case.context_tokens + reference.successful_steps
            ),
            "transaction_abort_count": str(reference.aborted_steps),
            "transaction_layer_write_count": str(
                (2 * case.context_tokens + reference.successful_steps)
                * case.num_layers
                + 1
            ),
            "transaction_rollback_block_count": "1",
            "engine_transaction_layer_step_count": str(
                reference.successful_steps * case.num_layers + 1
            ),
            "engine_transaction_abort_count": str(reference.aborted_steps),
            "prefix_registration_count": "1",
            "prefix_hit_count": "2",
            "prefix_eviction_count": "1",
            "final_open_transaction_count": "0",
            "final_used_blocks": "0",
            "validated_invariants": "True",
            "timing_scope": "integrated lifecycle wall",
        }
    )
    return row


def _rows(trials=3):
    return [
        _row(dtype, case, trial, trials=trials)
        for dtype in ("float16", "bfloat16")
        for trial in range(1, trials + 1)
        for case in CASES
    ]


class IntegratedWorkloadSummaryTests(unittest.TestCase):
    def test_validator_accepts_complete_reference_bound_matrix(self):
        rows = _rows()
        self.assertEqual(validate_rows(rows), rows)
        aggregates = aggregate(rows)
        markdown = render_markdown("results.csv", rows, aggregates)
        self.assertTrue(markdown.startswith("# Integrated Runtime Lifecycle Summary\n"))
        self.assertEqual(len(aggregates), 8)
        self.assertIn("Rows: 24; trials: 3", markdown)
        self.assertIn("materialized transaction metadata", markdown)
        self.assertIn("not a shared-prefix speedup A/B", markdown)
        self.assertIn("zero-used cleanup", markdown)

    def test_validator_rejects_matrix_digest_and_cleanup_drift(self):
        with self.assertRaisesRegex(IntegratedValidationError, "matrix mismatch"):
            validate_rows(_rows()[:-1])

        cases = (
            ("trajectory_digest", "bad", "trajectory digest"),
            ("transaction_layer_write_count", "1", "layer_write"),
            ("block_reuse_count", "0", "reuse"),
            ("final_used_blocks", "1", "final_used_blocks"),
            ("metadata_policy", "persistent", "metadata_policy"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                rows = _rows()
                rows[0][field] = value
                with self.assertRaisesRegex(IntegratedValidationError, message):
                    validate_rows(rows)

    def test_validator_rejects_trial_order_or_seed_drift(self):
        rows = _rows()
        for row in rows:
            if row["trial"] == "2":
                row["case_order"] = "->".join(CASES)
        with self.assertRaisesRegex(IntegratedValidationError, "case order"):
            validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
