"""Dependency-free strict validation tests for R4-B benchmark evidence."""

import math
from pathlib import Path
import subprocess
import sys
import unittest

from benchmarks.run_persistent_transaction_metadata import (
    PROFILE_TIMING_SCOPE,
    WALL_TIMING_SCOPE,
)
from benchmarks.summarize_persistent_transaction_metadata import (
    MetadataValidationError,
    REQUIRED_FIELDS,
    aggregate,
    render_markdown,
    validate_rows,
)


def _row(dtype, trial, metadata_path):
    layers = 2
    batch = 4
    context = 128
    repeats = 20
    block_size = 32
    final_seq_len = context + repeats
    max_blocks = batch * math.ceil((final_seq_len + 1) / block_size)
    used_blocks = batch * math.ceil(final_seq_len / block_size)
    profile_steps = 2
    rollback_repeats = 2
    materialized = metadata_path == "materialized"
    mean_ms = 4.0 if materialized else 3.2
    views = 2 * layers + 2 if materialized else 1
    reuses = 0 if materialized else layers
    order = (
        "materialized->persistent"
        if trial % 2
        else "persistent->materialized"
    )

    row = {field: "1" for field in REQUIRED_FIELDS}
    row.update(
        {
            "name": "persistent_transaction_metadata",
            "mean_ms": f"{mean_ms:.6f}",
            "p50_ms": "4.000000" if materialized else "3.200000",
            "p90_ms": "4.400000" if materialized else "3.600000",
            "p99_ms": "4.800000" if materialized else "4.000000",
            "min_ms": "3.500000" if materialized else "3.000000",
            "max_ms": "5.000000" if materialized else "4.200000",
            "repeats": str(repeats),
            "date": "2026-07-19T12:00:00+08:00",
            "run_id": "20260719T120000+0800-abc1234",
            "op": "persistent_transaction_metadata",
            "case": "l2_b4_c128",
            "metadata_path": metadata_path,
            "append_backend": "fused_cuda",
            "raw_dispatch": "trusted",
            "decode_backend": "triton",
            "dtype": dtype,
            "device": "NVIDIA GeForce RTX 5070",
            "torch": "2.11.0+cu128",
            "cuda": "12.8",
            "git_commit": "abc1234",
            "num_layers": str(layers),
            "batch_size": str(batch),
            "context_tokens": str(context),
            "num_q_heads": "32",
            "num_kv_heads": "8",
            "head_dim": "128",
            "block_size": str(block_size),
            "num_warps": "2",
            "warmup": "3",
            "trial": str(trial),
            "trial_count": "3",
            "path_order": order,
            "seed": str(810 + trial),
            "begin_host_p50_ms": "0.500000",
            "commit_host_p50_ms": "0.250000",
            "decode_tokens_per_second": (
                "1000.000" if materialized else "1250.000"
            ),
            "layer_steps_per_second": (
                "2000.000" if materialized else "2500.000"
            ),
            "kv_write_bytes_per_token": "32768",
            "cache_capacity_bytes": "5242880",
            "final_seq_len": str(final_seq_len),
            "final_used_blocks": str(used_blocks),
            "final_free_blocks": str(max_blocks - used_blocks),
            "final_request_blocks": str(used_blocks),
            "max_blocks": str(max_blocks),
            "allocation_count": str(used_blocks),
            "fresh_allocation_count": str(used_blocks),
            "reuse_count": "0",
            "capacity_failure_count": "0",
            "transaction_begin_count": str(final_seq_len),
            "transaction_commit_count": str(final_seq_len),
            "transaction_abort_count": "0",
            "transaction_layer_write_count": str(final_seq_len * layers),
            "engine_completed_step_count": str(repeats),
            "engine_appended_token_count": str(repeats * batch),
            "validated_invariants": "True",
            "timing_scope": WALL_TIMING_SCOPE,
            "wall_timer_cuda_events": "False",
            "profile_timing_scope": PROFILE_TIMING_SCOPE,
            "metadata_build_delta": str(repeats),
            "metadata_materialization_delta": str(repeats * views),
            "metadata_reuse_delta": str(repeats * reuses),
            "metadata_release_delta": str(repeats),
            "metadata_resident_before": "0",
            "metadata_resident_after": "0",
            "metadata_builds_per_token": "1.000000",
            "metadata_materializations_per_token": f"{views:.6f}",
            "metadata_reuses_per_token": f"{reuses:.6f}",
            "metadata_releases_per_token": "1.000000",
            "profile_steps": str(profile_steps),
            "profile_token_count": str(profile_steps),
            "profile_append_count": str(profile_steps * layers),
            "profile_decode_count": str(profile_steps * layers),
            "profile_append_cpu_ms_per_layer": (
                "0.400000" if materialized else "0.200000"
            ),
            "profile_item_count": "0",
            "profile_local_scalar_dense_count": "0",
            "profile_attempt_count": "1",
            "rollback_repeats": str(rollback_repeats),
            "rollback_p50_ms": "1.500000",
            "rollback_blocks": str(rollback_repeats * batch),
            "rollback_metadata_releases": str(rollback_repeats),
            "rollback_metadata_resident_after": "0",
            "rollback_validated": "True",
            "parity_steps": "2",
            "parity_output_equal": "True",
            "parity_cache_equal": "True",
            "parity_state_equal": "True",
            "parity_validated": "True",
            "speedup_vs_materialized_p50": (
                "1.0000" if materialized else "1.2500"
            ),
        }
    )
    return row


def _rows():
    return [
        _row(dtype, trial, path)
        for dtype in ("float16", "bfloat16")
        for trial in range(1, 4)
        for path in ("materialized", "persistent")
    ]


def _validate(rows):
    return validate_rows(
        rows,
        expected_trials=3,
        expected_cases=("l2_b4_c128",),
        expected_dtypes=("float16", "bfloat16"),
    )


class PersistentTransactionMetadataSummaryTests(unittest.TestCase):
    def test_cli_help_runs_without_torch_or_cuda(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "benchmarks/summarize_persistent_transaction_metadata.py"
                ),
                "--help",
            ],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--expected-trials", result.stdout)

    def test_validate_aggregate_and_render_full_tail_ranges(self):
        pairs = _validate(_rows())
        aggregates, overall = aggregate(pairs)
        markdown = render_markdown("results.csv", pairs, aggregates, overall)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(aggregates), 2)
        self.assertAlmostEqual(overall["p50"], 1.25)
        self.assertAlmostEqual(overall["decode_tokens_per_second"], 1.25)
        self.assertAlmostEqual(overall["profile_append_cpu"], 2.0)
        self.assertTrue(overall["screening_gate_passed"])
        self.assertFalse(overall["formal_matrix_complete"])
        self.assertFalse(overall["keep_gate_passed"])
        self.assertEqual(aggregates[0]["direction"], "persistent_faster")
        self.assertIn("Rows: 12; paired trials: 6", markdown)
        self.assertIn("p90 [min,max]", markdown)
        self.assertIn("p99 [min,max]", markdown)
        self.assertIn("2L+2", markdown)
        self.assertIn("Cache transaction-view materializations only", markdown)
        self.assertIn("overall p50 >= 1.05x", markdown)
        self.assertIn("Observed-matrix screening: `pass`", markdown)
        self.assertIn("Formal matrix coverage: 2/16 groups (`incomplete`)", markdown)
        self.assertIn("Formal keep gate: `not_evaluated`", markdown)
        self.assertIn("validator", markdown)
        self.assertNotIn("append device", markdown)

    def test_rejects_bad_metadata_count(self):
        rows = _rows()
        target = next(
            row for row in rows if row["metadata_path"] == "persistent"
        )
        target["metadata_reuse_delta"] = "39"
        with self.assertRaisesRegex(
            MetadataValidationError, "metadata_reuse_delta mismatch"
        ):
            _validate(rows)

    def test_rejects_pair_drift_or_missing_pair(self):
        rows = _rows()
        target = next(
            row
            for row in rows
            if row["metadata_path"] == "persistent" and row["trial"] == "1"
        )
        target["seed"] = "999"
        with self.assertRaisesRegex(
            MetadataValidationError, "same seed per trial|paired inputs"
        ):
            _validate(rows)

        with self.assertRaisesRegex(MetadataValidationError, "matrix incomplete"):
            _validate(_rows()[:-1])

    def test_rejects_bad_latency_range(self):
        rows = _rows()
        rows[0]["p90_ms"] = "3.000000"
        with self.assertRaisesRegex(MetadataValidationError, "percentile range"):
            _validate(rows)

    def test_rejects_missing_or_unknown_schema_fields(self):
        rows = _rows()
        for row in rows:
            del row["metadata_build_delta"]
        with self.assertRaisesRegex(MetadataValidationError, "missing"):
            _validate(rows)

        rows = _rows()
        for row in rows:
            row["unexpected"] = "value"
        with self.assertRaisesRegex(MetadataValidationError, "unexpected"):
            _validate(rows)

    def test_rejects_scalar_sync_and_profiler_range_drift(self):
        rows = _rows()
        rows[0]["profile_item_count"] = "1"
        with self.assertRaisesRegex(MetadataValidationError, "must be zero"):
            _validate(rows)

        rows = _rows()
        rows[0]["profile_append_cpu_ms_per_layer"] = "0"
        with self.assertRaisesRegex(
            MetadataValidationError,
            "profile_append_cpu_ms_per_layer must be positive",
        ):
            _validate(rows)


if __name__ == "__main__":
    unittest.main()
