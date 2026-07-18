"""Dependency-free validation tests for fused transaction fast-path evidence."""

import math
from pathlib import Path
import subprocess
import sys
import unittest

from benchmarks.run_fused_transaction_fast_path import (
    PROFILE_TIMING_SCOPE,
    WALL_TIMING_SCOPE,
)
from benchmarks.summarize_fused_transaction_fast_path import (
    FastPathValidationError,
    REQUIRED_FIELDS,
    aggregate,
    render_markdown,
    validate_rows,
)


def _row(dtype, trial, transaction_path):
    layers = 2
    batch = 4
    context = 128
    repeats = 20
    block_size = 32
    final_seq_len = context + repeats
    max_blocks = batch * math.ceil((final_seq_len + 1) / block_size)
    used_blocks = batch * math.ceil(final_seq_len / block_size)
    profile_steps = 2
    parity_steps = 2
    rollback_repeats = 2
    order = "checked->trusted" if trial % 2 else "trusted->checked"
    mean_ms = 4.0 if transaction_path == "checked" else 3.2

    row = {field: "1" for field in REQUIRED_FIELDS}
    row.update(
        {
            "name": "fused_transaction_fast_path",
            "op": "fused_transaction_fast_path",
            "date": "2026-07-18T12:00:00+08:00",
            "run_id": "20260718T120000+0800-abc1234",
            "case": "l2_b4_c128",
            "transaction_path": transaction_path,
            "append_backend": "fused_cuda",
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
            "seed": str(700 + trial),
            "repeats": str(repeats),
            "mean_ms": f"{mean_ms:.6f}",
            "p50_ms": "4.000000" if transaction_path == "checked" else "3.200000",
            "p90_ms": "4.400000" if transaction_path == "checked" else "4.000000",
            "p99_ms": "5.000000" if transaction_path == "checked" else "4.200000",
            "min_ms": "3.500000" if transaction_path == "checked" else "3.000000",
            "max_ms": "5.000000" if transaction_path == "checked" else "4.500000",
            "begin_host_p50_ms": "0.500000",
            "commit_host_p50_ms": "0.250000",
            "decode_tokens_per_second": (
                "1000.000" if transaction_path == "checked" else "1250.000"
            ),
            "layer_steps_per_second": (
                "2000.000" if transaction_path == "checked" else "2500.000"
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
            "profile_steps": str(profile_steps),
            "profile_token_count": str(profile_steps),
            "profile_append_count": str(profile_steps * layers),
            "profile_decode_count": str(profile_steps * layers),
            "profile_cuda_event_count": (
                "80" if transaction_path == "checked" else "20"
            ),
            "profile_append_cpu_ms_per_layer": (
                "0.400000" if transaction_path == "checked" else "0.200000"
            ),
            "profile_append_device_ms_per_layer": (
                "0.400000" if transaction_path == "checked" else "0.200000"
            ),
            "profile_decode_device_ms_per_layer": "0.800000",
            "profile_item_count": "20" if transaction_path == "checked" else "0",
            "profile_local_scalar_dense_count": (
                "20" if transaction_path == "checked" else "0"
            ),
            "parity_steps": str(parity_steps),
            "parity_output_equal": "True",
            "parity_cache_equal": "True",
            "parity_state_equal": "True",
            "parity_validated": "True",
            "rollback_repeats": str(rollback_repeats),
            "rollback_p50_ms": "1.500000",
            "rollback_blocks": str(rollback_repeats * batch),
            "rollback_validated": "True",
            "validated_invariants": "True",
            "timing_scope": WALL_TIMING_SCOPE,
            "wall_timer_cuda_events": "False",
            "profile_timing_scope": PROFILE_TIMING_SCOPE,
            "speedup_vs_checked_p50": (
                "1.0000" if transaction_path == "checked" else "1.2500"
            ),
        }
    )
    return row


def _rows():
    return [
        _row(dtype, trial, transaction_path)
        for dtype in ("float16", "bfloat16")
        for trial in range(1, 4)
        for transaction_path in ("checked", "trusted")
    ]


def _validate(rows):
    return validate_rows(
        rows,
        expected_trials=3,
        expected_cases=("l2_b4_c128",),
        expected_dtypes=("float16", "bfloat16"),
    )


class FusedTransactionFastPathSummaryTests(unittest.TestCase):
    def test_summary_cli_help_runs_from_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "benchmarks/summarize_fused_transaction_fast_path.py"),
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

    def test_validate_aggregate_and_render(self):
        pairs = _validate(_rows())
        aggregates, overall = aggregate(pairs)
        markdown = render_markdown("results.csv", pairs, aggregates, overall)

        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(aggregates), 2)
        self.assertAlmostEqual(overall["p50"], 1.25)
        self.assertAlmostEqual(overall["decode_tokens_per_second"], 1.25)
        self.assertAlmostEqual(overall["profile_append_cpu"], 2.0)
        self.assertAlmostEqual(overall["profile_cuda_events"], 4.0)
        self.assertEqual(aggregates[0]["direction"], "trusted_faster")
        self.assertEqual(
            aggregates[0]["absolute"]["trusted"]["profile_item_count"], 0
        )
        self.assertIn("Rows: 12; paired trials: 6", markdown)
        self.assertIn("pure synchronized wall time", markdown)
        self.assertIn("extra retries: 0", markdown)
        self.assertIn("trusted_faster", markdown)
        self.assertIn("Absolute Attribution Medians", markdown)

        aggregates[0]["absolute"]["checked"][
            "profile_attempt_count"
        ] = 1.5
        markdown = render_markdown(
            "results.csv", pairs, aggregates, overall
        )
        self.assertIn("| 1.5 |", markdown)

    def test_rejects_incomplete_or_unknown_matrix(self):
        with self.assertRaisesRegex(FastPathValidationError, "matrix incomplete"):
            _validate(_rows()[:-1])

        rows = _rows()
        rows[0]["transaction_path"] = "unknown"
        with self.assertRaisesRegex(FastPathValidationError, "unsupported"):
            _validate(rows)

    def test_rejects_missing_column_or_duplicate_row(self):
        rows = _rows()
        for row in rows:
            del row["profile_item_count"]
        with self.assertRaisesRegex(FastPathValidationError, "missing"):
            _validate(rows)

        rows = _rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(FastPathValidationError, "duplicate row"):
            _validate(rows)

    def test_rejects_unknown_csv_column(self):
        rows = _rows()
        for row in rows:
            row["unexpected"] = "value"
        with self.assertRaisesRegex(FastPathValidationError, "unexpected"):
            _validate(rows)

    def test_rejects_trajectory_timing_or_evidence_drift(self):
        mutations = (
            ("final_used_blocks", "19", "block accounting"),
            ("transaction_layer_write_count", "295", "layer write"),
            ("profile_append_count", "3", "profile append"),
            ("parity_cache_equal", "False", "parity_cache_equal"),
            ("rollback_blocks", "7", "rollback block"),
            ("wall_timer_cuda_events", "True", "wall[_ ]timer"),
            ("speedup_vs_checked_p50", "1.1000", "speedup"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                rows = _rows()
                target = next(
                    row for row in rows if row["transaction_path"] == "trusted"
                )
                target[field] = value
                with self.assertRaisesRegex(FastPathValidationError, message):
                    _validate(rows)

    def test_rejects_tps_or_byte_derivation_drift(self):
        mutations = (
            ("decode_tokens_per_second", "1000", "decode TPS"),
            ("layer_steps_per_second", "2000", "layer TPS"),
            ("kv_write_bytes_per_token", "1", "KV write byte"),
            ("cache_capacity_bytes", "1", "capacity byte"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                rows = _rows()
                target = next(
                    row for row in rows if row["transaction_path"] == "trusted"
                )
                target[field] = value
                with self.assertRaisesRegex(FastPathValidationError, message):
                    _validate(rows)

    def test_rejects_zero_profiler_append_cpu_time(self):
        rows = _rows()
        target = next(
            row for row in rows if row["transaction_path"] == "trusted"
        )
        target["profile_append_cpu_ms_per_layer"] = "0"
        with self.assertRaisesRegex(
            FastPathValidationError,
            "profile_append_cpu_ms_per_layer must be positive and finite",
        ):
            _validate(rows)

    def test_rejects_seed_or_order_drift(self):
        rows = _rows()
        for row in rows:
            if row["trial"] == "2":
                row["seed"] = "999"
        with self.assertRaisesRegex(FastPathValidationError, "increase by one"):
            _validate(rows)

        rows = _rows()
        for row in rows:
            if row["trial"] == "2":
                row["path_order"] = "checked->trusted"
        with self.assertRaisesRegex(FastPathValidationError, "reverse path order"):
            _validate(rows)

    def test_rejects_checked_or_trusted_profiler_scalar_sync_count_drift(self):
        for path, value in (("checked", "19"), ("trusted", "1")):
            for field in (
                "profile_item_count",
                "profile_local_scalar_dense_count",
            ):
                with self.subTest(path=path, field=field):
                    rows = _rows()
                    target = next(
                        row
                        for row in rows
                        if row["transaction_path"] == path
                    )
                    target[field] = value
                    with self.assertRaisesRegex(
                        FastPathValidationError, f"{field} mismatch"
                    ):
                        _validate(rows)

    def test_validates_bounded_profiler_capture_attempts(self):
        for value in ("1", "2", "3"):
            with self.subTest(value=value):
                rows = _rows()
                rows[0]["profile_attempt_count"] = value
                _validate(rows)

        for value in ("0", "4"):
            with self.subTest(value=value):
                rows = _rows()
                rows[0]["profile_attempt_count"] = value
                with self.assertRaisesRegex(
                    FastPathValidationError,
                    "profile_attempt_count must be in",
                ):
                    _validate(rows)

    def test_aggregate_marks_p50_direction_crossing_one_as_unstable(self):
        rows = _rows()
        target = next(
            row
            for row in rows
            if row["dtype"] == "float16"
            and row["trial"] == "2"
            and row["transaction_path"] == "trusted"
        )
        target.update(
            {
                "p50_ms": "5.000000",
                "p90_ms": "5.000000",
                "p99_ms": "5.000000",
                "max_ms": "5.000000",
                "speedup_vs_checked_p50": "0.8000",
            }
        )
        pairs = _validate(rows)
        aggregates, _ = aggregate(pairs)
        result = next(
            row
            for row in aggregates
            if row["dtype"] == "float16" and row["case"] == "l2_b4_c128"
        )
        self.assertEqual(result["direction"], "unstable_crosses_1")


if __name__ == "__main__":
    unittest.main()
