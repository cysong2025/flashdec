"""Dependency-free validation tests for multi-layer trial summaries."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.summarize_multi_layer_trials import (
    MultiLayerValidationError,
    REQUIRED_FIELDS,
    aggregate,
    render_markdown,
    validate_rows,
)


def _row(dtype, case, trial, backend, *, layers=2, batch=4, context=128):
    row = {field: "1" for field in REQUIRED_FIELDS}
    repeats = 20
    profile_steps = 2
    rollback_repeats = 3 if layers > 1 else 0
    order = "torch->fused_cuda" if trial % 2 else "fused_cuda->torch"
    max_blocks = batch * 5
    row.update(
        {
            "name": "multi_layer_decode_engine",
            "op": "multi_layer_decode_engine",
            "case": case,
            "append_backend": backend,
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
            "block_size": "32",
            "num_warps": "2",
            "warmup": "3",
            "trial": str(trial),
            "trial_count": "3",
            "backend_order": order,
            "seed": str(540 + trial),
            "repeats": str(repeats),
            "kv_write_bytes_per_token": "131072",
            "cache_capacity_bytes": "10485760",
            "final_seq_len": str(context + repeats),
            "final_used_blocks": str(batch * 5),
            "final_free_blocks": str(max_blocks - batch * 5),
            "max_blocks": str(max_blocks),
            "transaction_begin_count": str(context + repeats),
            "transaction_commit_count": str(context + repeats),
            "transaction_abort_count": "0",
            "transaction_layer_write_count": str(
                (context + repeats) * layers
            ),
            "profile_steps": str(profile_steps),
            "profile_token_count": str(profile_steps),
            "profile_append_count": str(profile_steps * layers),
            "profile_decode_count": str(profile_steps * layers),
            "rollback_repeats": str(rollback_repeats),
            "rollback_blocks": str(rollback_repeats * batch),
            "rollback_p50_ms": "1.5" if layers > 1 else "0.0",
            "rollback_validated": "True",
            "validated_invariants": "True",
            "timing_scope": "non-instrumented complete token",
            "profile_timing_scope": "separate profiler",
            "begin_host_mean_ms": "0.05",
            "commit_host_mean_ms": "0.04",
            "layer_steps_per_second": "8000",
            "profile_cuda_event_count": "80" if backend == "torch" else "20",
            "profile_append_device_ms_per_layer": (
                "0.40" if backend == "torch" else "0.20"
            ),
            "profile_decode_device_ms_per_layer": "0.80",
        }
    )
    if backend == "torch":
        row.update(
            {
                "mean_ms": "4.0",
                "p50_ms": "4.0",
                "p90_ms": "4.4",
                "p99_ms": "5.0",
                "device_p50_ms": "3.5",
                "layer_device_p50_ms": "1.75",
                "decode_tokens_per_second": "1000",
                "speedup_vs_torch_p50": "1.0",
            }
        )
    else:
        row.update(
            {
                "mean_ms": "3.2",
                "p50_ms": "3.2",
                "p90_ms": "4.0",
                "p99_ms": "4.0",
                "device_p50_ms": "2.8",
                "layer_device_p50_ms": "1.4",
                "decode_tokens_per_second": "1250",
                "speedup_vs_torch_p50": "1.25",
            }
        )
    return row


def _rows():
    return [
        _row(dtype, "l2_b4_c128", trial, backend)
        for dtype in ("float16", "bfloat16")
        for trial in range(1, 4)
        for backend in ("torch", "fused_cuda")
    ]


class MultiLayerWorkloadSummaryTests(unittest.TestCase):
    def test_validate_aggregate_and_render(self):
        pairs = validate_rows(
            _rows(),
            expected_cases=("l2_b4_c128",),
        )
        aggregates, overall = aggregate(pairs)
        markdown = render_markdown("results.csv", pairs, aggregates, overall)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(aggregates), 2)
        self.assertAlmostEqual(overall["p50"], 1.25)
        self.assertAlmostEqual(overall["profile_cuda_events"], 4.0)
        self.assertAlmostEqual(
            aggregates[0]["absolute"]["torch"][
                "profile_append_device_ms_per_layer"
            ],
            0.4,
        )
        self.assertAlmostEqual(
            aggregates[0]["absolute"]["fused_cuda"]["p50_ms"],
            3.2,
        )
        self.assertIn("Rows: 12; paired trials: 6", markdown)
        self.assertIn("attribution-only", markdown)
        self.assertIn("fused_faster", markdown)
        self.assertIn("Absolute Attribution Medians", markdown)
        self.assertIn("0.400000", markdown)
        self.assertIn("0.200000", markdown)

    def test_rejects_incomplete_matrix(self):
        with self.assertRaisesRegex(MultiLayerValidationError, "incomplete"):
            validate_rows(
                _rows()[:-1],
                expected_cases=("l2_b4_c128",),
            )

    def test_rejects_transaction_or_profile_count_drift(self):
        cases = (
            ("final_seq_len", "147", "final_seq_len"),
            ("transaction_commit_count", "147", "transaction commit"),
            ("transaction_layer_write_count", "295", "layer write"),
            ("profile_append_count", "3", "profile append"),
            ("validated_invariants", "False", "invariant failure"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                rows = _rows()
                rows[0][field] = value
                with self.assertRaisesRegex(MultiLayerValidationError, message):
                    validate_rows(
                        rows,
                        expected_cases=("l2_b4_c128",),
                    )

    def test_rejects_paired_trajectory_drift(self):
        rows = _rows()
        rows[1]["final_used_blocks"] = "19"
        rows[1]["final_free_blocks"] = "1"
        with self.assertRaisesRegex(
            MultiLayerValidationError, "paired trajectory differs"
        ):
            validate_rows(rows, expected_cases=("l2_b4_c128",))

    def test_single_layer_requires_no_rollback_probe(self):
        rows = [
            _row(
                "float16",
                "l1_b4_c128",
                trial,
                backend,
                layers=1,
            )
            for trial in range(1, 4)
            for backend in ("torch", "fused_cuda")
        ]
        pairs = validate_rows(
            rows,
            expected_cases=("l1_b4_c128",),
            expected_dtypes=("float16",),
        )
        self.assertEqual(len(pairs), 3)


if __name__ == "__main__":
    unittest.main()
