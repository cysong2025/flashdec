"""Dependency-free strict tests for the R5 FlashInfer baseline summary."""

import csv
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.run_flashinfer_baseline import (
    BACKENDS,
    DEFAULT_CASES,
    DTYPES,
    EXPECTED_FLASHINFER_VERSION,
    FORMAL_REPEATS,
    FORMAL_WARMUP,
    QUICK_REPEATS,
    QUICK_WARMUP,
    TIMING_SCOPE,
    _logical_workload_bytes,
)
from benchmarks.summarize_flashinfer_baseline import (
    CASE_SHAPES,
    FlashInferBaselineValidationError,
    REQUIRED_FIELDS,
    aggregate,
    render_markdown,
    validate_rows,
)


def _rotate(values, offset):
    values = tuple(values)
    offset %= len(values)
    return values[offset:] + values[:offset]


def _row(
    dtype,
    case,
    trial,
    backend,
    trials=3,
    warmup=FORMAL_WARMUP,
    repeats=FORMAL_REPEATS,
):
    num_seqs, num_q_heads, num_kv_heads, head_dim, context_tokens = (
        CASE_SHAPES[case]
    )
    pages_per_seq = math.ceil(context_tokens / 32)
    backend_order = _rotate(BACKENDS, trial - 1)
    case_order = _rotate(DEFAULT_CASES, trial - 1)
    dtype_order = _rotate(DTYPES, trial - 1)
    p50_ms = {
        "flashdec_triton": 4.0,
        "flashinfer_fa2_cuda_core": 2.0,
        "flashinfer_fa2_tensor_core": 1.0,
    }[backend]
    logical_workload_bytes = _logical_workload_bytes(
        num_seqs=num_seqs,
        context_tokens=context_tokens,
        dtype_name=dtype,
    )
    digest_payload = f"{dtype}:{case}:{trial}".encode("ascii")
    row = {field: "1" for field in REQUIRED_FIELDS}
    row.update(
        {
            "name": "r5_flashinfer_paged_decode",
            "op": "paged_decode_attention",
            "date": "2026-07-23T12:00:00+08:00",
            "device": "NVIDIA GeForce RTX 5070",
            "python": "3.12.3",
            "torch": "2.11.0+cu128",
            "triton": "3.6.0",
            "cuda": "12.8",
            "git_commit": "abc1234",
            "git_worktree_clean": "True",
            "command": (
                "/usr/bin/python benchmarks/run_flashinfer_baseline.py "
                "--require-clean"
            ),
            "flashinfer_version": EXPECTED_FLASHINFER_VERSION,
            "expected_flashinfer_version": EXPECTED_FLASHINFER_VERSION,
            "flashinfer_workspace_mib": "128",
            "case": case,
            "dtype": dtype,
            "backend": backend,
            "flashinfer_backend": "fa2",
            "flashinfer_use_tensor_cores": {
                "flashdec_triton": "not_applicable",
                "flashinfer_fa2_cuda_core": "False",
                "flashinfer_fa2_tensor_core": "True",
            }[backend],
            "num_seqs": str(num_seqs),
            "num_q_heads": str(num_q_heads),
            "num_kv_heads": str(num_kv_heads),
            "head_dim": str(head_dim),
            "context_tokens": str(context_tokens),
            "min_seq_len": str(context_tokens),
            "max_seq_len": str(context_tokens),
            "block_size": "32",
            "pages_per_seq": str(pages_per_seq),
            "num_pages": str(num_seqs * pages_per_seq),
            "flashdec_kv_layout": "token_major",
            "flashinfer_kv_layout": "HND",
            "pos_encoding_mode": "NONE",
            "sm_scale": f"{128**-0.5:.12f}",
            "trial": str(trial),
            "trial_count": str(trials),
            "backend_order": "->".join(backend_order),
            "case_order": "->".join(case_order),
            "dtype_order": "->".join(dtype_order),
            "base_seed": "1701",
            "seed": str(1701 + trial - 1),
            "warmup": str(warmup),
            "repeats": str(repeats),
            "timing_scope": TIMING_SCOPE,
            "page_table_digest": hashlib.sha256(digest_payload).hexdigest(),
            "reference_sample_size": str(min(2, num_seqs)),
            "reference_validated": "True",
            "cross_backend_validated": "True",
            "max_abs_error_vs_reference": "0.01",
            "max_abs_error_vs_flashdec": (
                "0.0" if backend == "flashdec_triton" else "0.01"
            ),
            "max_tolerance_ratio_vs_reference": "0.5",
            "max_tolerance_ratio_vs_flashdec": (
                "0.0" if backend == "flashdec_triton" else "0.5"
            ),
            "rtol": "0.03" if dtype == "bfloat16" else "0.02",
            "atol": "0.03" if dtype == "bfloat16" else "0.02",
            "mean_ms": f"{p50_ms:.6f}",
            "p50_ms": f"{p50_ms:.6f}",
            "p90_ms": f"{p50_ms * 1.1:.6f}",
            "p99_ms": f"{p50_ms * 1.2:.6f}",
            "min_ms": f"{p50_ms * 0.9:.6f}",
            "max_ms": f"{p50_ms * 1.3:.6f}",
            "decode_tokens_per_second": f"{num_seqs * 1_000.0 / p50_ms:.3f}",
            "logical_workload_bytes": str(logical_workload_bytes),
            "logical_workload_gbps_p50": (
                f"{logical_workload_bytes / (p50_ms * 1_000_000.0):.4f}"
            ),
            "validated_invariants": "True",
        }
    )
    return row


def _rows(trials=3):
    rows = []
    for trial in range(1, trials + 1):
        for dtype in _rotate(DTYPES, trial - 1):
            for case in _rotate(DEFAULT_CASES, trial - 1):
                for backend in _rotate(BACKENDS, trial - 1):
                    rows.append(_row(dtype, case, trial, backend, trials=trials))
    return rows


class FlashInferBaselineSummaryTests(unittest.TestCase):
    def test_summary_cli_help_runs_without_gpu_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "benchmarks/summarize_flashinfer_baseline.py"),
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
        self.assertIn("--expected-warmup", result.stdout)
        self.assertIn("--expected-repeats", result.stdout)

    def test_summary_cli_validates_documented_quick_matrix(self):
        root = Path(__file__).resolve().parents[1]
        rows = [
            _row(
                "float16",
                "medium_b16_ctx1024",
                1,
                backend,
                trials=1,
                warmup=QUICK_WARMUP,
                repeats=QUICK_REPEATS,
            )
            for backend in BACKENDS
        ]
        for row in rows:
            row["case_order"] = "medium_b16_ctx1024"
            row["dtype_order"] = "float16"
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "quick.csv"
            output_path = Path(directory) / "quick_summary.md"
            with input_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "benchmarks/summarize_flashinfer_baseline.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--expected-trials",
                    "1",
                    "--expected-warmup",
                    "2",
                    "--expected-repeats",
                    "10",
                    "--expected-cases",
                    "medium_b16_ctx1024",
                    "--expected-dtypes",
                    "float16",
                ],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validated 3 rows", result.stdout)
            self.assertIn("Rows: 3; trials: 1", output_path.read_text())

    def test_validate_aggregate_and_render_complete_72_row_matrix(self):
        rows = _rows()
        self.assertEqual(validate_rows(rows), rows)
        aggregates = aggregate(rows)
        self.assertEqual(len(rows), 72)
        self.assertEqual(len(aggregates), 16)

        core = next(
            item
            for item in aggregates
            if item["dtype"] == "float16"
            and item["case"] == "small_b1_ctx128"
            and item["backend"] == "flashinfer_fa2_cuda_core"
        )
        tensor = next(
            item
            for item in aggregates
            if item["dtype"] == "float16"
            and item["case"] == "small_b1_ctx128"
            and item["backend"] == "flashinfer_fa2_tensor_core"
        )
        self.assertEqual(core["p50_ratio"], {"median": 2.0, "min": 2.0, "max": 2.0})
        self.assertEqual(core["tps_ratio"], {"median": 2.0, "min": 2.0, "max": 2.0})
        self.assertEqual(tensor["p50_ratio"]["median"], 4.0)

        markdown = render_markdown("results.csv", rows, aggregates)
        self.assertIn("Rows: 72; trials: 3", markdown)
        self.assertIn(f"`{EXPECTED_FLASHINFER_VERSION}`", markdown)
        self.assertIn("`HND`", markdown)
        self.assertIn("`token_major`", markdown)
        self.assertIn("FlashDec/external", markdown)
        self.assertIn("external/FlashDec", markdown)
        self.assertIn("Absolute Tail Percentiles", markdown)
        self.assertIn("FlashDec p90 ms", markdown)
        self.assertIn("FlashDec p99 ms", markdown)
        self.assertIn("Runner command", markdown)
        self.assertIn("no pass/fail performance or winner gate", markdown)

    def test_rejects_missing_unknown_or_duplicate_columns_and_rows(self):
        rows = _rows()
        del rows[0]["device"]
        with self.assertRaisesRegex(FlashInferBaselineValidationError, "missing"):
            validate_rows(rows)

        rows = _rows()
        rows[0]["unexpected"] = "value"
        with self.assertRaisesRegex(FlashInferBaselineValidationError, "unexpected"):
            validate_rows(rows)

        rows = _rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(FlashInferBaselineValidationError, "duplicate row"):
            validate_rows(rows)

    def test_rejects_matrix_global_version_shape_or_layout_drift(self):
        mutations = (
            (0, "torch", "old", "global field torch"),
            (None, "flashinfer_version", "0.6.14", "flashinfer_version"),
            (None, "flashinfer_workspace_mib", "64", "workspace"),
            (None, "git_worktree_clean", "False", "git_worktree_clean"),
            (None, "date", "2026-07-23T12:00:00", "timezone"),
            (None, "command", "python other.py", "command"),
            (0, "num_seqs", "2", "shape mismatch"),
            (None, "flashinfer_kv_layout", "NHD", "flashinfer_kv_layout"),
        )
        for index, field, value, message in mutations:
            with self.subTest(field=field):
                rows = _rows()
                targets = rows if index is None else (rows[index],)
                for row in targets:
                    row[field] = value
                with self.assertRaisesRegex(
                    FlashInferBaselineValidationError, message
                ):
                    validate_rows(rows)

        with self.assertRaisesRegex(
            FlashInferBaselineValidationError, "matrix incomplete"
        ):
            validate_rows(_rows()[:-1])

    def test_rejects_rotated_order_seed_or_physical_sequence_drift(self):
        rows = _rows()
        for row in rows:
            if row["trial"] == "2":
                row["backend_order"] = "->".join(BACKENDS)
        with self.assertRaisesRegex(
            FlashInferBaselineValidationError, "backend_order"
        ):
            validate_rows(rows)

        rows = _rows()
        for row in rows:
            if row["trial"] == "2":
                row["seed"] = "999"
        with self.assertRaisesRegex(
            FlashInferBaselineValidationError, "increase by one"
        ):
            validate_rows(rows)

        rows = _rows()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(
            FlashInferBaselineValidationError, "execution order"
        ):
            validate_rows(rows)

    def test_rejects_correctness_reference_sample_or_page_pair_drift(self):
        mutations = (
            ("reference_validated", "False", "reference_validated"),
            ("cross_backend_validated", "False", "cross_backend_validated"),
            ("reference_sample_size", "0", "reference_sample_size"),
            ("page_table_digest", "f" * 64, "paired input differs"),
            ("max_abs_error_vs_flashdec", "0.1", "self-comparison"),
            (
                "max_tolerance_ratio_vs_reference",
                "1.1",
                "exceeds the recorded tolerance",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                rows = _rows()
                rows[0][field] = value
                with self.assertRaisesRegex(
                    FlashInferBaselineValidationError, message
                ):
                    validate_rows(rows)

    def test_rejects_formal_sampling_strength_drift(self):
        for field, value, message in (
            ("warmup", "0", "warmup mismatch"),
            ("repeats", "1", "repeats mismatch"),
        ):
            with self.subTest(field=field):
                rows = _rows()
                for row in rows:
                    row[field] = value
                with self.assertRaisesRegex(
                    FlashInferBaselineValidationError,
                    message,
                ):
                    validate_rows(rows)

    def test_rejects_invalid_latency_tps_or_gbps(self):
        mutations = (
            ("p90_ms", "0.5", "percentile order"),
            ("decode_tokens_per_second", "0", "positive"),
            ("decode_tokens_per_second", "1", "decode TPS mismatch"),
            ("logical_workload_gbps_p50", "0", "positive"),
            ("logical_workload_gbps_p50", "1", "logical workload GB/s mismatch"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                rows = _rows()
                rows[0][field] = value
                with self.assertRaisesRegex(
                    FlashInferBaselineValidationError, message
                ):
                    validate_rows(rows)


if __name__ == "__main__":
    unittest.main()
