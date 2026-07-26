"""Dependency-free helper coverage for complete DecodeEngine profiling."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.profile_decode_engine import (
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
    PROFILE_RANGE_ENGINE_STEP,
    PROFILE_RANGE_PREFLIGHT,
    ProfileValidationError,
    _device_event_count,
    _dtype_names,
    _selected_workloads,
    _stage_rows,
    _write_summary,
    validate_profile_rows,
)
from benchmarks.run_decode_engine_workload import WORKLOADS


def _profile_row(workload="mixed_steady", dtype="float16", backend="fused_cuda"):
    return {
        "workload": workload,
        "dtype": dtype,
        "append_backend": backend,
        "steps": 10,
        "successful_steps": 8,
        "backpressure_steps": 2,
        "cuda_event_count": 24,
        "engine_count": 10,
        "preflight_count": 10,
        "append_count": 8,
        "decode_count": 8,
        "p50_ms": 1.25,
        "p99_ms": 1.75,
        "engine_cpu_ms": 10.0,
        "engine_device_ms": 30.0,
        "append_device_ms": 8.0,
        "decode_device_ms": 22.0,
        "profile": "profile.txt",
        "trace": "trace.json",
        "git_commit": "abc1234",
    }


class DecodeEngineProfileHelperTests(unittest.TestCase):
    def test_selects_workloads_and_dtypes(self):
        self.assertEqual(_dtype_names("float16"), ["float16"])
        self.assertEqual(_dtype_names("both"), ["float16", "bfloat16"])
        self.assertEqual(
            _selected_workloads("mixed_steady"),
            [WORKLOADS["mixed_steady"]],
        )
        self.assertEqual(_selected_workloads("all"), list(WORKLOADS.values()))

    def test_stage_rows_support_device_time_and_missing_ranges(self):
        events = [
            SimpleNamespace(
                key=PROFILE_RANGE_ENGINE_STEP,
                count=3,
                cpu_time_total=3_000.0,
                self_cpu_time_total=300.0,
                device_time_total=2_400.0,
                self_device_time_total=100.0,
            ),
            SimpleNamespace(
                key=PROFILE_RANGE_APPEND,
                count=3,
                cpu_time_total=1_000.0,
                self_cpu_time_total=200.0,
                device_time_total=800.0,
                self_device_time_total=800.0,
            ),
        ]

        rows = {row["range"]: row for row in _stage_rows(events)}
        self.assertEqual(
            rows[PROFILE_RANGE_ENGINE_STEP],
            {
                "range": PROFILE_RANGE_ENGINE_STEP,
                "count": 3,
                "cpu_total_ms": 3.0,
                "cpu_self_ms": 0.3,
                "device_total_ms": 2.4,
                "device_self_ms": 0.1,
            },
        )
        self.assertEqual(rows[PROFILE_RANGE_APPEND]["device_total_ms"], 0.8)
        self.assertEqual(rows[PROFILE_RANGE_PREFLIGHT]["count"], 0)
        self.assertEqual(rows[PROFILE_RANGE_DECODE]["device_total_ms"], 0.0)

    def test_device_event_count_uses_cuda_device_records_only(self):
        events = [
            SimpleNamespace(device_type="DeviceType.CPU", count=10),
            SimpleNamespace(device_type="DeviceType.CUDA", count=7),
            SimpleNamespace(device_type="CUDA", count=3),
        ]
        self.assertEqual(_device_event_count(events), 10)

    def test_validate_profile_rows_accepts_complete_matrix(self):
        workloads = ("short_churn", "mixed_steady")
        dtypes = ("float16", "bfloat16")
        backends = ("torch", "fused_cuda")
        rows = [
            _profile_row(workload, dtype, backend)
            for workload in workloads
            for dtype in dtypes
            for backend in backends
        ]

        self.assertEqual(
            validate_profile_rows(rows, workloads, dtypes, backends),
            rows,
        )

    def test_validate_profile_rows_rejects_incomplete_matrix(self):
        rows = [_profile_row(backend="torch")]

        with self.assertRaisesRegex(ProfileValidationError, "incomplete"):
            validate_profile_rows(
                rows,
                ("mixed_steady",),
                ("float16",),
                ("torch", "fused_cuda"),
            )

    def test_validate_profile_rows_rejects_range_or_cuda_evidence_drift(self):
        cases = (
            ("engine_count", 9, "engine range count"),
            ("preflight_count", 9, "preflight range count"),
            ("append_count", 7, "append/decode range counts"),
            ("decode_count", 7, "append/decode range counts"),
            ("cuda_event_count", 0, "cuda_event_count"),
            ("p50_ms", float("nan"), "positive and finite"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                row = _profile_row()
                row[field] = value
                with self.assertRaisesRegex(ProfileValidationError, message):
                    validate_profile_rows(
                        [row],
                        ("mixed_steady",),
                        ("float16",),
                        ("fused_cuda",),
                    )

    def test_validate_profile_rows_rejects_mixed_commits(self):
        torch_row = _profile_row(backend="torch")
        fused_row = _profile_row(backend="fused_cuda")
        fused_row["git_commit"] = "different"

        with self.assertRaisesRegex(ProfileValidationError, "inconsistent git commits"):
            validate_profile_rows(
                [torch_row, fused_row],
                ("mixed_steady",),
                ("float16",),
                ("torch", "fused_cuda"),
            )

    def test_write_summary_records_commit_counts_and_stage_boundaries(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            _write_summary(output, [_profile_row()])
            text = output.read_text()

        self.assertTrue(text.startswith("# DecodeEngine Stage Profile Summary\n"))
        self.assertIn("mixed_steady", text)
        self.assertIn("fused_cuda", text)
        self.assertIn("`abc1234`", text)
        self.assertIn("10/10/8/8", text)
        self.assertIn("1.250000", text)
        self.assertIn("| 24 |", text)
        self.assertIn("8.000000", text)
        self.assertIn("22.000000", text)
        self.assertIn("profile.txt", text)
        self.assertIn("trace.json", text)
        self.assertIn("must not be added blindly", text)


if __name__ == "__main__":
    unittest.main()
