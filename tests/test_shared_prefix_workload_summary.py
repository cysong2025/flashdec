"""Dependency-free strict validation coverage for R3-C evidence."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.summarize_shared_prefix_trials import (
    DTYPES,
    HIT_RATES,
    SharedPrefixValidationError,
    validate_rows,
    write_summary,
)


ADMISSIONS = {0: 9, 25: 12, 50: 15, 75: 16}
COMMITTED = {0: 45, 25: 48, 50: 47, 75: 36}


def _rows(trials=3):
    rows = []
    request_count = 16
    prefix_blocks = 4
    tail_blocks = 1
    bytes_per_block = 1_048_576
    logical_context = request_count * prefix_blocks
    for dtype in DTYPES:
        for trial in range(1, trials + 1):
            offset = (trial - 1) % len(HIT_RATES)
            order = HIT_RATES[offset:] + HIT_RATES[:offset]
            for rate in HIT_RATES:
                hits = request_count * rate // 100
                misses = request_count - hits
                physical_context = (
                    logical_context
                    if hits == 0
                    else (misses + 1) * prefix_blocks
                )
                saved = max(hits - 1, 0) * prefix_blocks
                peak = physical_context + request_count * tail_blocks
                prefix_latency = "0" if hits == 0 else "1.0"
                rows.append(
                    {
                        "name": "shared_prefix_workload",
                        "op": "shared_prefix_decode_workload",
                        "device": "test-gpu",
                        "torch": "2.test",
                        "cuda": "12.8",
                        "git_commit": "abc1234",
                        "append_backend": "fused_cuda",
                        "decode_backend": "triton",
                        "num_layers": "1",
                        "num_q_heads": "32",
                        "num_kv_heads": "8",
                        "head_dim": "128",
                        "block_size": "32",
                        "num_warps": "2",
                        "request_count": str(request_count),
                        "context_tokens": "128",
                        "prefix_blocks": str(prefix_blocks),
                        "decode_tokens": "12",
                        "tail_blocks": str(tail_blocks),
                        "warmup": "2",
                        "trial_count": str(trials),
                        "capacity_ratio": "0.6000",
                        "capacity_probe_blocks": "48",
                        "latency_max_blocks": "80",
                        "bytes_per_block": str(bytes_per_block),
                        "logical_context_blocks": str(logical_context),
                        "timing_scope": "test timing",
                        "dtype": dtype,
                        "hit_rate_percent": str(rate),
                        "hit_count": str(hits),
                        "miss_count": str(misses),
                        "trial": str(trial),
                        "hit_rate_order": "->".join(str(value) for value in order),
                        "seed": str(612 + trial),
                        "repeats": "10",
                        "capacity_admitted_requests": str(ADMISSIONS[rate]),
                        "capacity_waiting_requests": str(request_count - ADMISSIONS[rate]),
                        "capacity_rejected_requests": "0",
                        "capacity_admission_rate": str(ADMISSIONS[rate] / request_count),
                        "capacity_committed_blocks": str(COMMITTED[rate]),
                        "capacity_physical_blocks": str(prefix_blocks if hits else 0),
                        "physical_context_blocks": str(physical_context),
                        "physical_context_bytes": str(physical_context * bytes_per_block),
                        "context_memory_saving_ratio": str(1.0 - physical_context / logical_context),
                        "peak_used_blocks": str(peak),
                        "peak_allocated_kv_bytes": str(peak * bytes_per_block),
                        "resident_prefix_blocks": str(prefix_blocks if hits else 0),
                        "active_prefix_references": str(hits),
                        "saved_prefix_blocks": str(saved),
                        "saved_prefix_bytes": str(saved * bytes_per_block),
                        "prefix_hit_count": str(hits),
                        "prefix_miss_count": "0",
                        "prefix_eviction_count": str(int(hits > 0)),
                        "registration_ms": prefix_latency,
                        "attach_mean_us": prefix_latency,
                        "attach_p50_us": prefix_latency,
                        "attach_p90_us": prefix_latency,
                        "eviction_us": prefix_latency,
                        "scheduler_p50_ms": "0.1",
                        "engine_step_p50_ms": "1.0",
                        "engine_step_p90_ms": "1.1",
                        "engine_step_p99_ms": "1.2",
                        "mean_ms": "1.1",
                        "p50_ms": "1.1",
                        "p90_ms": "1.2",
                        "complete_step_p99_ms": "1.3",
                        "decode_tokens_per_second": "1000.0",
                        "final_free_blocks": "80",
                        "validated_invariants": "True",
                    }
                )
    return rows


class SharedPrefixWorkloadSummaryTests(unittest.TestCase):
    def test_validator_accepts_complete_auditable_matrix(self):
        rows = _rows()
        self.assertEqual(validate_rows(rows), rows)

    def test_validator_rejects_missing_duplicate_and_bad_accounting(self):
        with self.assertRaisesRegex(SharedPrefixValidationError, "matrix mismatch"):
            validate_rows(_rows()[:-1])

        duplicate = _rows()
        duplicate.append(dict(duplicate[0]))
        with self.assertRaisesRegex(SharedPrefixValidationError, "duplicate"):
            validate_rows(duplicate)

        invalid = _rows()
        invalid[1]["saved_prefix_blocks"] = "999"
        with self.assertRaisesRegex(SharedPrefixValidationError, "saved_prefix_blocks"):
            validate_rows(invalid)

    def test_validator_rejects_capacity_admission_not_derived_from_fifo_commitments(self):
        invalid = _rows()
        target = next(
            row
            for row in invalid
            if row["dtype"] == "float16"
            and row["trial"] == "1"
            and row["hit_rate_percent"] == "25"
        )
        target["capacity_admitted_requests"] = "8"
        target["capacity_waiting_requests"] = "8"
        target["capacity_admission_rate"] = "0.5"
        with self.assertRaisesRegex(SharedPrefixValidationError, "FIFO admission"):
            validate_rows(invalid)

    def test_summary_reports_memory_admission_and_latency_without_speedup_claim(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            write_summary(_rows(), output, "input.csv")
            text = output.read_text()
        self.assertIn("Rows: 24", text)
        self.assertIn("context physical/logical blocks", text)
        self.assertIn("saved KV-capacity MiB", text)
        self.assertIn("75%", text)
        self.assertIn("Paired vs 0% Hit Rate", text)
        self.assertIn("Paired Latency Attribution vs 0%", text)
        self.assertIn("1.0000x [1.0000,1.0000]", text)
        self.assertIn("not a direct process-VRAM measurement", text)
        self.assertIn("system noise", text)

    def test_validator_and_summary_support_eight_trial_confirmation(self):
        rows = _rows(trials=8)
        self.assertEqual(validate_rows(rows, expected_trials=8), rows)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.md"
            write_summary(
                rows,
                output,
                "input.csv",
                expected_trials=8,
            )
            text = output.read_text()
        self.assertIn("Rows: 64; trials: 8.", text)


if __name__ == "__main__":
    unittest.main()
