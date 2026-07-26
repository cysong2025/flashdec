"""Dependency-free coverage for canonical public summary output paths."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from benchmarks.profile_paged_decode import parse_args as parse_paged_decode_profile
from benchmarks.summarize_flashinfer_baseline import parse_args as parse_flashinfer
from benchmarks.summarize_fused_transaction_fast_path import (
    parse_args as parse_fused_transaction,
)
from benchmarks.summarize_integrated_scheduled_multi_layer import (
    parse_args as parse_integrated,
)
from benchmarks.summarize_multi_layer_trials import parse_args as parse_multi_layer
from benchmarks.summarize_scheduler_workload import parse_args as parse_scheduler
from benchmarks.summarize_shared_prefix_trials import parse_args as parse_shared_prefix


class SummaryDefaultPathTests(unittest.TestCase):
    def _parse(self, parser):
        with patch.object(sys, "argv", ["summarizer", "--input", "input.csv"]):
            return parser()

    def test_default_outputs_use_canonical_topic_names(self):
        expected = (
            (parse_scheduler, "benchmarks/results/scheduler_capacity_progress_summary.md"),
            (parse_multi_layer, "benchmarks/results/multi_layer_transaction_summary.md"),
            (
                parse_shared_prefix,
                "benchmarks/results/shared_prefix_pre_metadata_cache_summary.md",
            ),
            (
                parse_fused_transaction,
                "benchmarks/results/trusted_transaction_summary.md",
            ),
            (
                parse_integrated,
                "benchmarks/results/integrated_runtime_lifecycle_summary.md",
            ),
            (
                parse_flashinfer,
                "benchmarks/results/flashinfer_paged_decode_baseline_summary.md",
            ),
        )
        for parser, output in expected:
            with self.subTest(output=output):
                self.assertEqual(self._parse(parser).output, output)

    def test_paged_decode_profile_uses_canonical_output_directory(self):
        with patch.object(sys, "argv", ["profile_paged_decode"]):
            args = parse_paged_decode_profile()
        self.assertEqual(args.output_dir, "benchmarks/profiles/paged_decode_default")
        self.assertEqual(
            args.summary_output,
            "benchmarks/results/paged_decode_default_profile_summary.md",
        )


if __name__ == "__main__":
    unittest.main()
