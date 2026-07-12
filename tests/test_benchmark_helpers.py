"""Dependency-free coverage for benchmark result and provenance helpers."""

from pathlib import Path
import re
import unittest

from flashdec.benchmark import git_commit, percentile, summarize_latencies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkHelperTests(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        values = [4.0, 1.0, 3.0, 2.0]

        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 50), 3.0)
        self.assertEqual(percentile(values, 100), 4.0)

    def test_summarize_latencies(self):
        result = summarize_latencies(
            "case",
            [1.0, 2.0, 3.0],
            metadata={"shape": "3"},
        )

        self.assertEqual(result.name, "case")
        self.assertEqual(result.mean_ms, 2.0)
        self.assertEqual(result.p50_ms, 2.0)
        self.assertEqual(result.p90_ms, 3.0)
        self.assertEqual(result.metadata, {"shape": "3"})
        self.assertEqual(result.as_row()["shape"], "3")

    def test_git_commit_binds_evidence_to_current_worktree(self):
        commit = git_commit(PROJECT_ROOT)

        self.assertRegex(commit, re.compile(r"^[0-9a-f]{7,40}$"))


if __name__ == "__main__":
    unittest.main()
