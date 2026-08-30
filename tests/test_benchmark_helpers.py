"""Dependency-free coverage for benchmark result and provenance helpers."""

from io import StringIO
from pathlib import Path
import re
import tempfile
import unittest

from flashdec.benchmark import (
    git_commit,
    percentile,
    summarize_latencies,
    validate_vllm_cache_root,
    vllm_cache_root_for_commit,
    write_vllm_cache_log_metadata,
)


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

    def test_vllm_cache_root_is_commit_scoped_and_non_destructive(self):
        commit = "abc1234"
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_root = Path(temp_dir) / "legacy-cache"
            legacy_root.mkdir()
            marker = legacy_root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            cache_root = vllm_cache_root_for_commit(
                commit,
                environ={"VLLM_CACHE_ROOT": str(legacy_root)},
            )

            self.assertEqual(cache_root, legacy_root.resolve() / commit)
            self.assertEqual(
                validate_vllm_cache_root(cache_root, commit), cache_root
            )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_flashdec_cache_base_overrides_parent_vllm_cache_root(self):
        commit = "abc1234"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_root = vllm_cache_root_for_commit(
                commit,
                environ={
                    "FLASHDEC_VLLM_CACHE_BASE": str(temp_path / "flashdec"),
                    "VLLM_CACHE_ROOT": str(temp_path / "generic"),
                },
            )

            self.assertEqual(
                cache_root, (temp_path / "flashdec").resolve() / commit
            )

    def test_vllm_cache_log_metadata_records_actual_root(self):
        stream = StringIO()
        cache_root = Path("/tmp/vllm-flashdec/abc1234")

        write_vllm_cache_log_metadata(
            stream,
            commit="abc1234",
            cache_root=cache_root,
            command="vllm bench latency",
        )

        self.assertEqual(
            stream.getvalue(),
            "flashdec_git_commit=abc1234\n"
            "VLLM_CACHE_ROOT=/tmp/vllm-flashdec/abc1234\n"
            "command=vllm bench latency\n",
        )

    def test_vllm_cache_root_rejects_non_commit_namespace(self):
        with self.assertRaisesRegex(ValueError, "Git commit SHA"):
            vllm_cache_root_for_commit("unknown")


if __name__ == "__main__":
    unittest.main()
