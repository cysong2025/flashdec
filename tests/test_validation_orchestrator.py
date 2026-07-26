"""Dependency-free coverage for the validation orchestrator."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_validation import (
    ALL_EVIDENCE_PHASES,
    ValidationStep,
    artifact_signatures,
    build_validation_steps,
    export_artifacts,
    main,
    normalize_phases,
    require_tracked_worktree_clean,
    run_steps,
    validate_gpu_environment,
    verify_step_artifacts,
)


class ValidationOrchestratorTests(unittest.TestCase):
    def test_all_expands_to_evidence_phases_without_release(self):
        phases = normalize_phases(["all"])

        self.assertEqual(phases, ALL_EVIDENCE_PHASES)
        self.assertNotIn("release", phases)

    def test_phase_selection_is_deduplicated_and_canonical(self):
        phases = normalize_phases(
            ["profile-formal", "local", "trials-formal", "local"]
        )

        self.assertEqual(phases, ("local", "trials-formal", "profile-formal"))

    def test_all_cannot_be_combined_with_another_phase(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            normalize_phases(["all", "local"])

    def test_formal_evidence_forbids_allow_dirty(self):
        stderr = io.StringIO()
        with patch(
            "sys.argv",
            [
                "run_validation.py",
                "--phase",
                "trials-formal",
                "--allow-dirty",
            ],
        ), redirect_stderr(stderr):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("forbidden for formal evidence", stderr.getvalue())

    def test_build_local_steps_uses_selected_python(self):
        steps = build_validation_steps("/venv/bin/python", ["local"])

        self.assertEqual(len(steps), 6)
        self.assertTrue(all(step.command[0] == "/venv/bin/python" for step in steps))
        self.assertEqual(
            [step.name for step in steps],
            [
                "scheduler-unittest",
                "scheduled-workload-config-unittest",
                "scheduler-workload-summary-unittest",
                "benchmark-helper-unittest",
                "profile-helper-unittest",
                "orchestrator-unittest",
            ],
        )

    def test_quick_trial_phase_builds_run_then_strict_summary(self):
        steps = build_validation_steps("python", ["trials-quick"])

        self.assertEqual([step.name for step in steps], [
            "trials2-quick-run",
            "trials2-quick-summary",
        ])
        self.assertIn("--quick", steps[0].command)
        self.assertIn("--expected-trials", steps[1].command)
        index = steps[1].command.index("--expected-trials")
        self.assertEqual(steps[1].command[index + 1], "2")
        self.assertEqual(
            steps[1].artifacts,
            ("benchmarks/results/decode_engine_workload_trials2_quick_summary.md",),
        )

    def test_formal_profile_phase_targets_release_evidence_path(self):
        step = build_validation_steps("python", ["profile-formal"])[0]

        self.assertIn("--workload", step.command)
        self.assertIn("all", step.command)
        self.assertIn("both", step.command)
        self.assertEqual(
            step.artifacts,
            ("benchmarks/results/decode_engine_stage_profile_summary.md",),
        )

    def test_release_phase_requires_clean_tree_and_evidence(self):
        step = build_validation_steps("python", ["release"])[0]

        self.assertIn("--require-clean", step.command)
        self.assertIn("--require-evidence", step.command)

    def test_gpu_environment_accepts_cuda_home_nvcc(self):
        with TemporaryDirectory() as directory:
            cuda_home = Path(directory) / "cuda"
            nvcc = cuda_home / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("placeholder")

            detected = validate_gpu_environment(
                {"CUDA_HOME": str(cuda_home)},
                which=lambda _: None,
            )

        self.assertEqual(detected, str(nvcc))

    def test_gpu_environment_rejects_missing_cuda_home(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA_HOME"):
            validate_gpu_environment({}, which=lambda _: None)

    def test_tracked_worktree_preflight_rejects_staged_source_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "-q"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            source = root / "source.py"
            source.write_text("changed\n")
            subprocess.run(
                ["git", "add", "source.py"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with self.assertRaisesRegex(RuntimeError, "tracked worktree changes"):
                require_tracked_worktree_clean(root)

    def test_worktree_preflight_allows_results_but_rejects_untracked_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "-q"],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            results = root / "benchmarks" / "results"
            results.mkdir(parents=True)
            (results / "generated.md").write_text("result\n")
            require_tracked_worktree_clean(root)

            (root / "untracked_source.py").write_text("source\n")
            with self.assertRaisesRegex(RuntimeError, "untracked source files"):
                require_tracked_worktree_clean(root)

    def test_verify_step_artifacts_rejects_missing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            step = ValidationStep(
                "test",
                "missing",
                ("python", "missing.py"),
                ("result.csv",),
            )
            with self.assertRaisesRegex(RuntimeError, "did not produce artifacts"):
                verify_step_artifacts(root, step)

    def test_verify_step_artifacts_rejects_stale_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.csv"
            output.write_text("old\n")
            step = ValidationStep(
                "test",
                "stale",
                ("python", "producer.py"),
                ("result.csv",),
            )
            before = artifact_signatures(root, step)

            with self.assertRaisesRegex(RuntimeError, "did not update artifacts"):
                verify_step_artifacts(root, step, before)

    def test_export_artifacts_copies_unique_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            output = Path(directory) / "export"
            first = root / "results" / "a.csv"
            second = root / "results" / "b.md"
            first.parent.mkdir(parents=True)
            first.write_text("a")
            second.write_text("b")
            steps = (
                ValidationStep("x", "one", ("true",), ("results/a.csv",)),
                ValidationStep(
                    "x",
                    "two",
                    ("true",),
                    ("results/a.csv", "results/b.md"),
                ),
            )

            with redirect_stdout(io.StringIO()):
                copied = export_artifacts(root, steps, output)

            self.assertEqual(len(copied), 2)
            self.assertEqual((output / "a.csv").read_text(), "a")
            self.assertEqual((output / "b.md").read_text(), "b")

    def test_dry_run_does_not_execute_commands_or_require_artifacts(self):
        step = ValidationStep(
            "test",
            "dry",
            ("this-command-must-not-exist",),
            ("missing.csv",),
        )

        with redirect_stdout(io.StringIO()):
            failures = run_steps(Path.cwd(), (step,), {}, dry_run=True)

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
