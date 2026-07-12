"""Run reproducible FlashDec R0 validation phases and export result artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_ORDER = (
    "local",
    "focused",
    "full",
    "trials-quick",
    "trials-formal",
    "profile-quick",
    "profile-formal",
    "release",
)
ALL_EVIDENCE_PHASES = PHASE_ORDER[:-1]
GPU_PHASES = frozenset(
    {
        "focused",
        "full",
        "trials-quick",
        "trials-formal",
        "profile-quick",
        "profile-formal",
    }
)
FORMAL_EVIDENCE_PHASES = frozenset({"trials-formal", "profile-formal"})
PHASE_CHOICES = (*PHASE_ORDER, "all")


@dataclass(frozen=True)
class ValidationStep:
    phase: str
    name: str
    command: tuple[str, ...]
    artifacts: tuple[str, ...] = ()


def normalize_phases(phases):
    """Expand `all`, reject unknown phases, and return canonical execution order."""
    phases = tuple(phases)
    if not phases:
        raise ValueError("at least one validation phase is required")
    unknown = sorted(set(phases) - set(PHASE_CHOICES))
    if unknown:
        raise ValueError(f"unknown validation phases: {unknown}")
    if "all" in phases:
        if len(phases) != 1:
            raise ValueError("phase 'all' cannot be combined with another phase")
        return ALL_EVIDENCE_PHASES
    selected = set(phases)
    return tuple(phase for phase in PHASE_ORDER if phase in selected)


def build_validation_steps(python_executable, phases):
    """Build exact subprocess commands for selected validation phases."""
    python = str(python_executable)
    phases = normalize_phases(phases)
    focused_tests = (
        "tests/test_cuda_kv_append.py",
        "tests/test_fused_rope_kv_append.py",
        "tests/test_rope_append.py",
        "tests/test_paged_cache.py",
        "tests/test_multi_layer_transaction.py",
        "tests/test_multi_layer_engine.py",
        "tests/test_paged_decode.py",
        "tests/test_decode_engine.py",
        "tests/test_workload.py",
        "tests/test_scheduled_workload.py",
        "tests/test_scheduler_workload_benchmark.py",
        "tests/test_workload_benchmark.py",
        "tests/test_decode_engine_trial_summary.py",
        "tests/test_profile_decode_engine.py",
        "tests/test_public_api.py",
    )
    by_phase = {
        "local": (
            ValidationStep(
                "local",
                "scheduler-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_scheduler.py",
                    "-v",
                ),
            ),
            ValidationStep(
                "local",
                "scheduled-workload-config-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_scheduled_workload_config.py",
                    "-v",
                ),
            ),
            ValidationStep(
                "local",
                "scheduler-workload-summary-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_scheduler_workload_summary.py",
                    "-v",
                ),
            ),
            ValidationStep(
                "local",
                "benchmark-helper-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_benchmark_helpers.py",
                    "-v",
                ),
            ),
            ValidationStep(
                "local",
                "profile-helper-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_profile_decode_engine.py",
                    "-v",
                ),
            ),
            ValidationStep(
                "local",
                "orchestrator-unittest",
                (
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_r0_validation.py",
                    "-v",
                ),
            ),
        ),
        "focused": (
            ValidationStep(
                "focused",
                "gpu-focused-pytest",
                (python, "-m", "pytest", "-vv", *focused_tests),
            ),
        ),
        "full": (
            ValidationStep("full", "full-pytest", (python, "-m", "pytest", "-vv")),
        ),
        "trials-quick": (
            ValidationStep(
                "trials-quick",
                "trials2-quick-run",
                (
                    python,
                    "benchmarks/run_decode_engine_workload.py",
                    "--quick",
                    "--trials",
                    "2",
                    "--dtype",
                    "both",
                    "--output",
                    "benchmarks/results/week12_decode_engine_workload_trials2_quick.csv",
                ),
                ("benchmarks/results/week12_decode_engine_workload_trials2_quick.csv",),
            ),
            ValidationStep(
                "trials-quick",
                "trials2-quick-summary",
                (
                    python,
                    "benchmarks/summarize_decode_engine_trials.py",
                    "--input",
                    "benchmarks/results/week12_decode_engine_workload_trials2_quick.csv",
                    "--output",
                    "benchmarks/results/week12_decode_engine_workload_trials2_quick_summary.md",
                    "--expected-trials",
                    "2",
                ),
                ("benchmarks/results/week12_decode_engine_workload_trials2_quick_summary.md",),
            ),
        ),
        "trials-formal": (
            ValidationStep(
                "trials-formal",
                "trials3-formal-run",
                (
                    python,
                    "benchmarks/run_decode_engine_workload.py",
                    "--trials",
                    "3",
                    "--dtype",
                    "both",
                    "--output",
                    "benchmarks/results/week12_decode_engine_workload_trials3.csv",
                ),
                ("benchmarks/results/week12_decode_engine_workload_trials3.csv",),
            ),
            ValidationStep(
                "trials-formal",
                "trials3-formal-summary",
                (
                    python,
                    "benchmarks/summarize_decode_engine_trials.py",
                    "--input",
                    "benchmarks/results/week12_decode_engine_workload_trials3.csv",
                    "--output",
                    "benchmarks/results/week12_decode_engine_workload_trials3_summary.md",
                ),
                ("benchmarks/results/week12_decode_engine_workload_trials3_summary.md",),
            ),
        ),
        "profile-quick": (
            ValidationStep(
                "profile-quick",
                "profile-quick-run",
                (
                    python,
                    "benchmarks/profile_decode_engine.py",
                    "--workload",
                    "mixed_steady",
                    "--dtype",
                    "float16",
                    "--append-backends",
                    "torch",
                    "fused_cuda",
                    "--quick",
                    "--export-trace",
                    "--output-dir",
                    "benchmarks/profiles/week12_decode_engine_quick",
                    "--summary-output",
                    "benchmarks/results/week12_decode_engine_profile_quick_summary.md",
                ),
                ("benchmarks/results/week12_decode_engine_profile_quick_summary.md",),
            ),
        ),
        "profile-formal": (
            ValidationStep(
                "profile-formal",
                "profile-formal-run",
                (
                    python,
                    "benchmarks/profile_decode_engine.py",
                    "--workload",
                    "all",
                    "--dtype",
                    "both",
                    "--append-backends",
                    "torch",
                    "fused_cuda",
                    "--output-dir",
                    "benchmarks/profiles/week12_decode_engine",
                    "--summary-output",
                    "benchmarks/results/week12_decode_engine_profile_summary.md",
                ),
                ("benchmarks/results/week12_decode_engine_profile_summary.md",),
            ),
        ),
        "release": (
            ValidationStep(
                "release",
                "release-candidate-check",
                (
                    python,
                    "scripts/check_release.py",
                    "--require-clean",
                    "--require-evidence",
                ),
            ),
        ),
    }
    return tuple(step for phase in phases for step in by_phase[phase])


def _git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def require_tracked_worktree_clean(root):
    """Reject tracked changes and untracked files outside result artifact roots."""
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    problems = []
    for line in status.splitlines():
        if line.startswith("?? "):
            relative = line[3:]
            if relative.startswith(("benchmarks/results/", "benchmarks/profiles/")):
                continue
        problems.append(line)
    if problems:
        raise RuntimeError(
            "tracked worktree changes or untracked source files would make benchmark "
            "provenance incomplete; "
            "commit/stash them or pass --allow-dirty for non-release experiments"
        )


def validate_gpu_environment(env, which=shutil.which):
    """Validate the native-extension toolchain required by RTX evidence phases."""
    cuda_home = env.get("CUDA_HOME")
    if not cuda_home:
        raise RuntimeError("CUDA_HOME is required for GPU validation phases")
    cuda_home = Path(cuda_home)
    home_nvcc = cuda_home / "bin" / "nvcc"
    nvcc = str(home_nvcc) if home_nvcc.is_file() else which("nvcc")
    if not cuda_home.is_dir() or not nvcc:
        raise RuntimeError("CUDA_HOME must be valid and nvcc must be available")
    return nvcc


def artifact_signatures(root, step):
    signatures = {}
    for relative in step.artifacts:
        path = root / relative
        if path.is_file():
            stat = path.stat()
            signatures[relative] = (stat.st_mtime_ns, stat.st_size)
        else:
            signatures[relative] = None
    return signatures


def verify_step_artifacts(root, step, previous_signatures=None):
    missing = [relative for relative in step.artifacts if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"step {step.name} did not produce artifacts: {missing}")
    if previous_signatures is not None:
        current = artifact_signatures(root, step)
        stale = [
            relative
            for relative in step.artifacts
            if previous_signatures.get(relative) == current.get(relative)
        ]
        if stale:
            raise RuntimeError(f"step {step.name} did not update artifacts: {stale}")


def export_artifacts(root, steps, export_dir, *, dry_run=False):
    """Copy selected result artifacts to a host-visible directory once all steps pass."""
    export_dir = Path(export_dir)
    artifacts = []
    seen = set()
    for step in steps:
        for relative in step.artifacts:
            if relative not in seen:
                seen.add(relative)
                artifacts.append(relative)
    if not artifacts:
        return ()
    if not dry_run:
        export_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in artifacts:
        source = root / relative
        destination = export_dir / source.name
        print(f"[export] {source} -> {destination}", flush=True)
        if not dry_run:
            if not source.is_file():
                raise RuntimeError(f"cannot export missing artifact: {source}")
            shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def run_steps(root, steps, env, *, dry_run=False, continue_on_error=False):
    failures = []
    for index, step in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {step.phase}: {step.name}", flush=True)
        print(f"$ {shlex.join(step.command)}", flush=True)
        if dry_run:
            continue
        previous_signatures = artifact_signatures(root, step)
        result = subprocess.run(step.command, cwd=root, env=env, check=False)
        if result.returncode != 0:
            failures.append((step.name, f"exit code {result.returncode}"))
        else:
            try:
                verify_step_artifacts(root, step, previous_signatures)
            except RuntimeError as exc:
                failures.append((step.name, str(exc)))
        if failures and not continue_on_error:
            break
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        action="append",
        required=True,
        choices=PHASE_CHOICES,
        help="Repeat to select phases; 'all' runs every evidence phase except release.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--export-dir",
        help="Optional host-visible result directory, e.g. /mnt/c/Users/user/flashdec_results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        phases = normalize_phases(args.phase)
        steps = build_validation_steps(args.python, phases)
        commit = _git(PROJECT_ROOT, "rev-parse", "--short", "HEAD")
        if args.allow_dirty and any(phase in FORMAL_EVIDENCE_PHASES for phase in phases):
            raise ValueError("--allow-dirty is forbidden for formal evidence phases")
        if not args.dry_run and any(phase in GPU_PHASES for phase in phases):
            if not args.allow_dirty:
                require_tracked_worktree_clean(PROJECT_ROOT)
            env = dict(os.environ)
            env.setdefault("MAX_JOBS", "1")
            nvcc = validate_gpu_environment(env)
            print(
                f"GPU toolchain: CUDA_HOME={env['CUDA_HOME']}; nvcc={nvcc}",
                flush=True,
            )
        else:
            env = dict(os.environ)
        print(f"FlashDec R0 validation commit: {commit}", flush=True)
        print(f"Phases: {', '.join(phases)}", flush=True)
        failures = run_steps(
            PROJECT_ROOT,
            steps,
            env,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        if failures:
            for name, detail in failures:
                print(f"FAILED: {name}: {detail}", file=sys.stderr)
            return 1
        if args.export_dir:
            export_artifacts(
                PROJECT_ROOT,
                steps,
                args.export_dir,
                dry_run=args.dry_run,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"R0 validation error: {exc}", file=sys.stderr)
        return 1
    print("R0 validation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
