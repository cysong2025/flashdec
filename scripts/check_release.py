"""Validate FlashDec release artifacts, version consistency, and Git gates."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess


REQUIRED_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    ".github/workflows/quality.yml",
    "pyproject.toml",
    "flashdec/__init__.py",
    "flashdec/cache.py",
    "flashdec/engine.py",
    "flashdec/scheduler.py",
    "flashdec/workload.py",
    "docs/AI_INFRA_SCOPE.md",
    "docs/API.md",
    "docs/INDEX.md",
    "docs/PROJECT_PLAN.md",
    "docs/ROADMAP.md",
    "docs/design_multi_layer_kv_transaction.md",
    "docs/design_shared_prefix_blocks.md",
    "docs/design_scheduler.md",
    "docs/reproducibility.md",
    "benchmarks/run_decode_engine_workload.py",
    "benchmarks/summarize_decode_engine_trials.py",
    "benchmarks/profile_decode_engine.py",
    "benchmarks/run_scheduler_workload.py",
    "benchmarks/summarize_scheduler_workload.py",
    "benchmarks/run_multi_layer_engine.py",
    "benchmarks/summarize_multi_layer_trials.py",
    "benchmarks/run_shared_prefix_workload.py",
    "benchmarks/summarize_shared_prefix_trials.py",
    "benchmarks/run_fused_transaction_fast_path.py",
    "benchmarks/summarize_fused_transaction_fast_path.py",
    "scripts/check_env.py",
    "scripts/check_docs.py",
    "scripts/check_release.py",
    "scripts/run_r0_validation.py",
    "tests/test_paged_cache.py",
    "tests/test_docs_check.py",
    "tests/test_decode_engine.py",
    "tests/test_workload.py",
    "tests/test_scheduler.py",
    "tests/test_r0_validation.py",
    "tests/test_release_check.py",
    "tests/test_scheduler_workload_summary.py",
    "tests/test_multi_layer_transaction.py",
    "tests/test_shared_prefix_blocks.py",
    "tests/test_multi_layer_engine.py",
    "tests/test_multi_layer_workload_benchmark.py",
    "tests/test_multi_layer_workload_summary.py",
    "tests/test_shared_prefix_workload_benchmark.py",
    "tests/test_shared_prefix_workload_summary.py",
    "tests/test_fused_transaction_fast_path_benchmark.py",
    "tests/test_fused_transaction_fast_path_summary.py",
)

RELEASE_EVIDENCE_PATHS = (
    "benchmarks/results/week10_num_stages_summary.md",
    "benchmarks/results/week11_rope_kv_append_summary.md",
    "benchmarks/results/week12_decode_engine_workload_trials3_summary.md",
    "benchmarks/results/week12_decode_engine_profile_summary.md",
    "benchmarks/results/r1_scheduler_workload_trials3_summary.md",
    "benchmarks/results/r2_multi_layer_engine_trials3_summary.md",
    "benchmarks/results/r3_shared_prefix_workload_trials8_summary.md",
    "docs/performance_report.md",
)


def _read_project_version(path):
    text = Path(path).read_text()
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 fallback
        match = re.search(
            r"(?ms)^\[project\]\s*$.*?^version\s*=\s*['\"]([^'\"]+)['\"]\s*$",
            text,
        )
        if match is None:
            raise ValueError("pyproject.toml does not contain [project].version")
        return match.group(1)
    data = tomllib.loads(text)
    try:
        return str(data["project"]["version"])
    except KeyError as exc:
        raise ValueError("pyproject.toml does not contain [project].version") from exc


def _read_package_version(path):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                break
            return value
    raise ValueError("flashdec/__init__.py does not define a string __version__")


def validate_release_tree(root, require_evidence=False):
    """Return release-structure problems without mutating the repository."""
    root = Path(root)
    problems = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            problems.append(f"missing required artifact: {relative}")
    if require_evidence:
        for relative in RELEASE_EVIDENCE_PATHS:
            if not (root / relative).is_file():
                problems.append(f"missing release evidence: {relative}")

    pyproject = root / "pyproject.toml"
    package_init = root / "flashdec/__init__.py"
    if pyproject.is_file() and package_init.is_file():
        try:
            project_version = _read_project_version(pyproject)
            package_version = _read_package_version(package_init)
        except (OSError, SyntaxError, ValueError) as exc:
            problems.append(str(exc))
        else:
            if project_version != package_version:
                problems.append(
                    f"version mismatch: pyproject={project_version}, package={package_version}"
                )

    changelog = root / "CHANGELOG.md"
    if changelog.is_file() and "## [Unreleased]" not in changelog.read_text():
        problems.append("CHANGELOG.md must contain an [Unreleased] section")
    reproducibility = root / "docs/reproducibility.md"
    if reproducibility.is_file() and "## Release gate status" not in reproducibility.read_text():
        problems.append("docs/reproducibility.md must contain Release gate status")
    return problems


def _git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--require-tag", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    problems = validate_release_tree(
        root,
        require_evidence=args.require_evidence or args.require_tag,
    )
    project_version = _read_project_version(root / "pyproject.toml")
    package_version = _read_package_version(root / "flashdec/__init__.py")

    commit_code, commit = _git(root, "rev-parse", "--short", "HEAD")
    status_code, status = _git(root, "status", "--porcelain")
    if commit_code != 0 or status_code != 0:
        problems.append("root must be a readable Git worktree")
    if args.require_clean and status:
        problems.append("Git worktree is not clean")

    expected_tag = f"v{project_version}"
    tag_code, tags = _git(root, "tag", "--points-at", "HEAD")
    tag_set = set(tags.splitlines()) if tag_code == 0 and tags else set()
    if args.require_tag:
        if project_version == "0.0.0":
            problems.append("release version is still 0.0.0")
        elif expected_tag not in tag_set:
            problems.append(f"HEAD is not tagged {expected_tag}")

    print("FlashDec release check")
    print("======================")
    print(f"Root: {root}")
    print(f"Project/package version: {project_version} / {package_version}")
    print(f"Commit: {commit if commit_code == 0 else 'unavailable'}")
    print(f"Worktree clean: {status_code == 0 and not status}")
    print(f"Expected tag on HEAD: {expected_tag if args.require_tag else 'not required'}")
    if problems:
        print("Result: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
