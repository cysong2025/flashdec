"""Pure-Python coverage for release artifact and version validation."""

from pathlib import Path

from scripts.check_release import (
    RELEASE_EVIDENCE_PATHS,
    REQUIRED_PATHS,
    _read_package_version,
    _read_project_version,
    validate_release_tree,
)


def _release_tree(tmp_path, project_version="0.1.0", package_version="0.1.0"):
    for relative in REQUIRED_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"flashdec\"\nversion = \"{}\"\n".format(project_version)
    )
    (tmp_path / "flashdec/__init__.py").write_text(
        "__version__ = {!r}\n".format(package_version)
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    (tmp_path / "docs/reproducibility.md").write_text(
        "# Reproducibility\n\n## Release gate status\n"
    )
    return tmp_path


def test_release_version_readers_support_pyproject_and_package(tmp_path):
    root = _release_tree(tmp_path)
    assert _read_project_version(root / "pyproject.toml") == "0.1.0"
    assert _read_package_version(root / "flashdec/__init__.py") == "0.1.0"


def test_validate_release_tree_accepts_complete_candidate(tmp_path):
    root = _release_tree(tmp_path)
    assert validate_release_tree(root) == []


def test_validate_release_tree_reports_missing_artifact_and_version_mismatch(tmp_path):
    root = _release_tree(tmp_path, project_version="0.1.0", package_version="0.0.0")
    missing = Path(root) / "benchmarks/profile_decode_engine.py"
    missing.unlink()

    problems = validate_release_tree(root)
    assert "missing required artifact: benchmarks/profile_decode_engine.py" in problems
    assert "version mismatch: pyproject=0.1.0, package=0.0.0" in problems


def test_validate_release_tree_requires_final_evidence_only_when_requested(tmp_path):
    root = _release_tree(tmp_path)
    assert validate_release_tree(root) == []

    problems = validate_release_tree(root, require_evidence=True)
    assert problems == [
        f"missing release evidence: {relative}" for relative in RELEASE_EVIDENCE_PATHS
    ]

    for relative in RELEASE_EVIDENCE_PATHS:
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("verified evidence\n")
    assert validate_release_tree(root, require_evidence=True) == []


def test_release_evidence_includes_scheduler_multi_layer_and_shared_prefix_summaries():
    assert (
        "benchmarks/results/r1_scheduler_workload_trials3_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/r2_multi_layer_engine_trials3_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/r3_shared_prefix_workload_trials8_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/r3_shared_prefix_workload_trials3_summary.md"
        not in RELEASE_EVIDENCE_PATHS
    )
