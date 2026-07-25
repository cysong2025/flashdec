"""Pure-Python coverage for release artifact and version validation."""

from pathlib import Path

from benchmarks.run_flashinfer_baseline import (
    EXPECTED_CUDA_BINDINGS_VERSION,
    EXPECTED_CUDA_PATHFINDER_VERSION,
    EXPECTED_CUDA_PYTHON_VERSION,
    EXPECTED_CUDA_TOOLKIT_VERSION,
    EXPECTED_FLASHINFER_VERSION,
    EXPECTED_NINJA_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRITON_VERSION,
)
from scripts.check_release import (
    RELEASE_EVIDENCE_PATHS,
    REQUIRED_PATHS,
    R5_CONSTRAINT_PINS,
    _read_constraint_pins,
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
    (tmp_path / "constraints/r5-cu128.txt").write_text(
        "".join(
            f"{name}=={version}\n"
            for name, version in R5_CONSTRAINT_PINS.items()
        )
    )
    return tmp_path


def test_release_version_readers_support_pyproject_and_package(tmp_path):
    root = _release_tree(tmp_path)
    assert _read_project_version(root / "pyproject.toml") == "0.1.0"
    assert _read_package_version(root / "flashdec/__init__.py") == "0.1.0"
    assert _read_constraint_pins(root / "constraints/r5-cu128.txt") == (
        R5_CONSTRAINT_PINS
    )


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


def test_validate_release_tree_rejects_r5_constraint_drift(tmp_path):
    root = _release_tree(tmp_path)
    constraints = root / "constraints/r5-cu128.txt"
    constraints.write_text(constraints.read_text().replace("3.6.0", "3.7.1"))

    problems = validate_release_tree(root)
    assert (
        "R5 constraint mismatch: triton='3.7.1', expected '3.6.0'" in problems
    )


def test_r5_release_constraints_match_runner_environment_contract():
    assert R5_CONSTRAINT_PINS == {
        "torch": EXPECTED_TORCH_VERSION,
        "triton": EXPECTED_TRITON_VERSION,
        "flashinfer-python": EXPECTED_FLASHINFER_VERSION,
        "cuda-toolkit": EXPECTED_CUDA_TOOLKIT_VERSION,
        "cuda-python": EXPECTED_CUDA_PYTHON_VERSION,
        "cuda-bindings": EXPECTED_CUDA_BINDINGS_VERSION,
        "cuda-pathfinder": EXPECTED_CUDA_PATHFINDER_VERSION,
        "ninja": EXPECTED_NINJA_VERSION,
    }


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


def test_release_tree_requires_r4_runner_validator_and_dependency_free_tests():
    for relative in (
        "benchmarks/run_fused_transaction_fast_path.py",
        "benchmarks/summarize_fused_transaction_fast_path.py",
        "tests/test_fused_transaction_fast_path_benchmark.py",
        "tests/test_fused_transaction_fast_path_summary.py",
    ):
        assert relative in REQUIRED_PATHS
    assert (
        "benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )


def test_release_tree_requires_r4c_implementation_and_formal_evidence():
    for relative in (
        "flashdec/integrated_workload.py",
        "docs/design_integrated_scheduled_multi_layer.md",
        "benchmarks/run_integrated_scheduled_multi_layer.py",
        "benchmarks/summarize_integrated_scheduled_multi_layer.py",
        "tests/test_integrated_workload.py",
        "tests/test_integrated_workload_config.py",
        "tests/test_integrated_workload_benchmark.py",
        "tests/test_integrated_workload_summary.py",
    ):
        assert relative in REQUIRED_PATHS
    assert (
        "benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )


def test_release_tree_requires_r5_baseline_implementation_but_not_pending_evidence():
    for relative in (
        "constraints/r5-cu128.txt",
        "docs/design_flashinfer_baseline.md",
        "docs/notes/from_paged_attention_to_decode_runtime.md",
        "benchmarks/run_flashinfer_baseline.py",
        "benchmarks/summarize_flashinfer_baseline.py",
        "tests/test_flashinfer_baseline.py",
        "tests/test_flashinfer_baseline_benchmark.py",
        "tests/test_flashinfer_baseline_summary.py",
    ):
        assert relative in REQUIRED_PATHS
    assert (
        "benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md"
        not in RELEASE_EVIDENCE_PATHS
    )
