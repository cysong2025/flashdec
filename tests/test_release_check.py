"""Pure-Python coverage for release artifact and version validation."""

import base64
from pathlib import Path
import zlib

import pytest

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
from benchmarks.run_vllm_model_latency import (
    SCHEMA_VERSION as MODEL_LATENCY_CSV_SCHEMA_VERSION,
    TIMING_SCOPE as MODEL_LATENCY_TIMING_SCOPE,
)
from benchmarks.run_vllm_model_latency_worker import (
    RESULT_SCHEMA_VERSION as MODEL_LATENCY_WORKER_SCHEMA_VERSION,
    TIMING_SCOPE as MODEL_LATENCY_WORKER_TIMING_SCOPE,
)
from benchmarks.summarize_vllm_model_latency import (
    FORMAL_CASE_SHAPES,
    FORMAL_PRIME_ITERS,
    GUARDRAIL_CASE,
    GUARDRAIL_RATIO_LIMIT,
    TARGET_CASE,
    TARGET_RATIO_LIMIT,
    TIMING_SCOPE as MODEL_LATENCY_SUMMARY_TIMING_SCOPE,
)
from scripts.check_docs import PUBLIC_ENTRY_FILES
from scripts.check_release import (
    PUBLIC_RELEASE_REQUIRED_PATHS,
    RELEASE_EVIDENCE_PATHS,
    REQUIRED_PATHS,
    WARP_SELECTION_EVIDENCE_MARKERS,
    WARP_SELECTION_EVIDENCE_PATH,
    FLASHINFER_CONSTRAINT_PINS,
    _read_constraint_pins,
    _license_id_from_text,
    _read_package_version,
    _read_project_license,
    _read_project_version,
    public_release_problems,
    validate_release_tree,
)


MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 FlashDec contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Canonical Apache-2.0 text (11,358 bytes), compressed so the fixture does not
# dominate this test module. The production gate compares its whitespace-
# normalized SHA-256, so this expands to an independent full positive fixture.
APACHE_LICENSE_TEXT = zlib.decompress(
    base64.b64decode(
        """
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
"""
    )
).decode("utf-8")


def _release_tree(tmp_path, project_version="0.1.0", package_version="0.1.0"):
    for relative in REQUIRED_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools>=68", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        'name = "flashdec"\n'
        'version = "{}"\n'.format(project_version)
    )
    (tmp_path / "flashdec/__init__.py").write_text(
        "__version__ = {!r}\n".format(package_version)
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    (tmp_path / "docs/reproducibility.md").write_text(
        "# Reproducibility\n\n## 已知安装与版本限制\n"
    )
    (tmp_path / "constraints/flashinfer-cu128.txt").write_text(
        "".join(
            f"{name}=={version}\n"
            for name, version in FLASHINFER_CONSTRAINT_PINS.items()
        )
    )
    (tmp_path / "SECURITY.md").write_text(
        "# Security Policy\n\n"
        "## Reporting a vulnerability\n\n"
        "Use private vulnerability reporting.\n"
    )
    (tmp_path / "CODE_OF_CONDUCT.md").write_text(
        "# FlashDec Code of Conduct\n\n"
        "## Expected behavior\n\n"
        "Be constructive.\n\n"
        "## Reporting and enforcement\n\n"
        "Report privately.\n"
    )
    (tmp_path / "SUPPORT.md").write_text(
        "# Support\n\nChoose the appropriate issue form or read SECURITY.md.\n"
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "title: FlashDec\n"
        "type: software\n"
        "authors:\n"
        "  - name: FlashDec contributors\n"
        "repository-code: https://github.com/cysong2025/flashdec\n"
        f"version: {project_version}\n"
    )
    (tmp_path / ".github/CODEOWNERS").write_text("* @cysong2025\n")
    (tmp_path / ".github/dependabot.yml").write_text(
        "version: 2\n"
        "updates:\n"
        "  - package-ecosystem: \"github-actions\"\n"
        "  - package-ecosystem: \"pip\"\n"
    )
    (tmp_path / ".github/ISSUE_TEMPLATE/question.yml").write_text(
        'name: "Usage / environment question"\n'
        "description: Ask for help.\n"
        "body:\n"
        "  - type: markdown\n"
        "    attributes:\n"
        "      value: Remove credentials, private paths, and unrelated logs.\n"
    )
    return tmp_path


def _configure_public_release(root, license_id, license_text):
    link_label = {
        "MIT": "MIT License",
        "Apache-2.0": "Apache License 2.0",
    }[license_id]
    (root / "LICENSE").write_text(license_text)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        .replace("setuptools>=68", "setuptools>=77")
        .replace(
            "[project]\n",
            f'[project]\nlicense = "{license_id}"\n'
            'license-files = ["LICENSE"]\n',
        )
    )
    citation = root / "CITATION.cff"
    citation.write_text(citation.read_text() + f"license: {license_id}\n")
    (root / "README.md").write_text(
        "# FlashDec\n\n## License\n\n"
        f"FlashDec is available under the [{link_label}](LICENSE).\n"
    )
    return root


def _configure_mit_public_release(root):
    return _configure_public_release(root, "MIT", MIT_LICENSE_TEXT)


def test_release_version_readers_support_pyproject_and_package(tmp_path):
    root = _release_tree(tmp_path)
    assert _read_project_version(root / "pyproject.toml") == "0.1.0"
    assert _read_package_version(root / "flashdec/__init__.py") == "0.1.0"
    assert _read_constraint_pins(root / "constraints/flashinfer-cu128.txt") == (
        FLASHINFER_CONSTRAINT_PINS
    )


def test_release_model_latency_protocol_requires_explicit_jit_prime():
    assert MODEL_LATENCY_WORKER_SCHEMA_VERSION == 2
    assert MODEL_LATENCY_CSV_SCHEMA_VERSION == 4
    assert FORMAL_PRIME_ITERS == 1
    assert GUARDRAIL_CASE == "qwen_b8_i512_o2"
    assert FORMAL_CASE_SHAPES[GUARDRAIL_CASE] == (8, 512, 2)
    assert GUARDRAIL_RATIO_LIMIT == 1.05
    assert TARGET_CASE == "qwen_b8_i8192_o4096"
    assert FORMAL_CASE_SHAPES[TARGET_CASE] == (8, 8192, 4096)
    assert TARGET_RATIO_LIMIT == 0.970
    assert (
        MODEL_LATENCY_TIMING_SCOPE
        == MODEL_LATENCY_WORKER_TIMING_SCOPE
        == MODEL_LATENCY_SUMMARY_TIMING_SCOPE
    )
    assert "full-length JIT-prime and warmup calls" in MODEL_LATENCY_TIMING_SCOPE


def test_project_license_reader_requires_pep639_spdx_string(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nlicense = "Apache-2.0"\n')
    assert _read_project_license(pyproject) == "Apache-2.0"

    pyproject.write_text('[project]\nlicense = { file = "LICENSE" }\n')
    with pytest.raises(ValueError, match="legacy license tables"):
        _read_project_license(pyproject)


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


def test_validate_release_tree_rejects_governance_drift(tmp_path):
    root = _release_tree(tmp_path)
    (root / "SECURITY.md").write_text("# Security Policy\n")
    (root / "CITATION.cff").write_text(
        (root / "CITATION.cff").read_text().replace(
            "version: 0.1.0",
            "version: 9.9.9",
        )
    )

    problems = validate_release_tree(root)

    assert any("Reporting a vulnerability" in item for item in problems)
    assert "version mismatch: CITATION.cff=9.9.9, project=0.1.0" in problems


def test_public_license_gate_is_explicit_and_passes_when_metadata_align(tmp_path):
    root = _release_tree(tmp_path)

    assert validate_release_tree(root) == []
    pending = validate_release_tree(root, require_public=True)
    assert "missing public-release artifact: LICENSE" in pending
    assert "pyproject.toml does not contain [project].license" in pending
    assert (
        "pyproject.toml [project].license-files must be exactly ['LICENSE']"
        in pending
    )
    assert (
        "pyproject.toml [build-system].requires must include setuptools>=77"
        in pending
    )
    assert "CITATION.cff does not define license" in pending
    assert "README.md must contain an explicit ## License section" in pending

    _configure_mit_public_release(root)

    assert _license_id_from_text(MIT_LICENSE_TEXT) == "MIT"
    assert public_release_problems(root) == []
    assert validate_release_tree(root, require_public=True) == []


def test_complete_standard_apache_public_gate_passes(tmp_path):
    root = _configure_public_release(
        _release_tree(tmp_path),
        "Apache-2.0",
        APACHE_LICENSE_TEXT,
    )

    assert _license_id_from_text(APACHE_LICENSE_TEXT) == "Apache-2.0"
    assert _license_id_from_text("\n".join(APACHE_LICENSE_TEXT.split())) == "Apache-2.0"
    assert public_release_problems(root) == []
    assert validate_release_tree(root, require_public=True) == []


@pytest.mark.parametrize(
    ("license_id", "standard_text"),
    (
        ("MIT", MIT_LICENSE_TEXT),
        ("Apache-2.0", APACHE_LICENSE_TEXT),
    ),
)
def test_public_license_gate_rejects_restriction_after_standard_text(
    tmp_path,
    license_id,
    standard_text,
):
    root = _configure_public_release(
        _release_tree(tmp_path),
        license_id,
        standard_text + "\nAdditional restriction: redistribution is prohibited.\n",
    )

    problems = public_release_problems(root)

    assert _license_id_from_text((root / "LICENSE").read_text()) is None
    assert (
        "LICENSE must contain an unmodified standard MIT or Apache-2.0 text"
        in problems
    )


@pytest.mark.parametrize(
    "copyright_line",
    (
        "Copyright (c) 2026 FlashDec contributors",
        "Copyright © 2024-2026 FlashDec contributors",
        "Copyright 2026 FlashDec contributors <maintainers@example.com>",
    ),
)
def test_mit_recognizer_allows_reasonable_copyright_lines(copyright_line):
    varied = MIT_LICENSE_TEXT.replace(
        "Copyright (c) 2026 FlashDec contributors",
        copyright_line,
    )
    assert _license_id_from_text(varied) == "MIT"


@pytest.mark.parametrize(
    "truncated_text",
    (
        "MIT License\nPermission is hereby granted.\n",
        "Apache License\nVersion 2.0, January 2004\n",
        "MIT License\nPermission is hereby granted.\n" + "padding " * 200,
        "Apache License\nVersion 2.0, January 2004\n" + "padding " * 1500,
    ),
)
def test_public_license_gate_rejects_truncated_license_text(
    tmp_path,
    truncated_text,
):
    root = _release_tree(tmp_path)
    _configure_mit_public_release(root)
    (root / "LICENSE").write_text(truncated_text)

    problems = public_release_problems(root)

    assert (
        "LICENSE must contain an unmodified standard MIT or Apache-2.0 text"
        in problems
    )
    assert _license_id_from_text(truncated_text) is None


def test_public_license_gate_requires_modern_setuptools_backend(tmp_path):
    root = _configure_mit_public_release(_release_tree(tmp_path))
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        .replace("setuptools>=77", "setuptools>=76")
        .replace("setuptools.build_meta", "hatchling.build")
    )

    problems = public_release_problems(root)

    assert (
        "pyproject.toml [build-system].requires must include setuptools>=77"
        in problems
    )
    assert (
        "pyproject.toml [build-system].build-backend must be "
        "'setuptools.build_meta'"
        in problems
    )


def test_public_license_gate_rejects_legacy_metadata_and_license_files(tmp_path):
    root = _configure_mit_public_release(_release_tree(tmp_path))
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        .replace('license = "MIT"', 'license = { file = "LICENSE" }')
        .replace(
            'license-files = ["LICENSE"]',
            'license-files = ["LICENSE", "NOTICE"]',
        )
    )

    problems = public_release_problems(root)

    assert any("legacy license tables are not accepted" in item for item in problems)
    assert (
        "pyproject.toml [project].license-files must be exactly ['LICENSE']"
        in problems
    )


def test_public_license_gate_rejects_unsupported_spdx_expression(tmp_path):
    root = _configure_mit_public_release(_release_tree(tmp_path))
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('license = "MIT"', 'license = "BSD-3-Clause"')
    )

    problems = public_release_problems(root)

    assert (
        "pyproject.toml [project].license must be exactly 'MIT' or 'Apache-2.0'"
        in problems
    )


def test_public_license_gate_rejects_metadata_cff_and_readme_mismatch(tmp_path):
    root = _configure_mit_public_release(_release_tree(tmp_path))
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('license = "MIT"', 'license = "Apache-2.0"')
    )
    citation = root / "CITATION.cff"
    citation.write_text(citation.read_text().replace("license: MIT", "license: Apache-2.0"))
    readme = root / "README.md"
    readme.write_text(
        readme.read_text().replace(
            "[MIT License](LICENSE)",
            "[Apache License 2.0](LICENSE)",
        )
    )

    problems = public_release_problems(root)

    assert "license mismatch: pyproject='Apache-2.0', LICENSE='MIT'" in problems
    assert "license mismatch: CITATION.cff='Apache-2.0', LICENSE='MIT'" in problems
    assert "license mismatch: README.md='Apache-2.0', LICENSE='MIT'" in problems


def test_validate_release_tree_rejects_flashinfer_constraint_drift(tmp_path):
    root = _release_tree(tmp_path)
    constraints = root / "constraints/flashinfer-cu128.txt"
    constraints.write_text(constraints.read_text().replace("3.6.0", "3.7.1"))

    problems = validate_release_tree(root)
    assert (
        "FlashInfer constraint mismatch: triton='3.7.1', expected '3.6.0'" in problems
    )


def test_flashinfer_constraints_match_runner_environment_contract():
    assert FLASHINFER_CONSTRAINT_PINS == {
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
        f"missing release evidence: {relative}"
        for relative in RELEASE_EVIDENCE_PATHS
        if relative not in REQUIRED_PATHS
    ]

    for relative in RELEASE_EVIDENCE_PATHS:
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == WARP_SELECTION_EVIDENCE_PATH:
            source = (
                Path(__file__).resolve().parents[1]
                / WARP_SELECTION_EVIDENCE_PATH
            )
            path.write_text(source.read_text())
        else:
            path.write_text("verified evidence\n")
    assert validate_release_tree(root, require_evidence=True) == []


def test_release_evidence_includes_scheduler_multi_layer_and_shared_prefix_summaries():
    assert (
        "benchmarks/results/scheduler_capacity_progress_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/multi_layer_transaction_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/shared_prefix_capacity_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/shared_prefix_pre_metadata_cache_summary.md"
        not in RELEASE_EVIDENCE_PATHS
    )


def test_release_evidence_covers_canonical_kernel_and_engine_results():
    for relative in (
        "benchmarks/results/paged_decode_warp_selection_summary.md",
        "benchmarks/results/paged_decode_block_size_summary.md",
        "benchmarks/results/paged_decode_kv_layout_summary.md",
        "benchmarks/results/paged_decode_default_profile_summary.md",
        "benchmarks/results/paged_decode_staging_summary.md",
        "benchmarks/results/rope_kv_append_backends_summary.md",
        "benchmarks/results/decode_engine_workload_trials3_summary.md",
        "benchmarks/results/decode_engine_stage_profile_summary.md",
    ):
        assert relative in RELEASE_EVIDENCE_PATHS


def test_warp_selection_summary_preserves_historical_provenance_contract():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "benchmarks/results/paged_decode_warp_selection_summary.md"
    ).read_text()
    for required in WARP_SELECTION_EVIDENCE_MARKERS:
        assert required in text


def test_release_gate_rejects_incomplete_warp_selection_evidence(tmp_path):
    root = _release_tree(tmp_path)
    for relative in RELEASE_EVIDENCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("verified evidence\n")
    problems = validate_release_tree(root, require_evidence=True)
    assert any(
        problem.startswith("warp selection evidence missing marker:")
        for problem in problems
    )


def test_release_gate_rejects_warp_selection_digest_only_drift(tmp_path):
    root = _release_tree(tmp_path)
    canonical = (
        Path(__file__).resolve().parents[1] / WARP_SELECTION_EVIDENCE_PATH
    ).read_text()
    for relative in RELEASE_EVIDENCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            canonical + "\n" if relative == WARP_SELECTION_EVIDENCE_PATH
            else "verified evidence\n"
        )
    problems = validate_release_tree(root, require_evidence=True)
    assert not any("missing marker" in problem for problem in problems)
    assert any(
        problem.startswith("warp selection evidence content digest mismatch:")
        for problem in problems
    )


def test_release_tree_requires_public_documentation_and_scheduler_surface():
    for relative in (
        "docs/INDEX.md",
        "docs/compatibility.md",
        "docs/concepts/online_softmax.md",
        "docs/design.md",
        "docs/design_decode_engine.md",
        "docs/performance_report.md",
        "docs/references.md",
        "docs/research_questions.md",
        "docs/reproducibility.md",
        "benchmarks/README.md",
        "benchmarks/results/README.md",
        "scripts/README.md",
        "flashdec/scheduled_workload.py",
        "tests/test_scheduled_workload.py",
        "tests/test_scheduled_workload_config.py",
        "tests/test_scheduler_workload_benchmark.py",
    ):
        assert relative in REQUIRED_PATHS


def test_release_tree_requires_github_collaboration_surface():
    for relative in (
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "SUPPORT.md",
        "CITATION.cff",
        ".github/CODEOWNERS",
        ".github/dependabot.yml",
        ".github/workflows/quality.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/change_proposal.yml",
        ".github/ISSUE_TEMPLATE/question.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/pull_request_template.md",
    ):
        assert relative in REQUIRED_PATHS
    assert PUBLIC_RELEASE_REQUIRED_PATHS == ("LICENSE",)


def test_public_entry_files_exist_and_are_release_tracked():
    root = Path(__file__).resolve().parents[1]
    root_governance_files = {
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "SUPPORT.md",
    }

    for relative in PUBLIC_ENTRY_FILES:
        assert (root / relative).is_file(), relative
        assert (
            relative in REQUIRED_PATHS or relative in root_governance_files
        ), relative


def test_release_tree_requires_trusted_transaction_validation_surface():
    for relative in (
        "benchmarks/run_fused_transaction_fast_path.py",
        "benchmarks/summarize_fused_transaction_fast_path.py",
        "tests/test_fused_transaction_fast_path_benchmark.py",
        "tests/test_fused_transaction_fast_path_summary.py",
    ):
        assert relative in REQUIRED_PATHS
    assert (
        "benchmarks/results/trusted_transaction_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )
    assert (
        "benchmarks/results/persistent_metadata_candidate_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )


def test_release_tree_requires_integrated_runtime_evidence():
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
        "benchmarks/results/integrated_runtime_lifecycle_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )


def test_release_tree_requires_flashinfer_baseline_evidence():
    for relative in (
        "constraints/flashinfer-cu128.txt",
        "docs/design_flashinfer_baseline.md",
        "docs/research_questions.md",
        "benchmarks/run_flashinfer_baseline.py",
        "benchmarks/summarize_flashinfer_baseline.py",
        "tests/test_flashinfer_baseline.py",
        "tests/test_flashinfer_baseline_benchmark.py",
        "tests/test_flashinfer_baseline_summary.py",
    ):
        assert relative in REQUIRED_PATHS
    assert (
        "benchmarks/results/flashinfer_paged_decode_baseline_summary.md"
        in RELEASE_EVIDENCE_PATHS
    )


def test_release_tree_requires_public_results_chart_surface():
    for relative in (
        "benchmarks/results/public_results_snapshot.json",
        "docs/assets/flashdec-results-overview-dark.svg",
        "docs/assets/flashdec-results-overview-light.svg",
        "scripts/generate_public_results_chart.py",
        "tests/test_public_results_chart.py",
    ):
        assert relative in REQUIRED_PATHS
