"""Validate FlashDec release artifacts, version consistency, and Git gates."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import re
import subprocess


REQUIRED_PATHS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    "CITATION.cff",
    ".gitattributes",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/workflows/quality.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/change_proposal.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/pull_request_template.md",
    "pyproject.toml",
    "constraints/flashinfer-cu128.txt",
    "flashdec/__init__.py",
    "flashdec/cache.py",
    "flashdec/engine.py",
    "flashdec/integrated_workload.py",
    "flashdec/scheduled_workload.py",
    "flashdec/scheduler.py",
    "flashdec/vllm_backend.py",
    "flashdec/vllm_attestation.py",
    "flashdec/vllm_plugin.py",
    "flashdec/workload.py",
    "docs/API.md",
    "docs/INDEX.md",
    "docs/compatibility.md",
    "docs/concepts/online_softmax.md",
    "docs/design.md",
    "docs/design_paged_kv.md",
    "docs/design_rope_kv_append.md",
    "docs/design_cuda_kv_append.md",
    "docs/design_fused_rope_kv_append.md",
    "docs/design_dynamic_workload.md",
    "docs/design_multi_layer_kv_transaction.md",
    "docs/design_shared_prefix_blocks.md",
    "docs/design_integrated_scheduled_multi_layer.md",
    "docs/design_flashinfer_baseline.md",
    "docs/design_vllm_backend.md",
    "docs/design_decode_engine.md",
    "docs/assets/flashdec-architecture-dark.svg",
    "docs/assets/flashdec-architecture-light.svg",
    "docs/assets/flashdec-results-overview-dark.svg",
    "docs/assets/flashdec-results-overview-light.svg",
    "docs/design_scheduler.md",
    "docs/kernel_experiments.md",
    "docs/performance_report.md",
    "docs/references.md",
    "docs/research_questions.md",
    "docs/reproducibility.md",
    "benchmarks/run_decode_engine_workload.py",
    "benchmarks/README.md",
    "benchmarks/results/README.md",
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
    "benchmarks/run_integrated_scheduled_multi_layer.py",
    "benchmarks/summarize_integrated_scheduled_multi_layer.py",
    "benchmarks/run_flashinfer_baseline.py",
    "benchmarks/summarize_flashinfer_baseline.py",
    "benchmarks/run_vllm_attention_microbench.py",
    "benchmarks/summarize_vllm_attention_microbench.py",
    "benchmarks/run_vllm_model_correctness.py",
    "benchmarks/summarize_vllm_model_correctness.py",
    "benchmarks/run_vllm_model_latency.py",
    "benchmarks/run_vllm_model_latency_worker.py",
    "benchmarks/summarize_vllm_model_latency.py",
    "benchmarks/run_vllm_serving_benchmark.py",
    "benchmarks/summarize_vllm_serving_benchmark.py",
    "benchmarks/results/public_results_snapshot.json",
    "scripts/check_env.py",
    "scripts/check_docs.py",
    "scripts/check_release.py",
    "scripts/generate_public_results_chart.py",
    "scripts/README.md",
    "scripts/run_validation.py",
    "tests/test_paged_cache.py",
    "tests/test_docs_check.py",
    "tests/test_decode_engine.py",
    "tests/test_workload.py",
    "tests/test_scheduler.py",
    "tests/test_scheduled_workload.py",
    "tests/test_scheduled_workload_config.py",
    "tests/test_scheduler_workload_benchmark.py",
    "tests/test_validation_orchestrator.py",
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
    "tests/test_integrated_workload.py",
    "tests/test_integrated_workload_config.py",
    "tests/test_integrated_workload_benchmark.py",
    "tests/test_integrated_workload_summary.py",
    "tests/test_flashinfer_baseline_benchmark.py",
    "tests/test_flashinfer_baseline.py",
    "tests/test_flashinfer_baseline_summary.py",
    "tests/test_vllm_plugin.py",
    "tests/test_vllm_backend.py",
    "tests/test_vllm_attention_microbench_summary.py",
    "tests/test_vllm_model_correctness_summary.py",
    "tests/test_vllm_model_latency_summary.py",
    "tests/test_vllm_model_latency_runner.py",
    "tests/test_vllm_serving_benchmark_summary.py",
    "tests/test_public_results_chart.py",
)

# A source repository can be prepared and reviewed before the owner makes the
# legal license choice. Enable this stricter gate only for the final visibility
# change; the ordinary development/research-preview gate stays usable meanwhile.
PUBLIC_RELEASE_REQUIRED_PATHS = ("LICENSE",)

SUPPORTED_LICENSES = ("MIT", "Apache-2.0")

# SHA-256 is calculated after collapsing all whitespace. The MIT digest covers
# the invariant body beginning with "Permission is hereby granted"; copyright
# lines are validated separately so the year and holder can legitimately vary.
# The Apache digest covers the complete canonical 2.0 text, including its
# application appendix. Exact normalized matching rejects appended restrictions.
MIT_BODY_SHA256 = "fe2a9817987f862eaced948f0468c7f51d2fedfc48c5c505b246a49a3870e9a5"
APACHE_2_0_SHA256 = "0ffddef9e48f8a09aed5caf2d44f7ba1c1be2d9b8e0a6f693b1635b2d5566645"
MIT_COPYRIGHT = re.compile(
    r"Copyright(?:\s+\([cC]\)|\s+©)?\s+"
    r"(?:\d{4}(?:-\d{4})?|\[year\])\s+\S.{0,199}"
)
README_LICENSE_LINKS = {
    "MIT": re.compile(r"\[MIT License\]\((?:\./)?LICENSE\)", re.IGNORECASE),
    "Apache-2.0": re.compile(
        r"\[Apache License 2\.0\]\((?:\./)?LICENSE\)",
        re.IGNORECASE,
    ),
}

GOVERNANCE_MARKERS = {
    "SECURITY.md": (
        "# Security Policy",
        "## Reporting a vulnerability",
        "private vulnerability reporting",
    ),
    "CODE_OF_CONDUCT.md": (
        "# FlashDec Code of Conduct",
        "## Expected behavior",
        "## Reporting and enforcement",
    ),
    "SUPPORT.md": (
        "# Support",
        "SECURITY.md",
        "issue form",
    ),
    ".github/dependabot.yml": (
        "version: 2",
        "package-ecosystem:",
        "github-actions",
        "pip",
    ),
    ".github/ISSUE_TEMPLATE/question.yml": (
        "Usage / environment question",
        "body:",
        "Remove credentials, private paths, and unrelated logs.",
    ),
}

RELEASE_EVIDENCE_PATHS = (
    "benchmarks/results/paged_decode_warp_selection_summary.md",
    "benchmarks/results/paged_decode_block_size_summary.md",
    "benchmarks/results/paged_decode_kv_layout_summary.md",
    "benchmarks/results/paged_decode_default_profile_summary.md",
    "benchmarks/results/paged_decode_staging_summary.md",
    "benchmarks/results/rope_kv_append_backends_summary.md",
    "benchmarks/results/decode_engine_workload_trials3_summary.md",
    "benchmarks/results/decode_engine_stage_profile_summary.md",
    "benchmarks/results/scheduler_capacity_progress_summary.md",
    "benchmarks/results/multi_layer_transaction_summary.md",
    "benchmarks/results/shared_prefix_capacity_summary.md",
    "benchmarks/results/trusted_transaction_summary.md",
    "benchmarks/results/persistent_metadata_candidate_summary.md",
    "benchmarks/results/integrated_runtime_lifecycle_summary.md",
    "benchmarks/results/flashinfer_paged_decode_baseline_summary.md",
    "benchmarks/results/vllm_qwen_attention_summary.md",
    "benchmarks/results/vllm_qwen_model_correctness_summary.md",
    "benchmarks/results/vllm_qwen_model_latency_summary.md",
    "benchmarks/results/vllm_qwen_serving_summary.md",
    "docs/performance_report.md",
)

WARP_SELECTION_EVIDENCE_PATH = (
    "benchmarks/results/paged_decode_warp_selection_summary.md"
)
WARP_SELECTION_EVIDENCE_SHA256 = (
    "58bae4f421c127b3c994eda8871efa418fae8b245b5d144955a48197d98a585d"
)
WARP_SELECTION_EVIDENCE_MARKERS = (
    "# Paged Decode Warp Selection Summary",
    "Recorded experiment commit: `aa81af8`",
    "2.11.0+cu128 / 3.6.0 / 12.8",
    "--block-size 16",
    "seed 87, warmup 5, repeat 30",
    "84 Triton rows across 28 dtype/case groups",
    "`num_warps=2` p50 wins | 28",
    "best effective total GB/s at p50",
    "raw CSV was intentionally not tracked",
)

FLASHINFER_CONSTRAINT_PINS = {
    "torch": "2.11.0+cu128",
    "triton": "3.6.0",
    "flashinfer-python": "0.6.15.post1",
    "cuda-toolkit": "12.8.1",
    "cuda-python": "12.9.1",
    "cuda-bindings": "12.9.7",
    "cuda-pathfinder": "1.6.0",
    "ninja": "1.13.0",
}


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


def _read_constraint_pins(path):
    pins = {}
    lines = Path(path).read_text().splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(
                f"FlashInfer constraints line {line_number} must be an exact == pin"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise ValueError(
                f"FlashInfer constraints line {line_number} must include name and version"
            )
        if name in pins:
            raise ValueError(f"duplicate FlashInfer constraint: {name}")
        pins[name] = version
    return pins


def _read_top_level_yaml_scalar(path, key):
    """Read one simple top-level scalar from CFF/YAML without PyYAML."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.*?)\s*$", text)
    if match is None or not match.group(1):
        raise ValueError(f"{Path(path).name} does not define {key}")
    raw_value = match.group(1)
    if raw_value[:1] in {"'", '"'}:
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{Path(path).name} has an invalid quoted {key}"
            ) from exc
        if not isinstance(value, str):
            raise ValueError(f"{Path(path).name} {key} must be a string")
        return value
    return raw_value


def _toml_section(text, name):
    match = re.search(
        rf"(?ms)^\[{re.escape(name)}\]\s*$.*?(?=^\[|\Z)",
        text,
    )
    return match.group(0) if match is not None else None


def _fallback_toml_string(section, key):
    if section is None:
        return None
    match = re.search(
        rf"(?m)^{re.escape(key)}\s*=\s*['\"]([^'\"]+)['\"]\s*$",
        section,
    )
    return match.group(1) if match is not None else None


def _fallback_toml_string_list(section, key):
    if section is None:
        return None
    match = re.search(
        rf"(?ms)^{re.escape(key)}\s*=\s*(\[.*?\])\s*$",
        section,
    )
    if match is None:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"pyproject.toml has invalid {key} metadata") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"pyproject.toml {key} must be a string list")
    return value


def _read_publication_metadata(path):
    """Read the PEP 639 and build-backend fields used by the public gate."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 fallback
        project = _toml_section(text, "project")
        build_system = _toml_section(text, "build-system")
        if project is None:
            raise ValueError("pyproject.toml does not contain [project]")
        license_value = _fallback_toml_string(project, "license")
        if license_value is None and re.search(
            r"(?m)^license\s*=",
            project,
        ):
            license_value = {"legacy-table": True}
        return {
            "license": license_value,
            "license_files": _fallback_toml_string_list(
                project,
                "license-files",
            ),
            "build_requires": _fallback_toml_string_list(
                build_system,
                "requires",
            ),
            "build_backend": _fallback_toml_string(
                build_system,
                "build-backend",
            ),
        }

    data = tomllib.loads(text)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml does not contain [project]")
    build_system = data.get("build-system")
    if not isinstance(build_system, dict):
        build_system = {}
    return {
        "license": project.get("license"),
        "license_files": project.get("license-files"),
        "build_requires": build_system.get("requires"),
        "build_backend": build_system.get("build-backend"),
    }


def _read_project_license(path):
    """Return the modern PEP 639 SPDX string or raise for legacy metadata."""
    license_value = _read_publication_metadata(path)["license"]
    if license_value is None:
        raise ValueError("pyproject.toml does not contain [project].license")
    if not isinstance(license_value, str) or not license_value.strip():
        raise ValueError(
            "pyproject.toml [project].license must be a PEP 639 SPDX string; "
            "legacy license tables are not accepted"
        )
    return license_value.strip()


def _setuptools_requirement_is_modern(requirement):
    compact = re.sub(r"\s+", "", requirement)
    if re.match(r"^setuptools(?=[<>=!~;,\[]|$)", compact, re.IGNORECASE) is None:
        return False
    match = re.search(r"(?:^|,)>=([0-9]+(?:\.[0-9]+)*)", compact[len("setuptools"):])
    if match is None:
        return False
    version = tuple(int(part) for part in match.group(1).split("."))
    return version >= (77,)


def _license_id_from_text(text):
    """Recognize a normalized standard text without added restrictions."""
    text = text.lstrip("\ufeff")
    normalized = " ".join(text.split())
    apache_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if apache_digest == APACHE_2_0_SHA256:
        return "Apache-2.0"

    lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in text.splitlines()
        if line.strip()
    ]
    if not lines or lines[0] != "MIT License":
        return None
    try:
        body_start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("Permission is hereby granted")
        )
    except StopIteration:
        return None
    copyright_lines = lines[1:body_start]
    if not copyright_lines or not all(
        MIT_COPYRIGHT.fullmatch(line) for line in copyright_lines
    ):
        return None
    normalized_body = " ".join(lines[body_start:])
    mit_digest = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
    if mit_digest == MIT_BODY_SHA256:
        return "MIT"
    return None


def _readme_license_id(path):
    """Return the supported license named and linked in README's License section."""
    text = Path(path).read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^## License\s*$\s*(.*?)(?=^##\s|\Z)",
        text,
    )
    if section is None:
        raise ValueError("README.md must contain an explicit ## License section")
    matches = [
        license_id
        for license_id, pattern in README_LICENSE_LINKS.items()
        if pattern.search(section.group(1))
    ]
    if len(matches) != 1:
        raise ValueError(
            "README.md License section must contain exactly one supported link: "
            "[MIT License](LICENSE) or [Apache License 2.0](LICENSE)"
        )
    return matches[0]


def governance_problems(root, project_version=None):
    """Return missing governance content or inconsistent citation metadata."""
    root = Path(root)
    problems = []
    for relative, markers in GOVERNANCE_MARKERS.items():
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                problems.append(f"{relative} must contain {marker!r}")

    codeowners = root / ".github/CODEOWNERS"
    if codeowners.is_file():
        ownership_rules = [
            line.strip()
            for line in codeowners.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not any(
            re.match(r"^\*\s+@[A-Za-z0-9-]+(?:\s|$)", line)
            for line in ownership_rules
        ):
            problems.append(".github/CODEOWNERS must assign the repository root")

    citation = root / "CITATION.cff"
    if citation.is_file():
        required_scalars = {
            "cff-version": "1.2.0",
            "type": "software",
            "repository-code": "https://github.com/cysong2025/flashdec",
        }
        for key, expected in required_scalars.items():
            try:
                actual = _read_top_level_yaml_scalar(citation, key)
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
            else:
                if actual != expected:
                    problems.append(
                        f"CITATION.cff {key}={actual!r}, expected {expected!r}"
                    )
        if "authors:" not in citation.read_text(encoding="utf-8"):
            problems.append("CITATION.cff must define authors")
        if project_version is not None:
            try:
                citation_version = _read_top_level_yaml_scalar(citation, "version")
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
            else:
                if citation_version != project_version:
                    problems.append(
                        "version mismatch: "
                        f"CITATION.cff={citation_version}, project={project_version}"
                    )
    return problems


def public_release_problems(root):
    """Return final visibility blockers that require an owner license choice."""
    root = Path(root)
    problems = []
    for relative in PUBLIC_RELEASE_REQUIRED_PATHS:
        if not (root / relative).is_file():
            problems.append(f"missing public-release artifact: {relative}")

    license_path = root / "LICENSE"
    detected_license = None
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8")
        if not license_text.strip():
            problems.append("LICENSE must not be empty")
        else:
            detected_license = _license_id_from_text(license_text)
            if detected_license is None:
                problems.append(
                    "LICENSE must contain an unmodified standard MIT or "
                    "Apache-2.0 text"
                )

    pyproject = root / "pyproject.toml"
    project_license = None
    if pyproject.is_file():
        try:
            metadata = _read_publication_metadata(pyproject)
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
        else:
            raw_license = metadata["license"]
            if raw_license is None:
                problems.append("pyproject.toml does not contain [project].license")
            elif not isinstance(raw_license, str) or not raw_license.strip():
                problems.append(
                    "pyproject.toml [project].license must be a PEP 639 SPDX "
                    "string; legacy license tables are not accepted"
                )
            else:
                project_license = raw_license.strip()
                if project_license not in SUPPORTED_LICENSES:
                    problems.append(
                        "pyproject.toml [project].license must be exactly "
                        "'MIT' or 'Apache-2.0'"
                    )

            if metadata["license_files"] != ["LICENSE"]:
                problems.append(
                    "pyproject.toml [project].license-files must be exactly "
                    "['LICENSE']"
                )
            if metadata["build_backend"] != "setuptools.build_meta":
                problems.append(
                    "pyproject.toml [build-system].build-backend must be "
                    "'setuptools.build_meta'"
                )
            build_requires = metadata["build_requires"]
            if not (
                isinstance(build_requires, list)
                and all(isinstance(item, str) for item in build_requires)
                and any(
                    _setuptools_requirement_is_modern(item)
                    for item in build_requires
                )
            ):
                problems.append(
                    "pyproject.toml [build-system].requires must include "
                    "setuptools>=77"
                )

    if project_license is not None and detected_license is not None:
        if project_license != detected_license:
            problems.append(
                "license mismatch: "
                f"pyproject={project_license!r}, LICENSE={detected_license!r}"
            )

    citation = root / "CITATION.cff"
    citation_license = None
    if citation.is_file():
        try:
            citation_license = _read_top_level_yaml_scalar(citation, "license")
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
        else:
            if citation_license not in SUPPORTED_LICENSES:
                problems.append(
                    "CITATION.cff license must be exactly 'MIT' or 'Apache-2.0'"
                )
    if detected_license is not None and citation_license is not None:
        if detected_license != citation_license:
            problems.append(
                "license mismatch: "
                f"CITATION.cff={citation_license!r}, LICENSE={detected_license!r}"
            )

    readme = root / "README.md"
    readme_license = None
    if readme.is_file():
        try:
            readme_license = _readme_license_id(readme)
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
    if detected_license is not None and readme_license is not None:
        if detected_license != readme_license:
            problems.append(
                "license mismatch: "
                f"README.md={readme_license!r}, LICENSE={detected_license!r}"
            )
    return problems


def validate_release_tree(root, require_evidence=False, require_public=False):
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
        warp_summary = root / WARP_SELECTION_EVIDENCE_PATH
        if warp_summary.is_file():
            warp_text = warp_summary.read_text()
            for marker in WARP_SELECTION_EVIDENCE_MARKERS:
                if marker not in warp_text:
                    problems.append(
                        f"warp selection evidence missing marker: {marker}"
                    )
            warp_digest = hashlib.sha256(warp_summary.read_bytes()).hexdigest()
            if warp_digest != WARP_SELECTION_EVIDENCE_SHA256:
                problems.append(
                    "warp selection evidence content digest mismatch: "
                    f"{warp_digest}"
                )

    pyproject = root / "pyproject.toml"
    package_init = root / "flashdec/__init__.py"
    project_version = None
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

    problems.extend(governance_problems(root, project_version=project_version))
    if require_public:
        problems.extend(public_release_problems(root))

    changelog = root / "CHANGELOG.md"
    if changelog.is_file() and "## [Unreleased]" not in changelog.read_text():
        problems.append("CHANGELOG.md must contain an [Unreleased] section")
    reproducibility = root / "docs/reproducibility.md"
    limitations_heading = "## 已知安装与版本限制"
    if reproducibility.is_file() and limitations_heading not in reproducibility.read_text():
        problems.append(
            "docs/reproducibility.md must contain known installation and version limitations"
        )
    constraints = root / "constraints/flashinfer-cu128.txt"
    if constraints.is_file():
        try:
            pins = _read_constraint_pins(constraints)
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
        else:
            for name, expected in FLASHINFER_CONSTRAINT_PINS.items():
                actual = pins.get(name)
                if actual != expected:
                    problems.append(
                        f"FlashInfer constraint mismatch: {name}={actual!r}, expected {expected!r}"
                    )
            unexpected = sorted(set(pins) - set(FLASHINFER_CONSTRAINT_PINS))
            if unexpected:
                problems.append(f"unexpected FlashInfer constraints: {unexpected}")
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
    parser.add_argument(
        "--require-public",
        action="store_true",
        help="require the owner-selected root license and aligned metadata",
    )
    parser.add_argument("--require-tag", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    problems = validate_release_tree(
        root,
        require_evidence=args.require_evidence or args.require_tag,
        require_public=args.require_public,
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
    print(f"Public-release license gate: {'required' if args.require_public else 'not required'}")
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
