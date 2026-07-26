"""Validate repository-local Markdown links without external dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_RESOURCE = re.compile(
    r"\b(?:src|srcset)\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "#")
DEFAULT_ROOT_FILES = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
)
DEFAULT_DIRECTORIES = ("docs", "benchmarks", "scripts", ".github")
IGNORED_RESULT_DIRECTORIES = ("local_backups",)
IGNORED_RESULT_SUFFIXES = ("_quick_summary.md", "_smoke.md")
DISALLOWED_PORTFOLIO_TERMS = (
    "面试",
    "求职",
    "简历",
    "interview",
    "resume",
)

# These files describe the repository's current public-facing state. Historical
# weekly notes and benchmark summaries are deliberately excluded: preserving an
# old state description there is useful evidence, not stale front-page policy.
PUBLIC_ENTRY_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/change_proposal.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    ".github/pull_request_template.md",
    "docs/API.md",
    "docs/AI_INFRA_SCOPE.md",
    "docs/DELIVERY_STATUS.md",
    "docs/INDEX.md",
    "docs/NEXT_STEPS.md",
    "docs/PROJECT_PLAN.md",
    "docs/PUBLIC_RELEASE_CHECKLIST.md",
    "docs/ROADMAP.md",
    "docs/compatibility.md",
    "docs/design.md",
    "docs/performance_report.md",
    "docs/reproducibility.md",
)

PERSONAL_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/(?P<account>[^/\s`\"']+)/"),
    re.compile(r"[A-Za-z]:\\Users\\(?P<account>[^\\\s`\"']+)\\"),
)
LAN_IP = re.compile(
    r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d|\.\d)"
)
PLACEHOLDER_ACCOUNTS = {
    "$USER",
    "${USER}",
    "<user>",
    "<username>",
}
STALE_CURRENT_STATE_PATTERNS = (
    re.compile(r"\bprivate(?:-only)? maintenance\b", re.IGNORECASE),
    re.compile(r"\bprivate\s+`?0\.0\.0`?\s+development candidate\b", re.IGNORECASE),
    re.compile(
        r"\bpublic (?:repository |source )?release (?:is|remains) paused\b",
        re.IGNORECASE,
    ),
    re.compile(r"仓库当前为\s*private", re.IGNORECASE),
    re.compile(
        r"仓库[^。\n]{0,80}(?:仍为|仍是|目前为|当前为)\s*private",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brepository\b[^.\n]{0,80}\b(?:is|remains|still)\s+private\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:可见性|visibility)[^。\n]{0,60}(?:切换|change)[^。\n]{0,60}"
        r"(?:前|before)[^。\n]{0,30}\bprivate\b",
        re.IGNORECASE,
    ),
    re.compile(r"公开设置和\s*tag\s*暂停", re.IGNORECASE),
    re.compile(r"公开(?:和|与)\s*(?:release\s*)?tag\s*按所有者要求暂停", re.IGNORECASE),
)


def markdown_files(root: Path):
    """Return the Markdown files included in the canonical repository docs."""
    files = [root / name for name in DEFAULT_ROOT_FILES if (root / name).is_file()]
    for directory in DEFAULT_DIRECTORIES:
        path = root / directory
        if path.is_dir():
            files.extend(path.rglob("*.md"))
    public_files = []
    result_root = root / "benchmarks" / "results"
    for path in set(files):
        try:
            result_relative = path.relative_to(result_root)
        except ValueError:
            public_files.append(path)
            continue
        if (
            result_relative.parts
            and result_relative.parts[0] in IGNORED_RESULT_DIRECTORIES
        ):
            continue
        if len(result_relative.parts) == 1 and path.name.endswith(
            IGNORED_RESULT_SUFFIXES
        ):
            continue
        public_files.append(path)
    return sorted(public_files)


def document_targets(line: str):
    """Yield Markdown links and HTML image resources from one source line."""
    yield from MARKDOWN_LINK.findall(line)
    for raw_srcset in HTML_RESOURCE.findall(line):
        for candidate in raw_srcset.split(","):
            target = candidate.strip().split(maxsplit=1)[0]
            if target:
                yield target


def local_link_problems(root: Path):
    """Return missing repository-local Markdown link targets."""
    root = root.resolve()
    problems = []
    for document in markdown_files(root):
        text = document.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for raw_target in document_targets(line):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(SKIP_PREFIXES):
                    continue
                path_text = unquote(target.split("#", 1)[0])
                if not path_text:
                    continue
                candidate = (document.parent / path_text).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    problems.append(
                        f"{document.relative_to(root)}:{line_number}: "
                        f"link escapes repository: {target}"
                    )
                    continue
                if not candidate.exists():
                    problems.append(
                        f"{document.relative_to(root)}:{line_number}: "
                        f"missing link target: {target}"
                    )
    return problems


def portfolio_wording_problems(root: Path):
    """Return utility-oriented wording that does not belong in project docs."""
    root = root.resolve()
    problems = []
    for document in markdown_files(root):
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            lowered = line.casefold()
            for term in DISALLOWED_PORTFOLIO_TERMS:
                if term.casefold() in lowered:
                    problems.append(
                        f"{document.relative_to(root)}:{line_number}: "
                        f"disallowed portfolio wording: {term}"
                    )
    return problems


def public_safety_problems(root: Path):
    """Return machine-specific data and stale state in current entry points.

    This intentionally checks a small allowlist instead of every document. Terms
    such as ``request-private``, ``private tail``, and historical private-state
    notes are valid technical or provenance language and must remain untouched.
    """
    root = root.resolve()
    problems = []
    for relative in PUBLIC_ENTRY_FILES:
        document = root / relative
        if not document.is_file():
            continue
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            for pattern in PERSONAL_PATH_PATTERNS:
                for match in pattern.finditer(line):
                    if match.group("account") in PLACEHOLDER_ACCOUNTS:
                        continue
                    problems.append(
                        f"{relative}:{line_number}: personal absolute path in "
                        "public entry point"
                    )
            if LAN_IP.search(line):
                problems.append(
                    f"{relative}:{line_number}: private-network IP in public entry point"
                )
            for pattern in STALE_CURRENT_STATE_PATTERNS:
                if pattern.search(line):
                    problems.append(
                        f"{relative}:{line_number}: stale private-repository status in "
                        "public entry point"
                    )
                    break
    return problems


def documentation_problems(root: Path):
    """Return all repository documentation quality problems."""
    return (
        local_link_problems(root)
        + portfolio_wording_problems(root)
        + public_safety_problems(root)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    problems = documentation_problems(root)
    if problems:
        print("Documentation check: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"Documentation check: PASS ({len(markdown_files(root))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
