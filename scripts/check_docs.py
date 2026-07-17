"""Validate repository-local Markdown links without external dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "#")
DEFAULT_ROOT_FILES = ("README.md", "CHANGELOG.md", "CONTRIBUTING.md")
DEFAULT_DIRECTORIES = ("docs", "benchmarks", "scripts")


def markdown_files(root: Path):
    """Return the Markdown files included in the public documentation tree."""
    files = [root / name for name in DEFAULT_ROOT_FILES if (root / name).is_file()]
    for directory in DEFAULT_DIRECTORIES:
        path = root / directory
        if path.is_dir():
            files.extend(path.rglob("*.md"))
    return sorted(set(files))


def local_link_problems(root: Path):
    """Return missing repository-local Markdown link targets."""
    root = root.resolve()
    problems = []
    for document in markdown_files(root):
        text = document.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for raw_target in MARKDOWN_LINK.findall(line):
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
    problems = local_link_problems(root)
    if problems:
        print("Markdown link check: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"Markdown link check: PASS ({len(markdown_files(root))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
