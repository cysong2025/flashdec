"""Dependency-free tests for repository Markdown link validation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_docs import (
    documentation_problems,
    local_link_problems,
    markdown_files,
    portfolio_wording_problems,
)


class DocumentationCheckTests(unittest.TestCase):
    def test_discovers_public_root_documents(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md"):
                root.joinpath(name).write_text(f"# {name}\n")

            self.assertEqual(
                {path.name for path in markdown_files(root)},
                {"README.md", "CHANGELOG.md", "CONTRIBUTING.md"},
            )

    def test_accepts_existing_relative_and_external_links(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (root / "README.md").write_text(
                "[design](docs/design.md) [web](https://example.com)\n"
            )
            (docs / "design.md").write_text("# Design\n")

            self.assertEqual(local_link_problems(root), [])
            self.assertEqual(len(markdown_files(root)), 2)

    def test_reports_missing_and_escaping_targets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text(
                "[missing](docs/missing.md) [escape](../private.md)\n"
            )

            problems = local_link_problems(root)

            self.assertEqual(len(problems), 2)
            self.assertIn("missing link target", problems[0])
            self.assertIn("link escapes repository", problems[1])

    def test_rejects_utility_oriented_portfolio_wording(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text("面试准备 interview notes\n")

            problems = portfolio_wording_problems(root)

            self.assertEqual(len(problems), 2)
            self.assertTrue(all("disallowed portfolio wording" in item for item in problems))
            self.assertEqual(documentation_problems(root), problems)


if __name__ == "__main__":
    unittest.main()
