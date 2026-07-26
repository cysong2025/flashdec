"""Dependency-free tests for repository Markdown link validation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.check_docs import (
    document_targets,
    documentation_problems,
    local_link_problems,
    markdown_files,
    portfolio_wording_problems,
    public_safety_problems,
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

    def test_discovers_root_governance_documents(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "SECURITY.md",
                "CODE_OF_CONDUCT.md",
                "SUPPORT.md",
            ):
                root.joinpath(name).write_text(f"# {name}\n")

            self.assertEqual(
                {path.name for path in markdown_files(root)},
                {"SECURITY.md", "CODE_OF_CONDUCT.md", "SUPPORT.md"},
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

    def test_checks_html_picture_sources_and_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "docs" / "assets"
            assets.mkdir(parents=True)
            (assets / "light.svg").write_text("<svg/>")
            root.joinpath("README.md").write_text(
                '<picture>\n'
                '  <source srcset="docs/assets/dark.svg 1x, https://example.com/dark.svg 2x">\n'
                '  <img src="docs/assets/light.svg">\n'
                '</picture>\n'
            )

            self.assertEqual(
                list(document_targets('<img src="docs/assets/light.svg">')),
                ["docs/assets/light.svg"],
            )
            problems = local_link_problems(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("docs/assets/dark.svg", problems[0])

    def test_scans_github_markdown_templates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            github = root / ".github"
            github.mkdir()
            template = github / "pull_request_template.md"
            template.write_text("[missing](../docs/missing.md)\n")

            self.assertEqual(markdown_files(root), [template])
            self.assertEqual(len(local_link_problems(root)), 1)

    def test_ignores_local_and_quick_result_markdown(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "benchmarks" / "results"
            backups = results / "local_backups"
            backups.mkdir(parents=True)
            archive = results / "archive"
            archive.mkdir()
            tracked_backup = archive / "local_backups" / "tracked.md"
            tracked_backup.parent.mkdir()
            canonical = results / "r1_trials3_summary.md"
            archived_quick = archive / "r1_quick_summary.md"
            canonical.write_text("# Canonical\n")
            archived_quick.write_text("# Tracked archive\n")
            tracked_backup.write_text("# Tracked nested backup\n")
            (results / "r1_quick_summary.md").write_text("[missing](nope.md)\n")
            (results / "r1_smoke.md").write_text("[missing](nope.md)\n")
            (backups / "review.md").write_text("[missing](nope.md)\n")

            self.assertEqual(
                markdown_files(root), [tracked_backup, archived_quick, canonical]
            )
            self.assertEqual(documentation_problems(root), [])

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

    def test_rejects_machine_details_and_stale_status_in_current_entry_points(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("README.md").write_text(
                "Run from /Users/alice/work/flashdec/ on 192.168.1.42.\n"
                "Do not paste C:\\Users\\bob\\flashdec\\ output.\n"
                "FlashDec is a private `0.0.0` development candidate.\n"
            )

            problems = public_safety_problems(root)

            self.assertEqual(len(problems), 4)
            self.assertTrue(any("personal absolute path" in item for item in problems))
            self.assertTrue(any("private-network IP" in item for item in problems))
            self.assertTrue(any("stale private-repository status" in item for item in problems))

    def test_accepts_placeholders_technical_private_terms_and_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weekly = root / "docs" / "weekly"
            weekly.mkdir(parents=True)
            root.joinpath("README.md").write_text(
                "Use /home/<user>/work/flashdec/ or C:\\Users\\<username>\\repo\\.\n"
                "The request-private transition preserves the private tail.\n"
            )
            root.joinpath("SECURITY.md").write_text(
                "Use private vulnerability reporting and remove private paths.\n"
            )
            (weekly / "week_1.md").write_text(
                "At that time the project used private maintenance.\n"
                "Host: 192.168.1.42\n"
            )

            self.assertEqual(public_safety_problems(root), [])

    def test_rejects_stale_visibility_wording_in_current_design_and_plan_docs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "design.md").write_text(
                "仓库在最终 public-readiness gate 完成前仍为 private。\n"
            )
            (docs / "ROADMAP.md").write_text(
                "The repository remains private until the visibility change.\n"
            )

            problems = public_safety_problems(root)

            self.assertEqual(len(problems), 2)
            self.assertTrue(
                all("stale private-repository status" in item for item in problems)
            )


if __name__ == "__main__":
    unittest.main()
