"""Dependency-free checks for the public FlashDec architecture diagram."""

from __future__ import annotations

from xml.etree import ElementTree
import unittest

from scripts.generate_public_architecture import DEFAULT_OUTPUTS, check_outputs, render_svg


class PublicArchitectureTests(unittest.TestCase):
    def test_architecture_svgs_are_deterministic_and_accessible(self):
        self.assertEqual(check_outputs(), [])
        for theme, path in DEFAULT_OUTPUTS.items():
            tracked = path.read_text(encoding="utf-8")
            self.assertEqual(tracked, render_svg(theme))
            document = ElementTree.fromstring(tracked)
            self.assertEqual(document.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(document.attrib["role"], "img")
            self.assertEqual(
                document.attrib["aria-labelledby"],
                "architecture-title architecture-description",
            )
            self.assertEqual(document.attrib["data-theme"], theme)
            title = document.find("{http://www.w3.org/2000/svg}title")
            description = document.find("{http://www.w3.org/2000/svg}desc")
            self.assertIsNotNone(title)
            self.assertEqual(title.attrib["id"], "architecture-title")
            self.assertIsNotNone(description)
            self.assertEqual(description.attrib["id"], "architecture-description")
            visible = " ".join(document.itertext())
            self.assertIn("A focused decode path inside vLLM", visible)
            self.assertIn("FlashDec plugin router", visible)
            self.assertIn("Unsupported path → vLLM Triton", visible)
            self.assertIn("Transactional PagedKVCache", visible)
            self.assertIn("Split-KV PagedAttention", visible)


if __name__ == "__main__":
    unittest.main()
