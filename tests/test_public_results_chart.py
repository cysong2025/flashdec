"""Dependency-free validation for the tracked public evidence overview."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

from scripts.generate_public_results_chart import (
    DEFAULT_OUTPUTS,
    DEFAULT_SNAPSHOT,
    ROOT,
    check_outputs,
    load_snapshot,
    render_svg,
    validate_snapshot,
)


class PublicResultsChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_snapshot()

    def test_snapshot_is_processed_and_canonical_sources_are_audited(self):
        artifact = self.data["artifact"]
        self.assertEqual(artifact["device"], "NVIDIA GeForce RTX 5070")
        self.assertIn("not a raw benchmark dataset", artifact["data_class"])
        self.assertIn("not comparable across stages", artifact["ratio_boundary"])

        sources = {
            self.data["r1_scheduler_progress"]["source"],
            self.data["r3_shared_prefix_capacity"]["source"],
            self.data["r4c_integrated_correctness"]["source"],
            self.data["r5_flashinfer_kernel_baseline"]["source"],
            *(
                row["source"]
                for row in self.data["optimization_outcomes"]["entries"]
            ),
        }
        self.assertEqual(len(sources), 7)
        self.assertTrue(all(ROOT.joinpath(source).is_file() for source in sources))

        provenance = {
            "R1": self.data["r1_scheduler_progress"],
            "R2": self.data["optimization_outcomes"]["entries"][0],
            "R3": self.data["r3_shared_prefix_capacity"],
            "R4-A": self.data["optimization_outcomes"]["entries"][1],
            "R4-B": self.data["optimization_outcomes"]["entries"][2],
            "R4-C": self.data["r4c_integrated_correctness"],
            "R5": self.data["r5_flashinfer_kernel_baseline"],
        }
        self.assertEqual(
            {
                stage: (
                    section["evidence_commit"],
                    section["rows"],
                    section["trials"],
                )
                for stage, section in provenance.items()
            },
            {
                "R1": ("16de9d4", 36, 3),
                "R2": ("fa0f89a", 144, 3),
                "R3": ("fe72e27", 64, 8),
                "R4-A": ("4018449", 160, 5),
                "R4-B": ("8047a9c", 160, 5),
                "R4-C": ("6912894", 24, 3),
                "R5": ("d7d4feb", 72, 3),
            },
        )

    def test_snapshot_preserves_required_results_and_boundaries(self):
        r1 = self.data["r1_scheduler_progress"]
        outcomes = {
            row["id"]: (row["completed"], row["cancelled"], row["deadlock"])
            for row in r1["policies"]
        }
        self.assertEqual(outcomes["lifetime_fifo_aging"], (2, 0, False))
        self.assertEqual(outcomes["cancel_on_backpressure"], (1, 1, False))
        self.assertEqual(outcomes["greedy_step_only"], (0, 0, True))

        r3_points = self.data["r3_shared_prefix_capacity"]["points"]
        self.assertEqual(
            [row["physical_context_blocks"] for row in r3_points],
            [64, 52, 36, 20],
        )
        self.assertEqual(
            [row["admitted_requests"] for row in r3_points],
            [9, 12, 15, 16],
        )
        self.assertEqual(
            [row["saved_kv_capacity_mib"] for row in r3_points],
            [0.0, 1.5, 3.5, 5.5],
        )

        scorecard = {
            row["stage"]: row for row in self.data["optimization_outcomes"]["entries"]
        }
        self.assertEqual(scorecard["R2"]["stable_groups"], 20)
        self.assertEqual(scorecard["R2"]["outcome"], "observed")
        self.assertEqual(scorecard["R4-A"]["stable_groups"], 16)
        self.assertEqual(scorecard["R4-B"]["stable_groups"], 13)
        self.assertEqual(scorecard["R4-B"]["outcome"], "rejected_and_rolled_back")

        r4c = self.data["r4c_integrated_correctness"]
        self.assertEqual((r4c["status"], r4c["rows"], r4c["trials"]), ("passed", 24, 3))

        r5 = self.data["r5_flashinfer_kernel_baseline"]
        self.assertIn("kernel only", r5["scope"])
        self.assertIn("above 1 favor FlashInfer", r5["ratio_direction"])
        self.assertEqual(
            [row["p50_ratio_geometric_mean"] for row in r5["backends"]],
            [1.2003, 1.2284],
        )
        self.assertEqual(
            (r5["observed_p50_ranges_above_one"], r5["total_p50_ranges"]),
            (16, 16),
        )

    def test_validate_snapshot_rejects_a_misrepresented_r4b_decision(self):
        changed = json.loads(json.dumps(self.data))
        changed["optimization_outcomes"]["entries"][2]["outcome"] = "accepted"
        with self.assertRaisesRegex(ValueError, "scorecard"):
            validate_snapshot(changed)

    def test_validate_snapshot_rejects_a_hard_coded_r5_range_count(self):
        changed = json.loads(json.dumps(self.data))
        changed["r5_flashinfer_kernel_baseline"]["observed_p50_ranges_above_one"] = 15
        with self.assertRaisesRegex(ValueError, "range counts"):
            validate_snapshot(changed)

    def test_tracked_svgs_are_deterministic_and_accessible(self):
        self.assertEqual(check_outputs(self.data), [])
        for theme, path in DEFAULT_OUTPUTS.items():
            tracked = path.read_text(encoding="utf-8")
            self.assertEqual(tracked, render_svg(self.data, theme))
            document = ElementTree.fromstring(tracked)
            self.assertEqual(document.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(document.attrib["role"], "img")
            self.assertEqual(
                document.attrib["aria-labelledby"],
                "chart-title chart-description",
            )
            self.assertEqual(document.attrib["data-theme"], theme)
            tags = {element.tag.rsplit("}", 1)[-1] for element in document.iter()}
            self.assertTrue({"title", "desc", "metadata"}.issubset(tags))
            title = document.find("{http://www.w3.org/2000/svg}title")
            description = document.find("{http://www.w3.org/2000/svg}desc")
            metadata = document.find("{http://www.w3.org/2000/svg}metadata")
            self.assertEqual(title.attrib["id"], "chart-title")
            self.assertEqual(description.attrib["id"], "chart-description")
            embedded = json.loads(metadata.text)
            self.assertEqual(
                [item["stage"] for item in embedded["evidence"]],
                ["R1", "R2", "R3", "R4-A", "R4-B", "R4-C", "R5"],
            )
            self.assertNotIn("class-=", tracked)
            self.assertIn('class="panel"', tracked)
            visible_text = " ".join(document.itertext())
            self.assertIn("R4-B rolled back", visible_text)
            self.assertIn("R4-C lifecycle PASS", visible_text)
            self.assertIn("REJECTED", visible_text)
            self.assertIn(">1 favors FlashInfer", visible_text)
            self.assertIn("KERNEL-ONLY", visible_text)
            self.assertIn("not a raw dataset", visible_text)
            self.assertIn("not comparable across stages", visible_text)
            for commit in (
                "16de9d4",
                "fa0f89a",
                "fe72e27",
                "4018449",
                "8047a9c",
                "6912894",
                "d7d4feb",
            ):
                self.assertIn(commit, tracked)
            self.assertNotRegex(tracked, r'="\d+\.\d{4,}"')

    def test_check_detects_a_stale_output(self):
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            outputs = {
                "light": temporary / "light.svg",
                "dark": temporary / "dark.svg",
            }
            for theme, path in outputs.items():
                path.write_text(render_svg(self.data, theme), encoding="utf-8")
            self.assertEqual(check_outputs(self.data, outputs), [])
            outputs["dark"].write_text("<svg/>\n", encoding="utf-8")
            self.assertEqual(len(check_outputs(self.data, outputs)), 1)
            self.assertIn("stale dark chart", check_outputs(self.data, outputs)[0])

    def test_default_snapshot_is_valid_json(self):
        self.assertEqual(
            json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8")),
            self.data,
        )


if __name__ == "__main__":
    unittest.main()
