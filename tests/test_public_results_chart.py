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
        self.assertIn("not comparable across panels", artifact["ratio_boundary"])

        sources = {
            self.data["scheduler_progress"]["source"],
            self.data["shared_prefix_capacity"]["source"],
            self.data["integrated_lifecycle"]["source"],
            self.data["flashinfer_kernel_baseline"]["source"],
            *(
                row["source"]
                for row in self.data["optimization_outcomes"]["entries"]
            ),
        }
        self.assertEqual(
            sources,
            {
                "benchmarks/results/scheduler_capacity_progress_summary.md",
                "benchmarks/results/multi_layer_transaction_summary.md",
                "benchmarks/results/shared_prefix_capacity_summary.md",
                "benchmarks/results/trusted_transaction_summary.md",
                "benchmarks/results/persistent_metadata_candidate_summary.md",
                "benchmarks/results/integrated_runtime_lifecycle_summary.md",
                "benchmarks/results/flashinfer_paged_decode_baseline_summary.md",
            },
        )
        self.assertTrue(all(ROOT.joinpath(source).is_file() for source in sources))

        optimization = {
            row["id"]: row for row in self.data["optimization_outcomes"]["entries"]
        }
        provenance = {
            "scheduler_progress": self.data["scheduler_progress"],
            "fused_append": optimization["fused_append"],
            "shared_prefix_capacity": self.data["shared_prefix_capacity"],
            "trusted_transaction": optimization["trusted_transaction"],
            "persistent_metadata": optimization["persistent_metadata"],
            "integrated_lifecycle": self.data["integrated_lifecycle"],
            "flashinfer_kernel_baseline": self.data[
                "flashinfer_kernel_baseline"
            ],
        }
        self.assertEqual(
            {
                evidence_id: (
                    section["evidence_commit"],
                    section["rows"],
                    section["trials"],
                )
                for evidence_id, section in provenance.items()
            },
            {
                "scheduler_progress": ("16de9d4", 36, 3),
                "fused_append": ("fa0f89a", 144, 3),
                "shared_prefix_capacity": ("fe72e27", 64, 8),
                "trusted_transaction": ("4018449", 160, 5),
                "persistent_metadata": ("8047a9c", 160, 5),
                "integrated_lifecycle": ("6912894", 24, 3),
                "flashinfer_kernel_baseline": ("d7d4feb", 72, 3),
            },
        )

    def test_snapshot_schema_uses_mechanism_names(self):
        self.assertEqual(self.data["schema_version"], 2)
        self.assertTrue(
            {
                "scheduler_progress",
                "shared_prefix_capacity",
                "optimization_outcomes",
                "integrated_lifecycle",
                "flashinfer_kernel_baseline",
            }.issubset(self.data)
        )
        optimization = self.data["optimization_outcomes"]
        self.assertNotIn("scope_boundary", optimization)
        self.assertIn("different workloads and timing boundaries", optimization["technical_boundary"])
        self.assertEqual(
            [(row["id"], row["feature"]) for row in optimization["entries"]],
            [
                ("fused_append", "Fused append path"),
                ("trusted_transaction", "Trusted transaction path"),
                ("persistent_metadata", "Persistent metadata"),
            ],
        )

        def mapping_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from mapping_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from mapping_keys(child)

        self.assertNotIn("stage", set(mapping_keys(self.data)))

    def test_snapshot_preserves_required_results_and_boundaries(self):
        scheduler = self.data["scheduler_progress"]
        outcomes = {
            row["id"]: (row["completed"], row["cancelled"], row["deadlock"])
            for row in scheduler["policies"]
        }
        self.assertEqual(outcomes["lifetime_fifo_aging"], (2, 0, False))
        self.assertEqual(outcomes["cancel_on_backpressure"], (1, 1, False))
        self.assertEqual(outcomes["greedy_step_only"], (0, 0, True))

        shared_prefix_points = self.data["shared_prefix_capacity"]["points"]
        self.assertEqual(
            [row["physical_context_blocks"] for row in shared_prefix_points],
            [64, 52, 36, 20],
        )
        self.assertEqual(
            [row["admitted_requests"] for row in shared_prefix_points],
            [9, 12, 15, 16],
        )
        self.assertEqual(
            [row["saved_kv_capacity_mib"] for row in shared_prefix_points],
            [0.0, 1.5, 3.5, 5.5],
        )

        scorecard = {
            row["id"]: row for row in self.data["optimization_outcomes"]["entries"]
        }
        self.assertEqual(scorecard["fused_append"]["stable_groups"], 20)
        self.assertEqual(scorecard["fused_append"]["outcome"], "observed")
        self.assertEqual(scorecard["trusted_transaction"]["stable_groups"], 16)
        self.assertEqual(scorecard["persistent_metadata"]["stable_groups"], 13)
        self.assertEqual(
            scorecard["persistent_metadata"]["outcome"],
            "rejected_and_rolled_back",
        )

        integrated = self.data["integrated_lifecycle"]
        self.assertEqual(
            (integrated["status"], integrated["rows"], integrated["trials"]),
            ("passed", 24, 3),
        )

        external_baseline = self.data["flashinfer_kernel_baseline"]
        self.assertIn("kernel only", external_baseline["scope"])
        self.assertIn(
            "above 1 favor FlashInfer",
            external_baseline["ratio_direction"],
        )
        self.assertEqual(
            [
                row["p50_ratio_geometric_mean"]
                for row in external_baseline["backends"]
            ],
            [1.2003, 1.2284],
        )
        self.assertEqual(
            (
                external_baseline["observed_p50_ranges_above_one"],
                external_baseline["total_p50_ranges"],
            ),
            (16, 16),
        )

    def test_validate_snapshot_rejects_a_misrepresented_metadata_decision(self):
        changed = json.loads(json.dumps(self.data))
        changed["optimization_outcomes"]["entries"][2]["outcome"] = "accepted"
        with self.assertRaisesRegex(ValueError, "scorecard"):
            validate_snapshot(changed)

    def test_validate_snapshot_rejects_a_hard_coded_external_range_count(self):
        changed = json.loads(json.dumps(self.data))
        changed["flashinfer_kernel_baseline"][
            "observed_p50_ranges_above_one"
        ] = 15
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
                [item["id"] for item in embedded["evidence"]],
                [
                    "scheduler_progress",
                    "fused_append",
                    "shared_prefix_capacity",
                    "trusted_transaction",
                    "persistent_metadata",
                    "integrated_lifecycle",
                    "flashinfer_kernel_baseline",
                ],
            )
            self.assertTrue(
                all("stage" not in item for item in embedded["evidence"])
            )
            self.assertNotIn("class-=", tracked)
            self.assertIn('class="panel"', tracked)
            visible_text = " ".join(document.itertext())
            self.assertIn("Persistent metadata was not adopted", visible_text)
            self.assertIn("Integrated lifecycle", visible_text)
            self.assertIn("NOT ADOPTED", visible_text)
            self.assertIn(">1 favors FlashInfer", visible_text)
            self.assertIn("KERNEL-ONLY", visible_text)
            self.assertIn("not a raw dataset", visible_text)
            self.assertIn("not comparable across panels", visible_text)
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
