"""Dependency-free R4-C trace and reference-trajectory coverage."""

import unittest

import flashdec
from flashdec.integrated_workload import (
    IntegratedWorkloadConfig,
    LayerFailure,
    RequestCancellation,
    build_integrated_reference,
    standard_integrated_arrivals,
    standard_integrated_config,
)


class IntegratedWorkloadConfigTests(unittest.TestCase):
    def test_standard_trace_has_stable_mixed_lifecycle_reference(self):
        config = standard_integrated_config(num_layers=2, context_tokens=64)
        reference = build_integrated_reference(
            config,
            block_size=32,
            max_blocks=8,
            resident_prefix_blocks=2,
        )

        self.assertEqual(len(reference.steps), 10)
        self.assertEqual(
            reference.completed_request_ids,
            ("miss-a", "hit-a", "miss-b"),
        )
        self.assertEqual(reference.cancelled_request_ids, ("hit-cancel",))
        self.assertEqual(reference.rejected_request_ids, ())
        self.assertEqual(reference.successful_steps, 9)
        self.assertEqual(reference.aborted_steps, 1)
        self.assertEqual(reference.completed_tokens, 13)
        self.assertEqual(len(reference.digest), 64)
        self.assertTrue(all(step.used_blocks <= step.committed_blocks for step in reference.steps))
        self.assertEqual(reference.steps[4].positions, (64,))
        self.assertTrue(reference.steps[4].aborted)
        self.assertEqual(reference.steps[-1].used_blocks, 2)

    def test_reference_digest_is_deterministic_and_shape_sensitive(self):
        config = standard_integrated_config(num_layers=4, context_tokens=64)
        first = build_integrated_reference(
            config,
            block_size=32,
            max_blocks=8,
            resident_prefix_blocks=2,
        )
        second = build_integrated_reference(
            config,
            block_size=32,
            max_blocks=8,
            resident_prefix_blocks=2,
        )
        different = build_integrated_reference(
            standard_integrated_config(num_layers=4, context_tokens=128),
            block_size=32,
            max_blocks=15,
            resident_prefix_blocks=4,
        )
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, different.digest)

    def test_config_rejects_missing_hit_miss_and_invalid_events(self):
        arrivals = standard_integrated_arrivals(context_tokens=64)
        defaults = {
            "name": "invalid",
            "num_layers": 2,
            "arrivals": arrivals,
        }
        cases = (
            ({"num_layers": 1}, "num_layers"),
            ({"cancellations": (RequestCancellation("unknown", 1),)}, "unknown"),
            ({"failures": (LayerFailure(1, 2),)}, "layer_idx"),
            (
                {
                    "failures": (
                        LayerFailure(1, 0),
                        LayerFailure(1, 1),
                    )
                },
                "at most one",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                values = dict(defaults)
                values.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    IntegratedWorkloadConfig(**values)

    def test_public_api_exports_r4c_symbols(self):
        self.assertIs(flashdec.IntegratedWorkloadConfig, IntegratedWorkloadConfig)
        self.assertIs(flashdec.LayerFailure, LayerFailure)
        self.assertIs(flashdec.RequestCancellation, RequestCancellation)
        self.assertIs(flashdec.build_integrated_reference, build_integrated_reference)
        self.assertIs(flashdec.standard_integrated_config, standard_integrated_config)


if __name__ == "__main__":
    unittest.main()
