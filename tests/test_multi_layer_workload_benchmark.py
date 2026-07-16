"""Dependency-free configuration tests for the multi-layer benchmark."""

from types import SimpleNamespace
import unittest

from benchmarks.run_multi_layer_engine import (
    CASES,
    _max_blocks,
    _profiler_kwargs,
    _quick_case,
    _selected_cases,
    _trial_backend_order,
)


class MultiLayerWorkloadBenchmarkTests(unittest.TestCase):
    def test_formal_matrix_has_expected_shapes(self):
        self.assertEqual(len(CASES), 12)
        self.assertEqual(
            {
                (case.num_layers, case.batch_size, case.context_tokens)
                for case in CASES.values()
            },
            {
                (layers, batch, context)
                for layers in (1, 2, 4)
                for batch in (4, 16)
                for context in (128, 1024)
            },
        )

    def test_backend_order_alternates_without_mutation(self):
        backends = ["torch", "fused_cuda"]
        self.assertEqual(
            _trial_backend_order(backends, 0), ["torch", "fused_cuda"]
        )
        self.assertEqual(
            _trial_backend_order(backends, 1), ["fused_cuda", "torch"]
        )
        self.assertEqual(backends, ["torch", "fused_cuda"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _trial_backend_order(backends, -1)

    def test_quick_case_preserves_identity_and_reduces_context(self):
        case = CASES["l4_b16_c1024"]
        quick = _quick_case(case)
        self.assertEqual(quick.name, "l4_b16_c64")
        self.assertEqual(quick.num_layers, 4)
        self.assertEqual(quick.batch_size, 16)
        self.assertEqual(quick.context_tokens, 64)
        self.assertEqual(
            _selected_cases("l4_b16_c1024", quick=True), [quick]
        )

    def test_max_blocks_covers_context_measurement_and_boundary(self):
        case = CASES["l2_b4_c128"]
        self.assertEqual(_max_blocks(case, 20), 20)

    def test_profiler_accumulates_events_across_internal_cycles(self):
        torch = SimpleNamespace(
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")
            )
        )
        self.assertEqual(
            _profiler_kwargs(torch),
            {"activities": ["cpu", "cuda"], "acc_events": True},
        )


if __name__ == "__main__":
    unittest.main()
