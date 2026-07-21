"""Dependency-free configuration coverage for the R4-C CUDA runner."""

import unittest

from benchmarks.run_integrated_scheduled_multi_layer import (
    CASES,
    _quick_case,
    _selected_cases,
    _trial_case_order,
)


class IntegratedWorkloadBenchmarkTests(unittest.TestCase):
    def test_formal_matrix_uses_two_and_four_layers(self):
        self.assertEqual(len(CASES), 4)
        self.assertEqual(
            {
                (case.num_layers, case.context_tokens, case.prefix_blocks, case.max_blocks)
                for case in CASES.values()
            },
            {
                (2, 64, 2, 8),
                (2, 128, 4, 12),
                (4, 64, 2, 8),
                (4, 128, 4, 12),
            },
        )

    def test_case_order_rotates_without_mutation(self):
        cases = list(CASES.values())
        self.assertEqual(_trial_case_order(cases, 0), cases)
        self.assertEqual(_trial_case_order(cases, 1), cases[1:] + cases[:1])
        self.assertEqual(_trial_case_order(cases, 3), cases[3:] + cases[:3])
        self.assertEqual(list(CASES.values()), cases)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _trial_case_order(cases, -1)

    def test_quick_case_keeps_full_block_prefix_and_reduces_context(self):
        quick = _quick_case(CASES["l4_c128"])
        self.assertEqual(quick.name, "l4_c64")
        self.assertEqual(quick.prefix_blocks, 2)
        self.assertEqual(quick.max_blocks, 8)
        self.assertEqual(_selected_cases("l4_c128", True), [quick])


if __name__ == "__main__":
    unittest.main()
