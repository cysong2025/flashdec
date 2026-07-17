"""Dependency-free configuration coverage for the R3-C benchmark."""

import unittest

from benchmarks.run_shared_prefix_workload import (
    HIT_RATES,
    SharedPrefixConfig,
    _quick_config,
    _selected_hit_rates,
    _trial_hit_rates,
    _validate_config,
)


class SharedPrefixWorkloadBenchmarkTests(unittest.TestCase):
    def test_formal_config_has_fixed_comparable_capacity(self):
        config = SharedPrefixConfig()
        self.assertEqual(config.prefix_blocks, 4)
        self.assertEqual(config.decode_tokens, 12)
        self.assertEqual(config.tail_blocks, 1)
        self.assertEqual(config.no_share_lifetime_blocks, 80)
        self.assertEqual(config.capacity_probe_blocks, 48)
        _validate_config(config, HIT_RATES)

    def test_quick_config_preserves_all_integral_hit_rates(self):
        config = _quick_config(SharedPrefixConfig())
        self.assertEqual(config.request_count, 4)
        self.assertEqual(config.context_tokens, 32)
        self.assertEqual(config.warmup, 1)
        self.assertEqual(config.repeat, 3)
        self.assertEqual(config.no_share_lifetime_blocks, 8)
        self.assertEqual(config.capacity_probe_blocks, 5)
        _validate_config(config, HIT_RATES)

    def test_hit_rate_selection_and_trial_rotation_are_deterministic(self):
        self.assertEqual(_selected_hit_rates("all"), [0, 25, 50, 75])
        self.assertEqual(_selected_hit_rates("50"), [50])
        self.assertEqual(_trial_hit_rates(HIT_RATES, 0), [0, 25, 50, 75])
        self.assertEqual(_trial_hit_rates(HIT_RATES, 1), [25, 50, 75, 0])
        self.assertEqual(_trial_hit_rates(HIT_RATES, 3), [75, 0, 25, 50])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _trial_hit_rates(HIT_RATES, -1)

    def test_config_rejects_partial_prefix_and_fractional_hit_count(self):
        with self.assertRaisesRegex(ValueError, "full-block"):
            _validate_config(
                SharedPrefixConfig(context_tokens=33),
                HIT_RATES,
            )
        with self.assertRaisesRegex(ValueError, "integral"):
            _validate_config(
                SharedPrefixConfig(request_count=6),
                HIT_RATES,
            )


if __name__ == "__main__":
    unittest.main()
