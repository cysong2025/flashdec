"""Dependency-free configuration coverage for scheduler policy workloads."""

import unittest

from flashdec.scheduled_workload import (
    GREEDY_STEP_ONLY,
    RequestArrival,
    SchedulerWorkloadConfig,
    boundary_deadlock_arrivals,
)
from flashdec.scheduler import RequestSpec


class SchedulerWorkloadConfigTests(unittest.TestCase):
    def test_boundary_trace_is_deterministic(self):
        first = boundary_deadlock_arrivals(num_requests=3, max_new_tokens=4)
        second = boundary_deadlock_arrivals(num_requests=3, max_new_tokens=4)

        self.assertEqual(first, second)
        self.assertEqual(tuple(item.spec.request_id for item in first), (0, 1, 2))
        self.assertEqual(tuple(item.arrival_step for item in first), (0, 0, 0))

    def test_config_normalizes_arrivals_and_rejects_invalid_values(self):
        arrival = RequestArrival(RequestSpec("r", 0, 2, 0), arrival_step=0)
        config = SchedulerWorkloadConfig(
            name="valid",
            arrivals=[arrival],
            policy=GREEDY_STEP_ONLY,
            max_active_requests=1,
            max_batch_requests=1,
        )
        self.assertEqual(config.arrivals, (arrival,))

        cases = (
            ({"name": ""}, "name"),
            ({"policy": "unknown"}, "policy"),
            ({"max_active_requests": 0}, "max_active_requests"),
            ({"max_batch_requests": 2}, "max_batch_requests"),
            ({"max_stalled_steps": 0}, "max_stalled_steps"),
            ({"max_steps": 1, "arrivals": (RequestArrival(RequestSpec("r2", 0, 1, 1), 1),)}, "arrival_step"),
        )
        defaults = {
            "name": "invalid",
            "arrivals": (arrival,),
            "policy": GREEDY_STEP_ONLY,
            "max_active_requests": 1,
            "max_batch_requests": 1,
        }
        for overrides, message in cases:
            with self.subTest(message=message):
                values = dict(defaults)
                values.update(overrides)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    SchedulerWorkloadConfig(**values)


if __name__ == "__main__":
    unittest.main()
