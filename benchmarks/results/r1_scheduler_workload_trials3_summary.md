# Scheduler Policy Workload Summary

## Validation

- Input: `benchmarks/results/r1_scheduler_workload_trials3.csv`.
- Rows: 36; expected trials: 3.
- Device: NVIDIA GeForce RTX 5070.
- Git commit: `16de9d4`.
- Exact case/dtype/policy/trial matrix and policy-specific invariants passed.

## Cross-trial Medians

| case | dtype | policy | completion | cancellations | deadlocks | p50 ms | p99 ms | useful TPS | wait p90 | scheduler p50 ms | max committed/physical |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boundary_deadlock | float16 | cancel_on_backpressure | 0.500 | 1 | 0 | 1.100729 | 2.100655 | 846.098 | 0.0 | 0.002944 | 0/2 |
| boundary_deadlock | float16 | greedy_step_only | 0.000 | 0 | 1 | 1.060858 | 1.763732 | 0.000 | 0.0 | 0.003054 | 0/2 |
| boundary_deadlock | float16 | lifetime_fifo_aging | 1.000 | 0 | 0 | 1.263126 | 3.114769 | 723.168 | 64.0 | 0.026644 | 2/2 |
| boundary_deadlock | bfloat16 | cancel_on_backpressure | 0.500 | 1 | 0 | 1.142498 | 1.979975 | 797.310 | 0.0 | 0.003205 | 0/2 |
| boundary_deadlock | bfloat16 | greedy_step_only | 0.000 | 0 | 1 | 1.163721 | 2.717084 | 0.000 | 0.0 | 0.003395 | 0/2 |
| boundary_deadlock | bfloat16 | lifetime_fifo_aging | 1.000 | 0 | 0 | 1.141757 | 2.640928 | 687.754 | 64.0 | 0.024147 | 2/2 |
| finite_queue | float16 | cancel_on_backpressure | 1.000 | 0 | 0 | 1.161623 | 7.501146 | 1235.963 | 21.0 | 0.005047 | 0/3 |
| finite_queue | float16 | greedy_step_only | 1.000 | 0 | 0 | 1.183553 | 3.772871 | 1455.282 | 21.0 | 0.005037 | 0/3 |
| finite_queue | float16 | lifetime_fifo_aging | 1.000 | 0 | 0 | 1.214667 | 1.897962 | 1505.926 | 21.0 | 0.032115 | 3/3 |
| finite_queue | bfloat16 | cancel_on_backpressure | 1.000 | 0 | 0 | 1.284488 | 3.570515 | 1303.505 | 21.0 | 0.005439 | 0/3 |
| finite_queue | bfloat16 | greedy_step_only | 1.000 | 0 | 0 | 1.130100 | 2.999268 | 1493.448 | 21.0 | 0.004437 | 0/3 |
| finite_queue | bfloat16 | lifetime_fifo_aging | 1.000 | 0 | 0 | 1.227714 | 3.788622 | 1354.027 | 21.0 | 0.034959 | 3/3 |

## Interpretation

- `useful TPS` only counts tokens belonging to requests that eventually completed.
- Boundary deadlock is a correctness/progress result; latency is secondary when completion differs.
- Scheduler and complete-step timings share one row but should still be interpreted with completion, wait, and memory metrics.
