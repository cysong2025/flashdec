# R4-C Integrated Scheduled Multi-layer Summary

## Validation

- Input: `/home/user/flashdec_results/r4c_6912894_20260721_212333/r4_integrated_trials3.csv`.
- Rows: 24; trials: 3.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `6912894`.
- Frozen path: fused CUDA append + Triton decode + R4-A materialized transaction metadata.
- Reference digest, dynamic admission/defer/completion/cancellation trajectory, rollback, transaction counts, prefix lifetime, released-block reuse, and final zero-used cleanup were validated.
- This matrix reports one integrated workload; it is not a shared-prefix speedup A/B and does not reopen frozen kernel tuning.

## Correctness Evidence

- Focused: `60 passed, 17 subtests passed in 3.09s`.
- Full: `425 passed, 57 subtests passed in 6.52s`.
- FP16 quick: 1 row, 1 trial, `l2_c32`; strict summary passed.

## Cross-trial Absolute Results

| dtype | case | complete p50 ms [min,max] | p90 ms | p99 ms [min,max] | scheduler p50 ms | context seed p50 ms | Engine p50 ms | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | l2_c128 | 1.360588 [1.356418,1.470878] | 90.111803 | 98.103147 [94.868145,111.947873] | 0.051740 | 0.002539 | 1.297275 | 66.123 |
| bfloat16 | l2_c64 | 1.389153 [1.350394,1.394155] | 45.809869 | 48.965221 [46.763219,77.612292] | 0.042038 | 0.002254 | 1.295337 | 122.698 |
| bfloat16 | l4_c128 | 2.146924 [2.048538,2.265855] | 135.499771 | 135.946653 [134.657499,161.423105] | 0.043662 | 0.002234 | 2.000556 | 44.887 |
| bfloat16 | l4_c64 | 2.130567 [2.049295,2.434063] | 67.406196 | 67.975530 [67.894562,74.171916] | 0.044498 | 0.002214 | 2.054000 | 83.670 |
| float16 | l2_c128 | 1.446410 [1.378498,1.487093] | 92.374594 | 93.078001 [91.765618,93.854188] | 0.042151 | 0.002293 | 1.357008 | 65.859 |
| float16 | l2_c64 | 1.371000 [1.316504,1.472778] | 45.547011 | 46.313351 [45.771050,47.927300] | 0.043239 | 0.002191 | 1.317445 | 126.641 |
| float16 | l4_c128 | 2.371724 [2.096305,2.432612] | 136.239373 | 148.718392 [142.375568,163.923760] | 0.050875 | 0.002303 | 2.114814 | 43.070 |
| float16 | l4_c64 | 2.188326 [2.046825,2.254027] | 68.713048 | 72.342594 [68.265204,96.158362] | 0.044686 | 0.002270 | 1.975431 | 81.959 |

## Interpretation

- R4-C 的主 gate 是组合 correctness 与 lifecycle closure，不是相对 latency ratio；24 行全部通过 strict validator，因此本阶段验收完成。
- p50 主要反映普通 scheduled multi-layer decode step；p90/p99 来自只有 10 个 logical steps 的有限 trace，并受到 private miss context-write admission steps 主导，不能解释成稳态 decode 尾延迟。
- Context seeding 是 private miss 的 caller-supplied multi-layer prompt state；shared hits 只 attach 固定 resident prefix。Random tensor construction、prefix registration 与 terminal eviction 不在计时区间。
- Terminal cleanup 只在所有 request finished/cancelled/rejected 后执行；第一版没有在线 prefix registration/eviction。
- 该矩阵不比较 R4-B persistent candidate，也不证明 shared prefix 带来 latency speedup；R4-A/materialized 继续作为生产主线。
