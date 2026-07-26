# DecodeEngine Workload Multi-trial Summary

## Validation

- Input: `benchmarks/results/week12_decode_engine_workload_trials3.csv`.
- Rows: 36; paired trials: 18.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `3708b87`.
- Decode backend: triton; block size: 32; num warps: 2.
- All rows passed engine/cache invariants, block accounting, pair trajectory, seed, and backend-order validation.

Ratios above 1 mean `fused_cuda` is better. Latency ratios are `torch/fused`; TPS ratio is `fused/torch`.

## Per-trial Ratios

| dtype | workload | trial | seed | backend order | p50 | p90 | p99 | mean | TPS |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| float16 | short_churn | 1 | 431 | torch->fused_cuda | 1.0042x | 0.9917x | 1.8226x | 1.0928x | 1.0928x |
| float16 | short_churn | 2 | 432 | fused_cuda->torch | 0.9311x | 0.7998x | 3.5203x | 0.9448x | 0.9448x |
| float16 | short_churn | 3 | 433 | torch->fused_cuda | 1.0001x | 1.0284x | 3.1075x | 1.1132x | 1.1132x |
| float16 | mixed_steady | 1 | 431 | torch->fused_cuda | 1.0927x | 1.0646x | 1.2699x | 1.1004x | 1.1004x |
| float16 | mixed_steady | 2 | 432 | fused_cuda->torch | 1.1109x | 1.3136x | 2.0583x | 1.2466x | 1.2466x |
| float16 | mixed_steady | 3 | 433 | torch->fused_cuda | 1.0837x | 1.1248x | 1.2761x | 1.0948x | 1.0948x |
| float16 | long_pressure | 1 | 431 | torch->fused_cuda | 1.1274x | 1.0993x | 1.0103x | 1.1242x | 1.1242x |
| float16 | long_pressure | 2 | 432 | fused_cuda->torch | 1.0614x | 0.9406x | 0.4886x | 0.9237x | 0.9237x |
| float16 | long_pressure | 3 | 433 | torch->fused_cuda | 1.0890x | 1.0900x | 1.0766x | 1.0899x | 1.0899x |
| bfloat16 | short_churn | 1 | 431 | torch->fused_cuda | 0.9892x | 0.9556x | 0.2444x | 0.8886x | 0.8886x |
| bfloat16 | short_churn | 2 | 432 | fused_cuda->torch | 1.0508x | 1.2267x | 0.6758x | 1.0735x | 1.0735x |
| bfloat16 | short_churn | 3 | 433 | torch->fused_cuda | 1.0366x | 1.3596x | 5.0578x | 1.3983x | 1.3983x |
| bfloat16 | mixed_steady | 1 | 431 | torch->fused_cuda | 1.0882x | 1.1779x | 1.1966x | 1.1651x | 1.1651x |
| bfloat16 | mixed_steady | 2 | 432 | fused_cuda->torch | 1.0948x | 1.0419x | 1.0020x | 1.1086x | 1.1086x |
| bfloat16 | mixed_steady | 3 | 433 | torch->fused_cuda | 1.2193x | 1.3672x | 1.3348x | 1.3016x | 1.3016x |
| bfloat16 | long_pressure | 1 | 431 | torch->fused_cuda | 1.1054x | 0.3532x | 0.3759x | 0.7037x | 0.7037x |
| bfloat16 | long_pressure | 2 | 432 | fused_cuda->torch | 1.0741x | 1.1829x | 3.3289x | 1.2272x | 1.2272x |
| bfloat16 | long_pressure | 3 | 433 | torch->fused_cuda | 1.0744x | 1.0854x | 1.0688x | 1.0754x | 1.0754x |

## Cross-trial Aggregates

| dtype | workload | trials | p50 median [min, max] | p90 median | p99 median [min, max] | TPS median | p50 direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bfloat16 | long_pressure | 3 | 1.0744x [1.0741, 1.1054] | 1.0854x | 1.0688x [0.3759, 3.3289] | 1.0754x | fused_faster |
| bfloat16 | mixed_steady | 3 | 1.0948x [1.0882, 1.2193] | 1.1779x | 1.1966x [1.0020, 1.3348] | 1.1651x | fused_faster |
| bfloat16 | short_churn | 3 | 1.0366x [0.9892, 1.0508] | 1.2267x | 0.6758x [0.2444, 5.0578] | 1.0735x | unstable_crosses_1 |
| float16 | long_pressure | 3 | 1.0890x [1.0614, 1.1274] | 1.0900x | 1.0103x [0.4886, 1.0766] | 1.0899x | fused_faster |
| float16 | mixed_steady | 3 | 1.0927x [1.0837, 1.1109] | 1.1248x | 1.2761x [1.2699, 2.0583] | 1.1004x | fused_faster |
| float16 | short_churn | 3 | 1.0001x [0.9311, 1.0042] | 0.9917x | 3.1075x [1.8226, 3.5203] | 1.0928x | unstable_crosses_1 |

## Overall Geometric Mean

| metric | fused vs torch |
| --- | ---: |
| p50 | 1.0668x |
| p90 | 1.0317x |
| p99 | 1.2590x |
| mean latency | 1.0811x |
| tokens/s | 1.0811x |

## Interpretation Rule

- `fused_faster`: all p50 trials are above 1.
- `torch_faster`: all p50 trials are below 1.
- `unstable_crosses_1`: trials cross 1; do not claim a stable backend win.
- p99 must be reported with its trial range; a single outlier is not a release-level conclusion.
