# Paged Decode Warp Selection Summary

## Provenance

- Recorded experiment commit: `aa81af8` (`2026-06-29`).
- Device/OS: NVIDIA GeForce RTX 5070, Linux/WSL2.
- Python/PyTorch/Triton/PyTorch CUDA: 3.12.3 / 2.11.0+cu128 / 3.6.0 / 12.8.
- Historical runner: `python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv`.
- Current equivalent runner: `python benchmarks/run_paged_decode_warp_sweep.py --dtype both --block-size 16 --kv-layout token_major --num-warps 2 4 8 --warmup 5 --repeat 30 --seed 87 --output benchmarks/results/paged_decode_warp_sweep.csv`.
- Matrix: FP16/BF16; batch `1/2/4/8/16/32/64/128` at context 1024; context `128/256/512/2048/4096/8192` at batch 16; `num_warps=2/4/8`; head dimension 128; 32 query heads; 8 KV heads; block size 16.
- Sampling: one recorded full sweep, seed 87, warmup 5, repeat 30; 84 Triton rows across 28 dtype/case groups.
- Every configuration was reference-validated before timing. CUDA events covered the kernel path.

The raw CSV was intentionally not tracked. This file preserves the exact public tables previously embedded in `docs/perf_experiments.md`; it is a provenance migration, not a newly executed benchmark. The original record remains inspectable with `git show aa81af8:docs/perf_experiments.md`.

## Selection Result

| metric | result |
| --- | ---: |
| dtype/case groups | 28 |
| `num_warps=2` p50 wins | 28 |
| `num_warps=4` p50 wins | 0 |
| `num_warps=8` p50 wins | 0 |
| w2 vs w4 p50 speedup range | 1.00x–2.99x |
| w2 vs w4 mean p50 speedup | 2.10x |
| w2 vs w8 p50 speedup range | 1.33x–4.81x |
| w2 vs w8 mean p50 speedup | 3.75x |

## Full-sweep Absolute Results for the Selected Configuration

The tables below retain every full-sweep group at `num_warps=2`, the selected configuration. `effective_total_gbps_p50` is a logical workload estimate rather than measured DRAM bandwidth.

### Batch sweep

| dtype | batch | p50 ms | p90 ms | mean ms | total context tokens | effective total GB/s at p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 1 | 0.060672 | 0.061664 | 0.065063 | 762 | 206.1456 |
| float16 | 2 | 0.060064 | 0.096576 | 0.066580 | 1306 | 356.9824 |
| float16 | 4 | 0.059360 | 0.088352 | 0.064257 | 2504 | 692.6059 |
| float16 | 8 | 0.124992 | 0.144896 | 0.128722 | 5698 | 748.4526 |
| float16 | 16 | 0.212768 | 0.218784 | 0.212847 | 13438 | 1036.6383 |
| float16 | 32 | 0.371712 | 0.377504 | 0.373450 | 23033 | 1017.3554 |
| float16 | 64 | 0.600288 | 0.611936 | 0.605295 | 49266 | 1347.2787 |
| float16 | 128 | 1.142720 | 1.162752 | 1.146318 | 99176 | 1424.7250 |
| bfloat16 | 1 | 0.061184 | 0.062880 | 0.065021 | 762 | 204.4205 |
| bfloat16 | 2 | 0.061440 | 0.101696 | 0.075761 | 1306 | 348.9875 |
| bfloat16 | 4 | 0.059232 | 0.086592 | 0.065114 | 2504 | 694.1026 |
| bfloat16 | 8 | 0.119968 | 0.145792 | 0.120962 | 5698 | 779.7962 |
| bfloat16 | 16 | 0.213792 | 0.219008 | 0.215164 | 13438 | 1031.6731 |
| bfloat16 | 32 | 0.375392 | 0.383040 | 0.375718 | 23033 | 1007.3822 |
| bfloat16 | 64 | 0.603648 | 0.609216 | 0.604226 | 49266 | 1339.7795 |
| bfloat16 | 128 | 1.153120 | 1.175040 | 1.190499 | 99176 | 1411.8753 |

### Context sweep

| dtype | maximum context | p50 ms | p90 ms | mean ms | total context tokens | effective total GB/s at p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 128 | 0.029568 | 0.059904 | 0.041477 | 1559 | 873.3507 |
| float16 | 256 | 0.047584 | 0.052608 | 0.052138 | 3091 | 1070.5259 |
| float16 | 512 | 0.089376 | 0.132224 | 0.105509 | 6138 | 1128.8793 |
| float16 | 2048 | 0.359296 | 0.366496 | 0.361342 | 23757 | 1084.7567 |
| float16 | 4096 | 0.663072 | 0.679904 | 0.667005 | 51899 | 1283.5304 |
| float16 | 8192 | 1.322528 | 1.328288 | 1.321173 | 99883 | 1238.3688 |
| bfloat16 | 128 | 0.029952 | 0.035104 | 0.033959 | 1559 | 862.1538 |
| bfloat16 | 256 | 0.048992 | 0.075136 | 0.054050 | 3091 | 1039.7596 |
| bfloat16 | 512 | 0.087840 | 0.130496 | 0.105999 | 6138 | 1148.6193 |
| bfloat16 | 2048 | 0.361216 | 0.370176 | 0.362059 | 23757 | 1078.9908 |
| bfloat16 | 4096 | 0.664480 | 0.685536 | 0.668438 | 51899 | 1280.8107 |
| bfloat16 | 8192 | 1.322112 | 1.331520 | 1.325954 | 99883 | 1238.7585 |

## Quick Paired Confirmation

The preceding-day quick sweep retained direct per-case p50 values for all three configurations. It is supporting evidence only; the full-sweep selection statistics above determined the default.

| dtype | case | w2 p50 ms | w4 p50 ms | w8 p50 ms | best | best effective total GB/s at p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| float16 | batch_b1_ctx1024 | 0.060704 | 0.061920 | 0.081824 | 2 | 206.0369 |
| float16 | batch_b16_ctx1024 | 0.206656 | 0.426976 | 0.758464 | 2 | 944.0942 |
| float16 | batch_b64_ctx1024 | 0.593024 | 1.454432 | 2.732800 | 2 | 1358.9605 |
| float16 | context_b16_ctx128 | 0.029120 | 0.058304 | 0.104928 | 2 | 892.4132 |
| float16 | context_b16_ctx4096 | 0.680000 | 1.638880 | 2.821824 | 2 | 1193.8966 |
| bfloat16 | batch_b1_ctx1024 | 0.060960 | 0.062304 | 0.082048 | 2 | 205.1717 |
| bfloat16 | batch_b16_ctx1024 | 0.208576 | 0.429088 | 0.753888 | 2 | 935.4035 |
| bfloat16 | batch_b64_ctx1024 | 0.590688 | 1.453280 | 2.711936 | 2 | 1364.3348 |
| bfloat16 | context_b16_ctx128 | 0.029184 | 0.057696 | 0.100064 | 2 | 890.4561 |
| bfloat16 | context_b16_ctx4096 | 0.681664 | 1.651424 | 2.804256 | 2 | 1190.9823 |

## Interpretation Boundary

- The evidence selects two warps for this Triton program mapping on the recorded RTX 5070 environment; it is not a general statement about Triton kernels or other GPUs.
- The experiment used block size 16. Later experiments selected block size 32 and separately checked the block-size/warp interaction before adopting the combined default.
- Sub-0.1 ms cases are sensitive to launch and host noise. The selection relies on the full 28-group direction, not one small shape.
- Logical effective GB/s excludes implementation-specific traffic, cache behavior and metadata rereads.
