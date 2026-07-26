# R5 FlashInfer Paged-decode Baseline Summary

## Validation

- Input artifact: `r5_flashinfer_paged_decode_trials3.csv` (local source path normalized for publication; raw CSV is not tracked).
- Rows: 72; trials: 3.
- Device: NVIDIA GeForce RTX 5070.
- Run started: 2026-07-26T15:28:08+08:00.
- Python/PyTorch/Triton/PyTorch CUDA: 3.12.3 / 2.11.0+cu128 / 3.6.0 / 12.8.
- CUDA packages (toolkit/python/bindings/pathfinder): 12.8.1 / 12.9.1 / 12.9.7 / 1.6.0; Ninja: 1.13.0.
- CUDA_HOME: `/usr/local/cuda-12.8` (realpath `/usr/local/cuda-12.8`); NVCC: release 12.8 / V12.8.93.
- FlashInfer CUDA arch list: `12.0a`.
- FlashInfer: `0.6.15.post1` (fixed expected version `0.6.15.post1`).
- FlashInfer workspace: 128 MiB per wrapper.
- Git commit: `d7d4feb`.
- Git worktree was clean when the runner started.
- Runner command (publication-normalized paths): `$PYTHON $REPO/benchmarks/run_flashinfer_baseline.py --case all --dtype both --trials 3 --warmup 10 --repeat 50 --require-clean --output $RESULT_DIR/r5_flashinfer_paged_decode_trials3.csv`.
- Common operation: paged single-token decode with 32 query heads, 8 KV heads, head dimension 128, page size 32, and FP16/BF16.
- FlashInfer consumes its documented `HND` paged layout; FlashDec consumes its physical `token_major` layout. Both views share the same logical pages and page table; no layout conversion is inside timing.
- CUDA events cover `run` only. Input construction, reference validation, FlashInfer plan/JIT, and synchronization setup are excluded.
- Every row passed the sampled PyTorch reference check, full cross-backend parity check, page-table pairing, and invariant validation.

## Paired Cross-trial Results

| dtype | case | external backend | FlashDec p50 ms | external p50 ms | p50 ratio FlashDec/external | FlashDec tokens/s | external tokens/s | TPS ratio external/FlashDec | external logical workload GB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | small_b1_ctx128 | flashinfer_fa2_cuda_core | 0.015616 | 0.014208 | 1.1116x [1.0878,2.8514] | 64036.885 | 70382.883 | 1.1116x [1.0878,2.8514] | 38.0541 |
| float16 | small_b1_ctx128 | flashinfer_fa2_tensor_core | 0.015616 | 0.013120 | 1.1955x [1.1873,4.1659] | 64036.885 | 76219.512 | 1.1955x [1.1873,4.1659] | 41.2098 |
| float16 | medium_b16_ctx1024 | flashinfer_fa2_cuda_core | 0.176128 | 0.157536 | 1.1172x [1.0815,1.1180] | 90843.023 | 101564.087 | 1.1172x [1.0815,1.1180] | 427.6547 |
| float16 | medium_b16_ctx1024 | flashinfer_fa2_tensor_core | 0.176128 | 0.155072 | 1.1360x [1.1331,1.5223] | 90843.023 | 103177.879 | 1.1360x [1.1331,1.5223] | 434.4499 |
| float16 | large_b16_ctx8192 | flashinfer_fa2_cuda_core | 1.014976 | 0.903584 | 1.1237x [1.1154,1.1249] | 15763.920 | 17707.264 | 1.1237x [1.1154,1.1249] | 594.4473 |
| float16 | large_b16_ctx8192 | flashinfer_fa2_tensor_core | 1.014976 | 0.904608 | 1.1200x [1.1197,1.1224] | 15763.920 | 17687.219 | 1.1200x [1.1197,1.1224] | 593.7744 |
| float16 | large_batch_b64_ctx4096 | flashinfer_fa2_cuda_core | 2.562368 | 1.791936 | 1.4299x [1.4166,1.4345] | 24976.896 | 35715.561 | 1.4299x [1.4166,1.4345] | 599.7928 |
| float16 | large_batch_b64_ctx4096 | flashinfer_fa2_tensor_core | 2.562368 | 1.763680 | 1.4529x [1.4518,1.4552] | 24976.896 | 36287.762 | 1.4529x [1.4518,1.4552] | 609.4022 |
| bfloat16 | small_b1_ctx128 | flashinfer_fa2_cuda_core | 0.016096 | 0.014432 | 1.1228x [1.0231,1.1574] | 62127.237 | 69290.466 | 1.1228x [1.0231,1.1574] | 37.4634 |
| bfloat16 | small_b1_ctx128 | flashinfer_fa2_tensor_core | 0.016096 | 0.013664 | 1.1780x [1.1486,1.1945] | 62127.237 | 73185.012 | 1.1780x [1.1486,1.1945] | 39.5691 |
| bfloat16 | medium_b16_ctx1024 | flashinfer_fa2_cuda_core | 0.181664 | 0.157824 | 1.1489x [1.1464,1.1539] | 88074.687 | 101378.751 | 1.1489x [1.1464,1.1539] | 426.8743 |
| bfloat16 | medium_b16_ctx1024 | flashinfer_fa2_tensor_core | 0.181664 | 0.155488 | 1.1664x [1.1663,1.1690] | 88074.687 | 102901.832 | 1.1664x [1.1663,1.1690] | 433.2875 |
| bfloat16 | large_b16_ctx8192 | flashinfer_fa2_cuda_core | 1.050496 | 0.901088 | 1.1658x [1.1622,1.2396] | 15230.900 | 17756.312 | 1.1658x [1.1622,1.2396] | 596.0939 |
| bfloat16 | large_b16_ctx8192 | flashinfer_fa2_tensor_core | 1.050496 | 0.909824 | 1.1545x [1.1539,1.2302] | 15230.900 | 17585.819 | 1.1545x [1.1539,1.2302] | 590.3703 |
| bfloat16 | large_batch_b64_ctx4096 | flashinfer_fa2_cuda_core | 2.606080 | 1.813504 | 1.4358x [1.4311,1.4404] | 24557.957 | 35290.796 | 1.4358x [1.4311,1.4404] | 592.6595 |
| bfloat16 | large_batch_b64_ctx4096 | flashinfer_fa2_tensor_core | 2.606080 | 1.763136 | 1.4789x [1.4706,1.4815] | 24557.957 | 36298.958 | 1.4789x [1.4706,1.4815] | 609.5902 |

## Absolute Tail Percentiles Across Trials

Each cell is `median [min,max]` in milliseconds across the paired trials.

| dtype | case | external backend | FlashDec p90 ms | external p90 ms | FlashDec p99 ms | external p99 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| float16 | small_b1_ctx128 | flashinfer_fa2_cuda_core | 0.049152 [0.043488,0.225536] | 0.022112 [0.015616,0.064576] | 0.069280 [0.053312,0.313152] | 0.055424 [0.053856,0.090528] |
| float16 | small_b1_ctx128 | flashinfer_fa2_tensor_core | 0.049152 [0.043488,0.225536] | 0.014464 [0.014272,0.014784] | 0.069280 [0.053312,0.313152] | 0.071840 [0.046880,0.132512] |
| float16 | medium_b16_ctx1024 | flashinfer_fa2_cuda_core | 0.181344 [0.180896,0.241792] | 0.165696 [0.162912,0.225568] | 0.219648 [0.195808,0.288736] | 0.185152 [0.184064,0.239584] |
| float16 | medium_b16_ctx1024 | flashinfer_fa2_tensor_core | 0.181344 [0.180896,0.241792] | 0.163680 [0.158944,0.175264] | 0.219648 [0.195808,0.288736] | 0.229088 [0.164992,0.233216] |
| float16 | large_b16_ctx8192 | flashinfer_fa2_cuda_core | 1.075520 [1.019264,1.078112] | 0.966240 [0.914944,0.966272] | 1.084384 [1.061312,1.108288] | 0.976192 [0.918048,0.982080] |
| float16 | large_b16_ctx8192 | flashinfer_fa2_tensor_core | 1.075520 [1.019264,1.078112] | 0.916160 [0.915200,0.968416] | 1.084384 [1.061312,1.108288] | 0.975104 [0.947424,0.977504] |
| float16 | large_batch_b64_ctx4096 | flashinfer_fa2_cuda_core | 2.574944 [2.572928,2.576672] | 1.852384 [1.794400,1.863264] | 2.600256 [2.594048,2.617632] | 1.893472 [1.812192,1.893600] |
| float16 | large_batch_b64_ctx4096 | flashinfer_fa2_tensor_core | 2.574944 [2.572928,2.576672] | 1.775424 [1.770016,1.775584] | 2.600256 [2.594048,2.617632] | 1.784512 [1.776448,1.849856] |
| bfloat16 | small_b1_ctx128 | flashinfer_fa2_cuda_core | 0.044352 [0.043072,0.046784] | 0.016768 [0.016352,0.030624] | 0.087584 [0.073728,0.088192] | 0.054464 [0.047520,0.088192] |
| bfloat16 | small_b1_ctx128 | flashinfer_fa2_tensor_core | 0.044352 [0.043072,0.046784] | 0.030720 [0.016032,0.032256] | 0.087584 [0.073728,0.088192] | 0.053120 [0.052064,0.054336] |
| bfloat16 | medium_b16_ctx1024 | flashinfer_fa2_cuda_core | 0.185856 [0.184928,0.187200] | 0.163936 [0.163712,0.165920] | 0.221600 [0.217952,0.225824] | 0.177600 [0.172768,0.180480] |
| bfloat16 | medium_b16_ctx1024 | flashinfer_fa2_tensor_core | 0.185856 [0.184928,0.187200] | 0.160096 [0.159104,0.161952] | 0.221600 [0.217952,0.225824] | 0.209792 [0.164416,0.228608] |
| bfloat16 | large_b16_ctx8192 | flashinfer_fa2_cuda_core | 1.070656 [1.061024,1.124096] | 0.913344 [0.909824,0.956544] | 1.101984 [1.078368,1.197216] | 0.981760 [0.924448,1.074400] |
| bfloat16 | large_b16_ctx8192 | flashinfer_fa2_tensor_core | 1.070656 [1.061024,1.124096] | 0.923648 [0.917824,0.968352] | 1.101984 [1.078368,1.197216] | 0.942208 [0.940704,1.559424] |
| bfloat16 | large_batch_b64_ctx4096 | flashinfer_fa2_cuda_core | 2.636608 [2.624064,2.662976] | 1.844064 [1.822240,1.881792] | 2.699616 [2.683232,2.710336] | 2.041696 [1.874848,2.325184] |
| bfloat16 | large_batch_b64_ctx4096 | flashinfer_fa2_tensor_core | 2.636608 [2.624064,2.662976] | 1.820192 [1.769120,1.828736] | 2.699616 [2.683232,2.710336] | 1.836416 [1.807904,1.977376] |

## Interpretation Boundary

- Ratios above 1 favor the named FlashInfer backend. Latency uses `FlashDec/external`; throughput uses `external/FlashDec`.
- These ratios are descriptive evidence only. R5 has no pass/fail performance or winner gate.
- Logical workload GB/s counts each Q/K/V/output element once. It excludes metadata, caching, and implementation-specific rereads, so it is a shape-normalized workload proxy rather than measured DRAM bandwidth.
- This is a common-shape, kernel-only comparison. It does not compare scheduler, KV ownership, prefix caching, multi-layer transactions, or end-to-end serving behavior.
- `fa2_cuda_core` and `fa2_tensor_core` are two execution choices from the same pinned FlashInfer installation, not separate library versions.

## Repository Acceptance

- Post-schema focused suite: `93 passed, 37 subtests passed in 5.60s`.
- Full suite: `453 passed, 94 subtests passed in 86.33s`.
- `python scripts/check_release.py --require-clean --require-evidence`: `PASS` on clean commit `d7d4feb` before this canonical summary was added to the evidence list.
- Across the eight dtype/case groups, the p50 ratio geometric mean is `1.2003x` for FA2 CUDA-core and `1.2284x` for FA2 tensor-core. All `16/16` backend/dtype/case three-trial p50 ranges are strictly above 1.
- The range is an observed three-trial minimum/maximum, not a confidence interval. Both FP16 small comparisons share the same FlashDec baseline and show large upper-end excursions, so the `2.8514x`/`4.1659x` endpoints are not treated as typical speedups.
- The absolute p99 table has overlapping three-trial ranges in `7/16` comparisons, including two tensor-core cells whose median direction favors FlashDec. With 50 repeats per row, p99 is close to a sample maximum; R5 therefore makes no stable or production tail-latency claim.
