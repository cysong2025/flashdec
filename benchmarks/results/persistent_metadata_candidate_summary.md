# Persistent Metadata Candidate Summary

## Validation

- Input artifact: `r4_persistent_transaction_metadata_trials5.csv` (local source path normalized for publication; raw CSV is not tracked).
- Rows: 160; paired trials: 80.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `8047a9c`.
- Materialized and persistent paths used identical trusted fused CUDA/Triton math; only Cache-owned transaction metadata lifetime differed.
- Matrix, rotating path order, seeds/inputs, exact parity, block/transaction/Engine trajectory, rollback, metadata build/reuse/release, CPU profiler ranges, and invariants were validated.
- `views/token` counts Cache transaction-view materializations only; it does not count unrelated public result tensors elsewhere in the system.
- Profiler captures: 160; extra retries: 0; maximum attempts per row: 1.
- Complete-token latency is pure synchronized wall time with no CUDA events; the separate profiler is CPU-only.

Ratios above 1 favor persistent metadata. Latency and append-CPU ratios are materialized/persistent; TPS is persistent/materialized.

## Cross-trial Cases

| dtype | case | p50 median [min,max] | p90 [min,max] | p99 [min,max] | TPS [min,max] | append CPU [min,max] | views mat/pers | reuses mat/pers | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bfloat16 | l2_b16_c1024 | 1.1481x [1.0574,1.2169] | 1.0211x [0.9460,1.2135] | 1.0981x [0.7724,1.1909] | 1.0900x [1.0325,1.2050] | 2.5628x [2.2876,3.2960] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| bfloat16 | l2_b16_c128 | 1.3663x [1.2944,1.4281] | 1.2455x [1.2169,1.5995] | 1.2954x [0.8743,1.9201] | 1.3039x [1.2495,1.5021] | 2.5394x [1.4798,4.1515] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| bfloat16 | l2_b4_c1024 | 1.1580x [0.6274,1.2970] | 0.9121x [0.2806,1.2569] | 0.9282x [0.4144,1.4616] | 1.0665x [0.4767,1.3147] | 3.2826x [2.8825,4.0136] | 6.0/1.0 | 0.0/2.0 | unstable_crosses_1 |
| bfloat16 | l2_b4_c128 | 1.3482x [1.2852,1.7790] | 1.3879x [1.1753,5.6979] | 1.3547x [1.1836,4.7344] | 1.3526x [1.2705,2.8134] | 2.8234x [2.2795,3.1215] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| bfloat16 | l4_b16_c1024 | 1.2236x [1.2103,1.3780] | 1.2759x [1.2351,1.3643] | 1.3850x [1.0444,1.5039] | 1.2491x [1.1974,1.3647] | 2.7769x [2.4530,3.1070] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| bfloat16 | l4_b16_c128 | 1.3759x [1.0187,1.5209] | 1.2771x [0.9962,1.7857] | 1.2254x [0.9533,1.7389] | 1.3274x [1.0453,1.5313] | 3.3161x [2.0677,3.8265] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| bfloat16 | l4_b4_c1024 | 1.2366x [1.1502,1.3243] | 1.3323x [0.9575,1.6672] | 1.4870x [1.0612,1.6919] | 1.2054x [1.1622,1.3983] | 3.3997x [2.8981,3.5514] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| bfloat16 | l4_b4_c128 | 1.4474x [1.1136,1.5203] | 1.4289x [0.8010,1.6340] | 1.3065x [0.8207,1.6357] | 1.4102x [1.0770,1.5468] | 3.5897x [3.0611,3.6993] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| float16 | l2_b16_c1024 | 1.1623x [1.1072,1.2742] | 1.1618x [0.9797,1.4895] | 1.1123x [0.1792,1.4622] | 1.1709x [0.8503,1.3569] | 2.7803x [2.5505,3.3465] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| float16 | l2_b16_c128 | 1.2564x [0.9729,1.3634] | 1.2474x [0.8021,1.5466] | 1.1407x [0.7832,1.4831] | 1.2912x [0.9489,1.3371] | 2.2078x [2.1079,3.0909] | 6.0/1.0 | 0.0/2.0 | unstable_crosses_1 |
| float16 | l2_b4_c1024 | 1.2313x [1.1257,1.3207] | 1.2788x [1.1211,1.6668] | 1.8174x [0.9818,12.4233] | 1.2731x [1.1434,2.1280] | 2.6694x [2.2726,3.6194] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| float16 | l2_b4_c128 | 1.2765x [1.0396,1.4114] | 1.2000x [1.1074,1.3591] | 1.3048x [0.8628,1.3607] | 1.2473x [1.0531,1.3209] | 3.3106x [2.6348,3.6009] | 6.0/1.0 | 0.0/2.0 | persistent_faster |
| float16 | l4_b16_c1024 | 1.2169x [1.0142,1.2253] | 1.2143x [0.5742,1.2625] | 1.1531x [0.1494,1.2803] | 1.1964x [0.6524,1.2305] | 3.0905x [2.6066,3.4899] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| float16 | l4_b16_c128 | 1.3710x [0.9499,1.4811] | 1.2323x [0.5100,1.5209] | 0.9941x [0.1655,1.4432] | 1.2873x [0.6301,1.4497] | 3.7792x [3.0040,5.1712] | 10.0/1.0 | 0.0/4.0 | unstable_crosses_1 |
| float16 | l4_b4_c1024 | 1.2285x [1.1295,1.3657] | 1.1669x [1.0184,1.6822] | 1.2099x [0.8893,2.1031] | 1.2274x [1.1241,1.5006] | 3.5826x [2.1565,3.8435] | 10.0/1.0 | 0.0/4.0 | persistent_faster |
| float16 | l4_b4_c128 | 1.4108x [1.2970,1.4400] | 1.3688x [1.3416,1.6488] | 1.4421x [1.2863,2.3447] | 1.4184x [1.3439,1.5420] | 3.3934x [3.1576,6.2037] | 10.0/1.0 | 0.0/4.0 | persistent_faster |

## Absolute Metadata Medians

| dtype | case | path | token p50 ms | append CPU ms/layer | builds/token | views/token | reuses/token | releases/token | resident after |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | l2_b16_c1024 | materialized | 0.891061 | 0.135164 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l2_b16_c1024 | persistent | 0.775588 | 0.050392 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| bfloat16 | l2_b16_c128 | materialized | 0.709592 | 0.140982 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l2_b16_c128 | persistent | 0.513955 | 0.055517 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| bfloat16 | l2_b4_c1024 | materialized | 0.704234 | 0.139958 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l2_b4_c1024 | persistent | 0.571156 | 0.043974 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| bfloat16 | l2_b4_c128 | materialized | 0.663663 | 0.135528 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l2_b4_c128 | persistent | 0.508786 | 0.049006 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| bfloat16 | l4_b16_c1024 | materialized | 1.250674 | 0.124696 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l4_b16_c1024 | persistent | 0.999091 | 0.044905 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| bfloat16 | l4_b16_c128 | materialized | 1.010336 | 0.132577 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l4_b16_c128 | persistent | 0.737017 | 0.039514 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| bfloat16 | l4_b4_c1024 | materialized | 1.080148 | 0.132305 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l4_b4_c1024 | persistent | 0.909466 | 0.038110 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| bfloat16 | l4_b4_c128 | materialized | 0.919238 | 0.125780 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| bfloat16 | l4_b4_c128 | persistent | 0.641743 | 0.035608 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| float16 | l2_b16_c1024 | materialized | 0.844877 | 0.130529 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| float16 | l2_b16_c1024 | persistent | 0.714233 | 0.043808 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| float16 | l2_b16_c128 | materialized | 0.687764 | 0.128125 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| float16 | l2_b16_c128 | persistent | 0.553511 | 0.057006 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| float16 | l2_b4_c1024 | materialized | 0.693224 | 0.130645 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| float16 | l2_b4_c1024 | persistent | 0.552143 | 0.050541 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| float16 | l2_b4_c128 | materialized | 0.694025 | 0.140559 | 1.0 | 6.0 | 0.0 | 1.0 | 0.0 |
| float16 | l2_b4_c128 | persistent | 0.546335 | 0.043073 | 1.0 | 1.0 | 2.0 | 1.0 | 0.0 |
| float16 | l4_b16_c1024 | materialized | 1.225832 | 0.130029 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| float16 | l4_b16_c1024 | persistent | 1.035718 | 0.044829 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| float16 | l4_b16_c128 | materialized | 0.991151 | 0.137665 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| float16 | l4_b16_c128 | persistent | 0.722933 | 0.037882 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| float16 | l4_b4_c1024 | materialized | 1.088996 | 0.133505 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| float16 | l4_b4_c1024 | persistent | 0.911402 | 0.036748 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |
| float16 | l4_b4_c128 | materialized | 0.984314 | 0.138718 | 1.0 | 10.0 | 0.0 | 1.0 | 0.0 |
| float16 | l4_b4_c128 | persistent | 0.709294 | 0.038791 | 1.0 | 1.0 | 4.0 | 1.0 | 0.0 |

## Overall Geometric Mean

| metric | persistent vs materialized |
| --- | ---: |
| complete-token p50 | 1.2493x |
| complete-token p90 | 1.2236x |
| complete-token p99 | 1.1863x |
| complete-token mean | 1.2392x |
| decode tokens/s | 1.2392x |
| profiler append CPU/layer | 3.0366x |

## Pre-registered Decision Rule

The validator verifies evidence integrity; the pre-registered rule determines whether the candidate replaces the default path.

- Required: overall p50 >= 1.05x and all 16 dtype/case groups have paired p50 min > 1.0x.
- Observed: overall p50 1.2493x; groups above 1 in every trial 13/16.
- Observed-matrix screening: `fail`.
- Matrix coverage: 16/16 groups (`complete`).
- Keep decision: `not adopted`.

## Interpretation

- `unstable_crosses_1` means the paired p50 direction changes across trials; do not claim a stable win.
- Persistent metadata must build exactly once, reuse once per layer, release once, and leave zero resident bundles per token. Materialized recreates the legacy `2L+2` view boundary only inside this benchmark.
- Cache transaction-view counts are not presented as a count of every Engine result tensor allocation.
- Both paths must report zero `aten::item` and `_local_scalar_dense`; CPU profiler totals explain host overhead but are not release latency.
- p50, p90, and p99 are reported with their full paired ranges; ratios below 1 are negative results.
