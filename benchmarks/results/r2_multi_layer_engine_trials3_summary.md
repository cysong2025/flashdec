# Multi-layer DecodeEngine Trial Summary

## Validation

- Input: `benchmarks/results/r2_multi_layer_engine_trials3.csv`.
- Rows: 144; paired trials: 72.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `fa0f89a`.
- Matrix, pair trajectory, block accounting, transaction counts, profiler ranges, rollback evidence, seed, and backend order were validated.
- Non-instrumented wall latency is the performance source; profiler fields are attribution-only.

Ratios above 1 favor fused CUDA. Latency ratios are torch/fused; throughput is fused/torch; CUDA-event ratio means fewer events for fused.

## Cross-trial Cases

| dtype | case | p50 median [min,max] | p90 | p99 [min,max] | TPS | append device | decode device | CUDA events | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bfloat16 | l1_b16_c1024 | 1.0816x [0.3479,1.4103] | 1.1829x | 1.0694x [0.4270,1.9258] | 1.0951x | 1.8452x | 1.0285x | 1.8286x | unstable_crosses_1 |
| bfloat16 | l1_b16_c128 | 1.4254x [1.1456,1.4979] | 1.0948x | 1.0912x [0.4537,3.9316] | 1.1370x | 2.0064x | 0.9971x | 1.8169x | fused_faster |
| bfloat16 | l1_b4_c1024 | 1.0673x [1.0465,1.4468] | 1.1180x | 1.5637x [1.2470,1.5768] | 1.1215x | 1.6645x | 0.9984x | 1.7234x | fused_faster |
| bfloat16 | l1_b4_c128 | 1.1166x [1.0245,1.1561] | 1.0542x | 1.1085x [1.0346,1.2955] | 1.0869x | 1.4973x | 0.9943x | 1.7234x | fused_faster |
| bfloat16 | l2_b16_c1024 | 1.2426x [1.1617,1.3402] | 1.5521x | 1.6089x [1.1732,5.1095] | 1.3694x | 1.5609x | 1.0121x | 2.1897x | fused_faster |
| bfloat16 | l2_b16_c128 | 1.2098x [1.1652,1.4130] | 1.1583x | 1.1191x [1.0023,1.2439] | 1.1766x | 1.5892x | 1.0068x | 2.1777x | fused_faster |
| bfloat16 | l2_b4_c1024 | 1.2400x [1.2021,1.4864] | 1.3635x | 1.9059x [1.2699,2.9663] | 1.3688x | 1.4243x | 1.0130x | 1.9128x | fused_faster |
| bfloat16 | l2_b4_c128 | 1.1325x [0.9966,1.3393] | 1.1264x | 1.0162x [0.2461,4.5236] | 1.1141x | 1.2920x | 0.9979x | 1.9128x | unstable_crosses_1 |
| bfloat16 | l4_b16_c1024 | 1.1988x [1.1632,1.5374] | 1.3570x | 1.8040x [0.4056,4.1503] | 1.3964x | 1.4039x | 1.0182x | 2.5246x | fused_faster |
| bfloat16 | l4_b16_c128 | 1.2494x [1.2322,1.4769] | 1.6070x | 2.6008x [1.0247,3.5500] | 1.3656x | 1.5706x | 1.0062x | 2.5114x | fused_faster |
| bfloat16 | l4_b4_c1024 | 1.4071x [1.1078,1.4493] | 1.5568x | 3.7026x [1.5632,5.9934] | 1.3723x | 1.4693x | 1.0047x | 2.0462x | fused_faster |
| bfloat16 | l4_b4_c128 | 1.1093x [1.0768,1.1393] | 1.2077x | 4.0696x [1.2302,5.1965] | 1.3502x | 0.4980x | 0.9952x | 2.0382x | fused_faster |
| float16 | l1_b16_c1024 | 1.2502x [1.0521,1.7465] | 1.2640x | 1.3797x [0.3998,4.1901] | 1.2675x | 1.8287x | 0.9908x | 1.7532x | fused_faster |
| float16 | l1_b16_c128 | 1.0721x [0.9834,1.1452] | 0.9798x | 1.0873x [0.4415,2.2387] | 1.0041x | 1.4976x | 1.0061x | 1.7436x | unstable_crosses_1 |
| float16 | l1_b4_c1024 | 1.0849x [0.8807,1.3102] | 1.1576x | 0.8927x [0.7725,1.1754] | 1.0542x | 1.7829x | 1.0011x | 1.6296x | unstable_crosses_1 |
| float16 | l1_b4_c128 | 1.0621x [1.0221,1.2003] | 1.0248x | 0.8956x [0.7622,6.0321] | 1.0294x | 1.5236x | 1.0136x | 1.6296x | fused_faster |
| float16 | l2_b16_c1024 | 1.2226x [1.1985,1.2571] | 1.8040x | 1.4301x [1.0995,5.0031] | 1.3331x | 1.7401x | 0.9852x | 2.1048x | fused_faster |
| float16 | l2_b16_c128 | 1.2508x [1.2381,1.5865] | 1.2922x | 1.3253x [1.1306,4.9819] | 1.2273x | 1.8007x | 0.9967x | 2.0943x | fused_faster |
| float16 | l2_b4_c1024 | 1.3269x [1.2370,1.5581] | 1.3702x | 1.3114x [0.9461,1.3908] | 1.2871x | 1.4273x | 1.0060x | 1.8293x | fused_faster |
| float16 | l2_b4_c128 | 1.1323x [1.0212,1.3934] | 1.1054x | 1.2149x [1.1072,1.4167] | 1.1099x | 1.4194x | 1.0012x | 1.8293x | fused_faster |
| float16 | l4_b16_c1024 | 1.2723x [1.1583,1.2737] | 1.2127x | 1.2425x [0.8023,1.4342] | 1.2616x | 1.7687x | 1.0122x | 2.4531x | fused_faster |
| float16 | l4_b16_c128 | 1.4418x [1.1725,1.4947] | 2.0280x | 3.8154x [1.2016,4.7224] | 1.7528x | 1.7344x | 1.0074x | 2.4410x | fused_faster |
| float16 | l4_b4_c1024 | 1.0754x [1.0629,1.7125] | 1.4726x | 3.5805x [1.2414,4.6387] | 1.4193x | 1.5333x | 1.0110x | 1.9891x | fused_faster |
| float16 | l4_b4_c128 | 1.1536x [1.1366,1.6657] | 1.1618x | 1.1311x [0.3101,4.1213] | 1.1294x | 1.2897x | 0.9870x | 1.9819x | fused_faster |

## Absolute Attribution Medians

| dtype | case | backend | token p50 ms | append device ms/layer | decode device ms/layer | CUDA events | begin host ms | commit host ms | rollback p50 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | l1_b16_c1024 | torch | 2.647822 | 1.638167 | 0.143361 | 256 | 0.903853 | 0.318427 | N/A |
| bfloat16 | l1_b16_c1024 | fused_cuda | 2.317847 | 0.942326 | 0.140524 | 140 | 0.928276 | 0.342258 | N/A |
| bfloat16 | l1_b16_c128 | torch | 2.530640 | 1.835216 | 0.017256 | 258 | 0.928084 | 0.223336 | N/A |
| bfloat16 | l1_b16_c128 | fused_cuda | 1.803488 | 0.811693 | 0.017306 | 142 | 0.931392 | 0.226540 | N/A |
| bfloat16 | l1_b4_c1024 | torch | 1.691182 | 1.423603 | 0.052230 | 162 | 0.346133 | 0.174677 | N/A |
| bfloat16 | l1_b4_c1024 | fused_cuda | 1.385232 | 0.824742 | 0.052211 | 94 | 0.339144 | 0.170790 | N/A |
| bfloat16 | l1_b4_c128 | torch | 1.829129 | 1.371098 | 0.008783 | 162 | 0.420035 | 0.215830 | N/A |
| bfloat16 | l1_b4_c128 | fused_cuda | 1.638176 | 0.915732 | 0.008814 | 94 | 0.420299 | 0.223755 | N/A |
| bfloat16 | l2_b16_c1024 | torch | 4.234573 | 1.626427 | 0.135289 | 427 | 0.908928 | 0.314803 | 2.491417 |
| bfloat16 | l2_b16_c1024 | fused_cuda | 3.407861 | 0.986897 | 0.133703 | 195 | 0.938224 | 0.324188 | 2.052024 |
| bfloat16 | l2_b16_c128 | torch | 3.881750 | 1.499588 | 0.017405 | 429 | 0.884595 | 0.213142 | 2.532329 |
| bfloat16 | l2_b16_c128 | fused_cuda | 3.208560 | 0.941917 | 0.017269 | 197 | 0.952280 | 0.227180 | 2.250387 |
| bfloat16 | l2_b4_c1024 | torch | 2.800123 | 1.161116 | 0.055334 | 285 | 0.336053 | 0.176160 | 1.882840 |
| bfloat16 | l2_b4_c1024 | fused_cuda | 2.089912 | 0.765647 | 0.054624 | 149 | 0.315075 | 0.165215 | 1.531274 |
| bfloat16 | l2_b4_c128 | torch | 2.993358 | 1.200647 | 0.008738 | 285 | 0.431335 | 0.217246 | 1.727663 |
| bfloat16 | l2_b4_c128 | fused_cuda | 2.643065 | 0.972060 | 0.008762 | 149 | 0.412935 | 0.217265 | 1.615752 |
| bfloat16 | l4_b16_c1024 | torch | 6.907581 | 1.439296 | 0.133593 | 767 | 0.922551 | 0.310246 | 2.418356 |
| bfloat16 | l4_b16_c1024 | fused_cuda | 5.493991 | 1.010677 | 0.156700 | 304 | 0.934994 | 0.322993 | 2.122200 |
| bfloat16 | l4_b16_c128 | torch | 6.693664 | 1.476704 | 0.017367 | 771 | 1.098936 | 0.237831 | 2.653139 |
| bfloat16 | l4_b16_c128 | fused_cuda | 5.338969 | 0.926597 | 0.017312 | 307 | 0.941079 | 0.214051 | 2.138132 |
| bfloat16 | l4_b4_c1024 | torch | 5.934697 | 1.179486 | 0.094759 | 532 | 0.501876 | 0.293252 | 1.893444 |
| bfloat16 | l4_b4_c1024 | fused_cuda | 4.518170 | 0.867165 | 0.094230 | 260 | 0.395312 | 0.241338 | 1.433606 |
| bfloat16 | l4_b4_c128 | torch | 5.190779 | 1.167496 | 0.008782 | 534 | 0.478688 | 0.232023 | 2.225669 |
| bfloat16 | l4_b4_c128 | fused_cuda | 4.679171 | 2.522828 | 0.008830 | 262 | 0.424202 | 0.207684 | 1.778864 |
| float16 | l1_b16_c1024 | torch | 2.472908 | 1.647393 | 0.133458 | 271 | 0.776795 | 0.262501 | N/A |
| float16 | l1_b16_c1024 | fused_cuda | 1.832943 | 0.904723 | 0.134695 | 155 | 0.713288 | 0.257680 | N/A |
| float16 | l1_b16_c128 | torch | 2.411204 | 1.490962 | 0.016766 | 272 | 0.903208 | 0.208004 | N/A |
| float16 | l1_b16_c128 | fused_cuda | 2.249040 | 0.984567 | 0.016664 | 156 | 1.022541 | 0.256048 | N/A |
| float16 | l1_b4_c1024 | torch | 1.466843 | 1.347582 | 0.048457 | 176 | 0.336693 | 0.167501 | N/A |
| float16 | l1_b4_c1024 | fused_cuda | 1.341874 | 0.786667 | 0.048586 | 108 | 0.335615 | 0.178086 | N/A |
| float16 | l1_b4_c128 | torch | 1.843876 | 1.487988 | 0.008342 | 176 | 0.390919 | 0.206143 | N/A |
| float16 | l1_b4_c128 | fused_cuda | 1.655886 | 0.976639 | 0.008230 | 108 | 0.414346 | 0.221265 | N/A |
| float16 | l2_b16_c1024 | torch | 4.020984 | 1.467313 | 0.127595 | 442 | 1.270815 | 0.326028 | 2.721670 |
| float16 | l2_b16_c1024 | fused_cuda | 3.302158 | 0.925668 | 0.129510 | 210 | 0.902599 | 0.330991 | 2.214933 |
| float16 | l2_b16_c128 | torch | 4.081228 | 1.716083 | 0.016586 | 444 | 0.917576 | 0.215999 | 2.841451 |
| float16 | l2_b16_c128 | fused_cuda | 3.198584 | 0.984973 | 0.016655 | 212 | 0.955313 | 0.206795 | 2.220765 |
| float16 | l2_b4_c1024 | torch | 2.935468 | 1.105709 | 0.050889 | 300 | 0.336816 | 0.171406 | 1.799432 |
| float16 | l2_b4_c1024 | fused_cuda | 2.180498 | 0.774665 | 0.050584 | 164 | 0.338071 | 0.168447 | 1.365374 |
| float16 | l2_b4_c128 | torch | 3.006327 | 1.193933 | 0.008350 | 300 | 0.406883 | 0.217835 | 1.843146 |
| float16 | l2_b4_c128 | fused_cuda | 2.655089 | 0.881496 | 0.008338 | 164 | 0.439021 | 0.195159 | 1.669806 |
| float16 | l4_b16_c1024 | torch | 7.034962 | 1.691405 | 0.126736 | 782 | 0.932360 | 0.325115 | 2.956635 |
| float16 | l4_b16_c1024 | fused_cuda | 5.652709 | 0.967114 | 0.125549 | 319 | 0.948108 | 0.314093 | 2.139671 |
| float16 | l4_b16_c128 | torch | 7.318289 | 1.638926 | 0.016833 | 786 | 1.092876 | 0.255036 | 2.567874 |
| float16 | l4_b16_c128 | fused_cuda | 5.134892 | 0.944959 | 0.016678 | 322 | 0.962261 | 0.216911 | 2.090678 |
| float16 | l4_b4_c1024 | torch | 5.253638 | 1.140408 | 0.085423 | 547 | 0.472648 | 0.253971 | 1.819436 |
| float16 | l4_b4_c1024 | fused_cuda | 4.834501 | 0.786095 | 0.084596 | 275 | 0.436884 | 0.245571 | 1.211589 |
| float16 | l4_b4_c128 | torch | 5.867642 | 1.128066 | 0.008294 | 549 | 0.437313 | 0.213243 | 1.652758 |
| float16 | l4_b4_c128 | fused_cuda | 4.705573 | 0.890961 | 0.008402 | 277 | 0.425455 | 0.212170 | 1.541031 |

## Overall Geometric Mean

| metric | fused vs torch |
| --- | ---: |
| complete-token p50 | 1.2101x |
| complete-token p90 | 1.3826x |
| complete-token p99 | 1.5407x |
| complete-token mean | 1.2800x |
| total CUDA p50 | 1.2117x |
| per-layer CUDA p50 | 1.3154x |
| decode tokens/s | 1.2800x |
| profiler append device/layer | 1.6103x |
| profiler decode device/layer | 1.0024x |
| profiler CUDA events | 1.9784x |

## Interpretation

- `unstable_crosses_1` means p50 crosses 1 across trials; do not claim a stable backend win.
- A ratio below 1 means fused is worse for that metric; inspect the absolute attribution table before explaining why.
- p99 must be reported with its range.
- Profiler device totals and event counts explain launch/stage behavior but are not release latency.
- Rollback latency remains an error-path metric and is not mixed into normal throughput.
