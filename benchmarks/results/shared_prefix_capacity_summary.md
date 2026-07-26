# Shared-prefix Capacity and Admission Summary

## Validation

- Input: `benchmarks/results/r3_shared_prefix_workload_trials8.csv`.
- Rows: 64; trials: 8.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `fe72e27`.
- Matrix, rotating hit-rate order, seed trajectory, capacity commitments, physical block/byte accounting, prefix lifecycle, materialized context, immutable prefix contents, and final cleanup were validated.
- Capacity admission uses a fixed bounded pool; decode latency uses a separate fixed pool large enough to keep the request batch constant.

## Cross-trial Medians

| dtype | hit rate | admitted | context physical/logical blocks | context saved | peak blocks | saved KV-capacity MiB | attach p50 us | complete p50 ms | p90 ms | p99 ms | TPS | evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 0% | 9/16 | 64/64 | 0.0% | 80 | 0.000 | 0.000 | 1.626907 | 1.696201 | 1.761722 | 9735.364 | 0 |
| float16 | 25% | 12/16 | 52/64 | 18.8% | 68 | 1.500 | 0.841 | 1.550646 | 1.652467 | 1.746854 | 10136.293 | 1 |
| float16 | 50% | 15/16 | 36/64 | 43.8% | 52 | 3.500 | 0.447 | 1.599714 | 1.691689 | 1.808949 | 9837.246 | 1 |
| float16 | 75% | 16/16 | 20/64 | 68.8% | 36 | 5.500 | 0.397 | 1.514131 | 1.609844 | 1.695667 | 10369.913 | 1 |
| bfloat16 | 0% | 9/16 | 64/64 | 0.0% | 80 | 0.000 | 0.000 | 1.560766 | 1.620899 | 1.665162 | 10178.191 | 0 |
| bfloat16 | 25% | 12/16 | 52/64 | 18.8% | 68 | 1.500 | 0.613 | 1.546176 | 1.594053 | 1.669183 | 10429.296 | 1 |
| bfloat16 | 50% | 15/16 | 36/64 | 43.8% | 52 | 3.500 | 0.440 | 1.578692 | 1.664048 | 1.816057 | 9869.772 | 1 |
| bfloat16 | 75% | 16/16 | 20/64 | 68.8% | 36 | 5.500 | 0.385 | 1.541059 | 1.651665 | 1.706098 | 10029.082 | 1 |

## Paired vs 0% Hit Rate

Ratios above 1 favor the shared-prefix case. Latency ratios are 0%/shared; TPS is shared/0%.

| dtype | hit rate | p50 median [min,max] | p90 | p99 | TPS | p50 direction |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| float16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | baseline |
| float16 | 25% | 1.0300x [0.9295,1.1273] | 1.0518x [0.6189,1.1107] | 1.0320x [0.3447,1.1205] | 1.0279x [0.7653,1.1210] | crosses_1 |
| float16 | 50% | 1.0011x [0.8285,1.0998] | 1.0048x [0.8369,1.0841] | 1.0027x [0.5904,1.0932] | 1.0104x [0.8424,1.0804] | crosses_1 |
| float16 | 75% | 1.0454x [0.7654,1.1811] | 1.0260x [0.6426,1.1365] | 1.0180x [0.6674,1.1253] | 1.0364x [0.7360,1.1602] | crosses_1 |
| bfloat16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | baseline |
| bfloat16 | 25% | 1.0207x [0.3056,1.0427] | 1.0329x [0.3359,1.0877] | 0.9693x [0.3451,1.0501] | 1.0269x [0.3118,1.0582] | crosses_1 |
| bfloat16 | 50% | 1.0088x [0.9332,1.0534] | 0.9804x [0.8398,1.1381] | 0.8949x [0.6667,1.0894] | 0.9850x [0.9103,1.0559] | crosses_1 |
| bfloat16 | 75% | 1.0094x [0.9376,1.0556] | 0.9786x [0.9290,1.1203] | 0.9894x [0.6932,1.1478] | 0.9924x [0.9465,1.0602] | crosses_1 |

## Paired Latency Attribution vs 0%

Ratios above 1 favor the shared-prefix case. Scheduler and Engine p50 values are measured separately and are not added together.

| dtype | hit rate | scheduler p50 ratio | Engine p50 ratio | scheduler p50 ms | Engine p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 0.097742 | 1.528000 |
| float16 | 25% | 0.9747x [0.9484,1.0918] | 1.0349x [0.9388,1.1352] | 0.094883 | 1.446423 |
| float16 | 50% | 0.9817x [0.8319,1.1218] | 1.0003x [0.8273,1.0990] | 0.097264 | 1.500034 |
| float16 | 75% | 0.9961x [0.5970,1.1312] | 1.0472x [0.7829,1.1933] | 0.096033 | 1.417353 |
| bfloat16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 0.095782 | 1.464255 |
| bfloat16 | 25% | 0.9867x [0.9518,1.0238] | 1.0242x [0.2932,1.0461] | 0.097387 | 1.441619 |
| bfloat16 | 50% | 0.9853x [0.8832,1.0798] | 1.0119x [0.9182,1.0556] | 0.096531 | 1.479655 |
| bfloat16 | 75% | 0.9850x [0.9590,1.0701] | 1.0040x [0.9402,1.0546] | 0.095455 | 1.442899 |

## Interpretation

- The primary result is physical KV reduction and higher admission under the same bounded block pool.
- Saved blocks/bytes are occupied KV-pool capacity avoided relative to private copies. The fixed-full-batch latency probe preallocates the same maximum tensor pool in every case, so this is not a direct process-VRAM measurement.
- Prefix attach is a host metadata lookup; registration copy and final eviction are reported separately.
- Decode latency keeps the same request count in every hit-rate case. Shared prefixes do not change the attention algorithm, so small latency differences should be treated as system noise unless repeated evidence is stable.
- `crosses_1` means the paired p50 direction changes across trials; do not claim a stable latency win. p99 uses few samples per trial and must be read with its full range.
- Non-instrumented synchronized wall time is the latency source; no profiler totals are mixed into release latency.
