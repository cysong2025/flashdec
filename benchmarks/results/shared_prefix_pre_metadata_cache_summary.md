# Shared-prefix Pre-metadata-cache Summary

## Validation

- Input: `benchmarks/results/r3_shared_prefix_workload_trials3.csv`.
- Rows: 24; trials: 3.
- Device: NVIDIA GeForce RTX 5070.
- PyTorch/CUDA: 2.11.0+cu128 / 12.8.
- Git commit: `1d5d8d0`.
- Matrix, rotating hit-rate order, seed trajectory, capacity commitments, physical block/byte accounting, prefix lifecycle, materialized context, immutable prefix contents, and final cleanup were validated.
- Capacity admission uses a fixed bounded pool; decode latency uses a separate fixed pool large enough to keep the request batch constant.

## Cross-trial Medians

| dtype | hit rate | admitted | context physical/logical blocks | context saved | peak blocks | saved KV-capacity MiB | attach p50 us | complete p50 ms | p90 ms | p99 ms | TPS | evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 0% | 9/16 | 64/64 | 0.0% | 80 | 0.000 | 0.000 | 1.850708 | 2.384000 | 2.452116 | 8004.126 | 0 |
| float16 | 25% | 12/16 | 52/64 | 18.8% | 68 | 1.500 | 0.736 | 1.784048 | 1.904592 | 2.049256 | 8865.171 | 1 |
| float16 | 50% | 15/16 | 36/64 | 43.8% | 52 | 3.500 | 0.497 | 1.712258 | 2.323746 | 2.624034 | 8380.598 | 1 |
| float16 | 75% | 16/16 | 20/64 | 68.8% | 36 | 5.500 | 0.388 | 1.973625 | 2.393425 | 2.850560 | 7658.891 | 1 |
| bfloat16 | 0% | 9/16 | 64/64 | 0.0% | 80 | 0.000 | 0.000 | 1.595258 | 1.791891 | 1.864662 | 9687.664 | 0 |
| bfloat16 | 25% | 12/16 | 52/64 | 18.8% | 68 | 1.500 | 0.696 | 1.623265 | 1.702620 | 1.846680 | 9804.030 | 1 |
| bfloat16 | 50% | 15/16 | 36/64 | 43.8% | 52 | 3.500 | 0.427 | 1.662529 | 1.763556 | 1.950771 | 9558.612 | 1 |
| bfloat16 | 75% | 16/16 | 20/64 | 68.8% | 36 | 5.500 | 0.388 | 1.761955 | 1.961373 | 2.143019 | 8740.578 | 1 |

## Paired vs 0% Hit Rate

Ratios above 1 favor the shared-prefix case. Latency ratios are 0%/shared; TPS is shared/0%.

| dtype | hit rate | p50 median [min,max] | p90 | p99 | TPS | p50 direction |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| float16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | baseline |
| float16 | 25% | 1.0672x [1.0076,1.1174] | 1.2652x [1.0859,1.2811] | 1.0127x [1.0086,1.2518] | 1.1040x [1.0180,1.1375] | shared_faster |
| float16 | 50% | 1.0499x [0.8190,1.0928] | 0.8900x [0.7112,1.0945] | 0.7909x [0.6971,1.0227] | 0.9623x [0.7616,1.0847] | crosses_1 |
| float16 | 75% | 0.9377x [0.9298,0.9870] | 0.9961x [0.7758,1.0486] | 0.7962x [0.7579,0.8602] | 0.9569x [0.8416,0.9712] | shared_slower |
| bfloat16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | baseline |
| bfloat16 | 25% | 0.9725x [0.9370,1.0205] | 0.9987x [0.9436,1.0525] | 0.9651x [0.9220,1.0097] | 0.9639x [0.9535,1.0214] | crosses_1 |
| bfloat16 | 50% | 0.9495x [0.9091,1.0015] | 0.9993x [0.9110,1.0868] | 0.9564x [0.8248,0.9725] | 0.9432x [0.9398,1.0237] | crosses_1 |
| bfloat16 | 75% | 0.9054x [0.8602,0.9816] | 0.9136x [0.8429,0.9705] | 0.8701x [0.6960,0.8911] | 0.9022x [0.8271,0.9795] | shared_slower |

## Paired Latency Attribution vs 0%

Ratios above 1 favor the shared-prefix case. Scheduler and Engine p50 values are measured separately and are not added together.

| dtype | hit rate | scheduler p50 ratio | Engine p50 ratio | scheduler p50 ms | Engine p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 0.107920 | 1.747613 |
| float16 | 25% | 0.9949x [0.9557,1.0151] | 1.0752x [1.0043,1.0952] | 0.111593 | 1.670435 |
| float16 | 50% | 0.9406x [0.6398,0.9553] | 1.0431x [0.8007,1.1007] | 0.114734 | 1.608273 |
| float16 | 75% | 0.8716x [0.6140,0.9435] | 0.9648x [0.9377,1.0440] | 0.123815 | 1.811459 |
| bfloat16 | 0% | 1.0000x [1.0000,1.0000] | 1.0000x [1.0000,1.0000] | 0.102269 | 1.486016 |
| bfloat16 | 25% | 0.9995x [0.9166,1.0294] | 0.9780x [0.9356,1.0173] | 0.102319 | 1.515066 |
| bfloat16 | 50% | 0.9734x [0.9094,0.9916] | 0.9515x [0.9190,1.0027] | 0.105304 | 1.557225 |
| bfloat16 | 75% | 0.8958x [0.8223,0.8965] | 0.9025x [0.8669,0.9929] | 0.117288 | 1.646606 |

## Interpretation

- The primary result is physical KV reduction and higher admission under the same bounded block pool.
- Saved blocks/bytes are occupied KV-pool capacity avoided relative to private copies. The fixed-full-batch latency probe preallocates the same maximum tensor pool in every case, so this is not a direct process-VRAM measurement.
- Prefix attach is a host metadata lookup; registration copy and final eviction are reported separately.
- Decode latency keeps the same request count in every hit-rate case. Shared prefixes do not change the attention algorithm, so small latency differences should be treated as system noise unless repeated evidence is stable.
- `crosses_1` means the paired p50 direction changes across trials; do not claim a stable latency win. p99 uses few samples per trial and must be read with its full range.
- Non-instrumented synchronized wall time is the latency source; no profiler totals are mixed into release latency.
