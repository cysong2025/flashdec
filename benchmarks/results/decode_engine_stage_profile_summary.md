# DecodeEngine Stage Profile Summary

Instrumented wall-clock values are not release benchmark numbers. Nested CPU/device profiler totals are attribution evidence and must not be added blindly.
Git commit: `3708b87`.

| workload | dtype | append | steps | successful | backpressure | CUDA events | ranges step/preflight/append/decode | wall p50 ms | wall p99 ms | engine CPU ms | engine device ms | append device ms | decode device ms | profile | trace |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| short_churn | float16 | torch | 120 | 120 | 0 | 20356 | 120/120/120/120 | 1.554634 | 8.283102 | 212.380779 | 8.781651 | 190.747410 | 0.359405 | benchmarks/profiles/week12_decode_engine/short_churn_float16_torch.txt | - |
| short_churn | float16 | fused_cuda | 120 | 120 | 0 | 15926 | 120/120/120/120 | 1.331134 | 1.693413 | 155.041455 | 3.036852 | 135.596248 | 0.361896 | benchmarks/profiles/week12_decode_engine/short_churn_float16_fused_cuda.txt | - |
| mixed_steady | float16 | torch | 160 | 160 | 0 | 36735 | 160/160/160/160 | 2.032006 | 4.689183 | 352.628500 | 16.419868 | 318.385744 | 1.794844 | benchmarks/profiles/week12_decode_engine/mixed_steady_float16_torch.txt | - |
| mixed_steady | float16 | fused_cuda | 160 | 160 | 0 | 28366 | 160/160/160/160 | 1.591443 | 2.134968 | 252.180253 | 5.379000 | 225.962256 | 1.807834 | benchmarks/profiles/week12_decode_engine/mixed_steady_float16_fused_cuda.txt | - |
| long_pressure | float16 | torch | 112 | 80 | 32 | 9296 | 112/112/80/80 | 1.785693 | 2.505679 | 149.627724 | 8.233909 | 136.276102 | 0.339746 | benchmarks/profiles/week12_decode_engine/long_pressure_float16_torch.txt | - |
| long_pressure | float16 | fused_cuda | 112 | 80 | 32 | 5056 | 112/112/80/80 | 1.623312 | 3.107323 | 155.146898 | 2.717375 | 140.618667 | 0.343627 | benchmarks/profiles/week12_decode_engine/long_pressure_float16_fused_cuda.txt | - |
| short_churn | bfloat16 | torch | 120 | 120 | 0 | 20356 | 120/120/120/120 | 1.429740 | 3.944754 | 174.635271 | 8.907688 | 153.687981 | 0.367061 | benchmarks/profiles/week12_decode_engine/short_churn_bfloat16_torch.txt | - |
| short_churn | bfloat16 | fused_cuda | 120 | 120 | 0 | 15926 | 120/120/120/120 | 1.342109 | 1.731949 | 153.941051 | 3.064746 | 134.433402 | 0.369740 | benchmarks/profiles/week12_decode_engine/short_churn_bfloat16_fused_cuda.txt | - |
| mixed_steady | bfloat16 | torch | 160 | 160 | 0 | 36736 | 160/160/160/160 | 2.099958 | 3.953874 | 348.594488 | 16.572458 | 313.063146 | 1.876571 | benchmarks/profiles/week12_decode_engine/mixed_steady_bfloat16_torch.txt | - |
| mixed_steady | bfloat16 | fused_cuda | 160 | 160 | 0 | 28368 | 160/160/160/160 | 1.573464 | 2.141580 | 250.452578 | 5.305550 | 225.804291 | 1.843908 | benchmarks/profiles/week12_decode_engine/mixed_steady_bfloat16_fused_cuda.txt | - |
| long_pressure | bfloat16 | torch | 112 | 80 | 32 | 9296 | 112/112/80/80 | 1.833850 | 2.312485 | 151.511422 | 8.357824 | 137.591441 | 0.350809 | benchmarks/profiles/week12_decode_engine/long_pressure_bfloat16_torch.txt | - |
| long_pressure | bfloat16 | fused_cuda | 112 | 80 | 32 | 5056 | 112/112/80/80 | 1.591408 | 2.096747 | 136.300157 | 2.733126 | 122.494083 | 0.350668 | benchmarks/profiles/week12_decode_engine/long_pressure_bfloat16_fused_cuda.txt | - |

Notes:

- `engine_step` is an inclusive range containing preflight, append, and decode.
- `rope_kv_append` includes Python allocator/metadata work plus the selected append backend.
- Matrix completeness, Git commit, CUDA event presence, and named-range counts were validated before this summary was written.
- Q/K/V generation and prompt prefill are captured by the global profiler but intentionally outside named Engine ranges.
- Final performance decisions remain based on non-instrumented multi-trial workload CSVs.
