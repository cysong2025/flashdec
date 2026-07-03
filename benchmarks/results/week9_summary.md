# Week 9 Summary

RTX 5070 PyTorch profiler run:

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

| case | dtype | num_warps | profiler_cuda_total | profiler_cuda_avg_call | profile |
| --- | --- | ---: | ---: | ---: | --- |
| small_b1_ctx128 | float16 | 2 | 74.285 us | 7.428 us | `benchmarks/profiles/week9_paged_decode/small_b1_ctx128_float16_triton_w2.txt` |
| medium_b16_ctx1024 | float16 | 2 | 1.585 ms | 158.493 us | `benchmarks/profiles/week9_paged_decode/medium_b16_ctx1024_float16_triton_w2.txt` |
| large_b16_ctx8192 | float16 | 2 | 12.524 ms | 1.252 ms | `benchmarks/profiles/week9_paged_decode/large_b16_ctx8192_float16_triton_w2.txt` |

RTX 5070 Chrome trace / CUDA event latency:

| case | dtype | num_warps | mean_ms | p50_ms | p90_ms | effective_total_gbps_p50 | profile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| medium_b16_ctx1024 | float16 | 2 | 0.203555 | 0.202880 | 0.208352 | 1059.3716 | `benchmarks/profiles/week9_paged_decode_trace/medium_b16_ctx1024_float16_triton_w2.txt` |
| large_b16_ctx8192 | float16 | 2 | 1.314675 | 1.309984 | 1.328576 | 1236.1614 | `benchmarks/profiles/week9_paged_decode_trace_large/large_b16_ctx8192_float16_triton_w2.txt` |

Large trace estimated traffic:

| metric | value |
| --- | ---: |
| `estimated_kv_read_bytes` | 1,618,067,456 |
| `estimated_total_bytes` | 1,619,351,552 |
| `profiler_cuda_avg_call` | 1.230 ms |
| `cuLaunchKernelEx_avg_call` | 8.602 us |

Notes:

- Profiler text artifacts are written under `benchmarks/profiles/` and are not committed by default.
- Treat effective bandwidth as a logical estimate; use Nsight Compute for hardware memory throughput when `ncu` is available.
- Current RTX 5070 WSL environment does not have `ncu` / `nsys`, so Nsight hardware counters are pending.
