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

Profiler text artifacts are written under `benchmarks/profiles/` and are not committed by default.
