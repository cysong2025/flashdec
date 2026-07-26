# Paged Decode Default Configuration Profile Summary

RTX 5070 correctness:

```text
76 passed in 4.49s
```

Final configuration: `kv_layout=token_major`, `block_size=32`, `num_warps=2`,
warmup 5, repeat 10. All eight profiling cases completed reference validation.

Shape order: `num_seqs x num_q_heads x num_kv_heads x head_dim x max_seq_len`.

| case | shape | impl | dtype | kv_layout | block_size | num_warps | p50_ms | p90_ms | mean_ms | effective_total_gbps_p50 | device | torch | cuda | profile |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| small_b1_ctx128 | 1x32x8x128x128 | triton | float16 | token_major | 32 | 2 | 0.015328 | 0.039776 | 0.023283 | 90.8894 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/small_b1_ctx128_float16_token_major_triton_b32_w2.txt |
| medium_b16_ctx1024 | 16x32x8x128x1024 | triton | float16 | token_major | 32 | 2 | 0.155520 | 0.159488 | 0.150432 | 1325.1028 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/medium_b16_ctx1024_float16_token_major_triton_b32_w2.txt |
| large_b16_ctx8192 | 16x32x8x128x8192 | triton | float16 | token_major | 32 | 2 | 0.884576 | 0.961632 | 0.920410 | 1745.7439 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/large_b16_ctx8192_float16_token_major_triton_b32_w2.txt |
| large_batch_b64_ctx4096 | 64x32x8x128x4096 | triton | float16 | token_major | 32 | 2 | 1.934560 | 1.973344 | 1.944810 | 1605.6542 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/large_batch_b64_ctx4096_float16_token_major_triton_b32_w2.txt |
| small_b1_ctx128 | 1x32x8x128x128 | triton | bfloat16 | token_major | 32 | 2 | 0.038176 | 0.065952 | 0.042870 | 36.4929 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/small_b1_ctx128_bfloat16_token_major_triton_b32_w2.txt |
| medium_b16_ctx1024 | 16x32x8x128x1024 | triton | bfloat16 | token_major | 32 | 2 | 0.160864 | 0.166208 | 0.160701 | 1281.0822 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/medium_b16_ctx1024_bfloat16_token_major_triton_b32_w2.txt |
| large_b16_ctx8192 | 16x32x8x128x8192 | triton | bfloat16 | token_major | 32 | 2 | 0.928064 | 0.930976 | 0.931446 | 1663.9404 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/large_b16_ctx8192_bfloat16_token_major_triton_b32_w2.txt |
| large_batch_b64_ctx4096 | 64x32x8x128x4096 | triton | bfloat16 | token_major | 32 | 2 | 1.961216 | 1.976128 | 1.963923 | 1583.8309 | NVIDIA GeForce RTX 5070 | 2.11.0+cu128 | 12.8 | benchmarks/profiles/week9_final_default/large_batch_b64_ctx4096_bfloat16_token_major_triton_b32_w2.txt |

Notes:

- Profile text files live under `benchmarks/profiles/` and are intentionally not committed by default.
- Treat effective bandwidth as a logical estimate; use Nsight Compute for hardware memory throughput.
- FP16 medium and large p50 improved by 1.305x and 1.481x respectively versus the earlier block16 event baseline.
- Medium, large, and large-batch BF16 p50 are within 1.4%-4.9% of FP16, consistent with a memory-dominated workload.
- Small-case latency is below 0.1 ms and visibly noisy; do not use it alone for dtype or configuration selection.
- PyTorch profiler captured fewer than 10 kernel events in a few rows. Use CUDA-event p50/p90 as the latency source of truth and profiler tables for kernel attribution/trends.
