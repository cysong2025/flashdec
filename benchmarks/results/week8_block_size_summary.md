# Week 8 Block Size Summary

RTX 5070 quick validation for commit `419e903`.

Correctness:

```text
36 passed in 6.17s
```

Fixed configuration for the block-size comparison:

- dtype: FP16 / BF16
- `head_dim=128`
- `num_q_heads=32`
- `num_kv_heads=8`
- `num_warps=2`
- repeat: 30

| dtype | case | p50 block8 | p50 block16 | p50 block32 | block32 speedup vs block16 |
| --- | --- | ---: | ---: | ---: | ---: |
| float16 | batch=1, context=1024 | 0.079680 ms | 0.060640 ms | 0.040672 ms | 1.49x |
| float16 | batch=16, context=1024 | 0.309504 ms | 0.206208 ms | 0.149248 ms | 1.38x |
| float16 | batch=64, context=1024 | 0.950624 ms | 0.594272 ms | 0.505152 ms | 1.18x |
| float16 | batch=16, context=128 | 0.041344 ms | 0.028704 ms | 0.024352 ms | 1.18x |
| float16 | batch=16, context=4096 | 1.052864 ms | 0.690592 ms | 0.489056 ms | 1.41x |
| bfloat16 | batch=1, context=1024 | 0.079232 ms | 0.062016 ms | 0.047104 ms | 1.32x |
| bfloat16 | batch=16, context=1024 | 0.302080 ms | 0.219328 ms | 0.155776 ms | 1.41x |
| bfloat16 | batch=64, context=1024 | 0.927616 ms | 0.591584 ms | 0.512128 ms | 1.16x |
| bfloat16 | batch=16, context=128 | 0.070752 ms | 0.031392 ms | 0.024224 ms | 1.30x |
| bfloat16 | batch=16, context=4096 | 1.030880 ms | 0.682368 ms | 0.501312 ms | 1.36x |

Summary:

- `block_size=32` achieved the best p50 in all 10 dtype/case combinations.
- Its p50 geometric-mean speedup was about 1.31x over block16 and 1.99x over block8.
- In the separate block-size/warp cross sweep, block32 with 2 warps won all 10 combinations.
- The repeated quick runs contained two unstable measurements, including one FP16 batch=1 block16/w2 outlier. Do not use that outlier to select 4 warps.

## Full Sweep

Configuration:

- batch sweep: 1/2/4/8/16/32/64/128 at maximum context 1024
- context sweep: 128/256/512/1024/2048/4096/8192 at batch 16
- dtype: FP16 / BF16
- `head_dim=128`, `num_q_heads=32`, `num_kv_heads=8`
- `num_warps=2`, repeat: 30

The full sweep produced 84 records and all were `validated=True`.

| metric | block16 wins | block32 wins | block32 geometric-mean speedup vs block16 |
| --- | ---: | ---: | ---: |
| p50 | 4 / 28 | 24 / 28 | 1.31x |
| p90 | 3 / 28 | 25 / 28 | 1.30x |
| mean | 2 / 28 | 26 / 28 | 1.29x |

The 4 p50 cases where block16 tied or won are all FP16 small-workload cases:

| case | p50 block16 | p50 block32 | observation |
| --- | ---: | ---: | --- |
| batch=1, context=1024 | 0.059424 ms | 0.063296 ms | block16 faster |
| batch=4, context=1024 | 0.059392 ms | 0.063264 ms | block16 faster |
| batch=16, context=256 | 0.048288 ms | 0.062560 ms | block16 faster |
| batch=16, context=512 | 0.087936 ms | 0.087936 ms | tie |

Decision:

- Use `block_size=32` and `num_warps=2` as the general benchmark/profile configuration.
- Keep block16 as a supported option for FP16 latency-critical small shapes; it is not the general default.
- `paged_decode_attention(..., block_size=None)` now infers block size from the cache shape, so existing block16 caches remain valid when callers omit the argument. The RTX 5070 regression run passed 38 tests in 4.29 seconds.
