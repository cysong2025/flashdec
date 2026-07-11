# Week 8 Block Size Quick Summary

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
- Keep the checked-in default at block16 until the full block-size sweep confirms the quick trend.
