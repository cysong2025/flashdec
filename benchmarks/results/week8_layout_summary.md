# Week 8 KV Layout Quick Summary

RTX 5070 quick validation for commit `36b223e`.

Correctness:

```text
73 passed in 9.40s
```

The validation run covered `tests/test_paged_cache.py`,
`tests/test_paged_decode.py`, and `tests/test_public_api.py`, including the
new token-major and dim-major cache paths.

Fixed benchmark configuration:

- GPU: NVIDIA GeForce RTX 5070
- PyTorch: `2.11.0+cu128`; CUDA: `12.8`
- dtype: FP16 / BF16
- `block_size=32`, `num_warps=2`
- `head_dim=128`, `num_q_heads=32`, `num_kv_heads=8`
- warmup: 5; repeat: 30

`dim/token` greater than 1 means token-major is faster.

| dtype | case | token-major p50 | dim-major p50 | dim/token p50 | observation |
| --- | --- | ---: | ---: | ---: | --- |
| FP16 | batch=1, context=1024 | 0.034112 ms | 0.029728 ms | 0.87x | dim-major faster |
| FP16 | batch=16, context=1024 | 0.125152 ms | 0.199424 ms | 1.59x | token-major faster |
| FP16 | batch=64, context=1024 | 0.484288 ms | 0.596128 ms | 1.23x | token-major faster |
| FP16 | batch=16, context=128 | 0.024736 ms | 0.030112 ms | 1.22x | token-major faster |
| FP16 | batch=16, context=4096 | 0.485280 ms | 0.652128 ms | 1.34x | token-major faster |
| BF16 | batch=1, context=1024 | 0.035744 ms | 0.028448 ms | 0.80x | dim-major faster |
| BF16 | batch=16, context=1024 | 0.143264 ms | 0.195008 ms | 1.36x | token-major faster |
| BF16 | batch=64, context=1024 | 0.494144 ms | 0.592416 ms | 1.20x | token-major faster |
| BF16 | batch=16, context=128 | 0.022720 ms | 0.030016 ms | 1.32x | token-major faster |
| BF16 | batch=16, context=4096 | 0.506592 ms | 0.654080 ms | 1.29x | token-major faster |

Summary:

- All 20 benchmark records completed reference validation (`validated=True`).
- token-major won 8/10 p50 and 8/10 p90 comparisons.
- The geometric mean of `dim-major p50 / token-major p50` is 1.20x; dim-major
  is therefore about 20% slower across this quick matrix.
- The two dim-major wins are both batch=1. They are not sufficient to justify
  a second runtime cache format or a shape dispatch path.
- Keep token-major `[num_blocks, num_kv_heads, block_size, head_dim]` as the
  default layout. Run the full sweep before treating the decision as final;
  use p50/p90 rather than mean alone because BF16 batch=64 dim-major contains
  a 9.645600 ms maximum-latency outlier.

Next command:

```bash
python benchmarks/run_layout_sweep.py --output benchmarks/results/week8_paged_decode_layout.csv
```
