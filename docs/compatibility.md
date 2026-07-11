# FlashDec 兼容性记录

本文记录当前 FlashDec 已支持和暂不支持的 shape / dtype / kernel 范围。结论只基于公开项目代码和个人 RTX 5070 验证结果。

## Paged KV Cache

`flashdec.cache.PagedKVCache` 当前支持：

- 固定大小 physical block。
- 每个 request 维护 logical block list。
- append 单个 decode token。
- 生成 padded `block_tables`。
- 维护 `seq_lens`。
- materialize dense KV cache 以对齐 reference。

当前限制：

- 只实现 append，不实现 request free / block reuse。
- Week 5 tests 主要验证单 layer 路径。
- 不包含 prefix cache、swap、evict、连续 batching 调度等 serving runtime 能力。

## Paged Decode Triton Kernel

`flashdec.kernels.paged_decode.paged_decode_attention` 当前支持：

| 能力 | 当前状态 |
| --- | --- |
| KV cache layout | token-major 为默认且已验证；dim-major 已通过 RTX 5070 correctness 与 quick benchmark |
| block table layout | `[num_seqs, max_blocks_per_seq]` |
| block size | `8`, `16`, `32`（均已通过 RTX 5070 correctness） |
| head dim | `64`, `128` |
| dtype | `float16`, `bfloat16` |
| MHA | 支持 |
| GQA | 支持 |
| MQA | 支持 |
| benchmark 默认 block size | `32` |
| 默认 `num_warps` | `2` |
| variable seq lens | 支持 |
| non-contiguous physical blocks | 支持 |
| `seq_len == 0` | 输出 zero |

当前限制：

- `block_size=8/32` 已通过 RTX 5070 correctness；full sweep 后 32 是通用 benchmark 默认值。FP16 的极小 batch/短 context 可单独测试 16。
- 暂不支持 `head_dim` 之外的 64/128。
- 暂不支持 FP32 Triton paged decode kernel。
- 已完成 `num_warps=2/4/8` 手动 sweep，但暂未做自动 autotune。
- block size quick/full sweep 已完成，暂未做 block size autotune。
- dim-major layout `[num_blocks, num_kv_heads, head_dim, block_size]` 已通过 RTX 5070 correctness 与 quick benchmark，但 p50 几何平均约慢 20%，不是默认 runtime layout；full layout sweep 待完成。
- 暂未和 FlashInfer / vLLM / TensorRT-LLM 做成熟库性能对比。

## 已验证 Correctness

### Week 5

`tests/test_paged_cache.py` 在 RTX 5070 上通过：

```text
6 passed in 1.68s
```

覆盖：

- Paged KV Cache append。
- 非连续 physical block。
- paged reference 与 dense reference 对齐。
- FP16 CUDA 路径。
- `seq_len == 0` 和错误输入。

### Week 6

`tests/test_paged_decode.py` 在 RTX 5070 上通过：

```text
6 passed in 3.79s
```

覆盖：

- FP16。
- `head_dim=64`。
- MHA/GQA。
- 非连续 physical block。
- `seq_len == 0`。
- 自定义 `sm_scale`。

### Week 7

`tests/test_paged_decode.py` 在 RTX 5070 上通过：

```text
14 passed in 4.48s
```

覆盖：

- `head_dim=128`。
- FP16/BF16。
- `num_q_heads=32, num_kv_heads=2` 的 GQA。
- `num_q_heads=16, num_kv_heads=1` 的 MQA。
- `seq_len == 0`、自定义 `sm_scale` 和不支持 shape 的报错路径。

### Block Size 扩展

`tests/test_paged_decode.py tests/test_public_api.py` 在 RTX 5070 上通过：

```text
36 passed in 6.17s
```

覆盖 block size 8/16/32、head dim 64/128、FP16/BF16、MHA/GQA/MQA、错误输入和公共 decode API。

### KV Layout 扩展

`tests/test_paged_cache.py tests/test_paged_decode.py tests/test_public_api.py` 在 RTX 5070 上通过：

```text
73 passed in 9.40s
```

覆盖 token-major/dim-major KV cache、两种 layout 的 block-size 推断、variable sequence lengths、non-contiguous physical blocks、MHA/GQA/MQA 与 FP16/BF16。quick benchmark 的 20 条记录全部 `validated=True`；token-major 在 8/10 个 p50、8/10 个 p90 比较中更快，因此继续作为默认 layout。

## Benchmark 路径

Week 6 默认 benchmark：

```bash
python benchmarks/run_paged_decode.py --output benchmarks/results/week6_paged_decode.csv
```

Week 7 shape sweep：

```bash
python benchmarks/run_week7_paged_decode.py --output benchmarks/results/week7_paged_decode.csv
```

快速冒烟 benchmark：

```bash
python benchmarks/run_week7_paged_decode.py --quick --mode triton --output benchmarks/results/week7_paged_decode_quick.csv
```

Week 8 `num_warps` sweep 与有效带宽估算：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
python benchmarks/run_block_size_sweep.py --output benchmarks/results/week8_paged_decode_block_size.csv
python benchmarks/run_layout_sweep.py --quick --output benchmarks/results/week8_paged_decode_layout_quick.csv
python benchmarks/run_layout_sweep.py --output benchmarks/results/week8_paged_decode_layout.csv
```

## 后续计划

- Week 8：`num_warps=2`、`block_size=32` 与 token-major 是当前通用 benchmark 默认配置；FP16 的少数小 shape 仍可单独测试 block16。下一步完成 layout full sweep，再推进 `num_stages` 或 profiler 对比。
- Week 9：补 profiling 报告和性能瓶颈分析。
