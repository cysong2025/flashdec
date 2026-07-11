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
| KV cache layout | `[num_blocks, num_kv_heads, block_size, head_dim]` |
| block table layout | `[num_seqs, max_blocks_per_seq]` |
| block size | `8`, `16`, `32`（均已通过 RTX 5070 correctness） |
| head dim | `64`, `128` |
| dtype | `float16`, `bfloat16` |
| MHA | 支持 |
| GQA | 支持 |
| MQA | 支持 |
| 默认 `num_warps` | `2` |
| variable seq lens | 支持 |
| non-contiguous physical blocks | 支持 |
| `seq_len == 0` | 输出 zero |

当前限制：

- `block_size=8/32` 已通过 RTX 5070 correctness；quick sweep 中 32 为 p50 最优候选，full sweep 完成前默认值仍为 16。
- 暂不支持 `head_dim` 之外的 64/128。
- 暂不支持 FP32 Triton paged decode kernel。
- 已完成 `num_warps=2/4/8` 手动 sweep，但暂未做自动 autotune。
- block size quick sweep 已完成，full sweep 待完成；暂未做 layout / block size autotune。
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
```

## 后续计划

- Week 8：`num_warps` 参数实验已完成，默认配置为 2；block size correctness 与 quick sweep 已完成，待 full sweep 后决定是否将默认 block size 从 16 调整为 32。
- Week 9：补 profiling 报告和性能瓶颈分析。
