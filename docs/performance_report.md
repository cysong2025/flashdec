# FlashDec 性能报告

本文记录 FlashDec paged decode attention 的性能结论、profiling 证据和后续优化方向。

## 当前结论

- Week 7 已完成真实 decode shape 覆盖：`head_dim=128`、FP16/BF16、GQA/MQA、batch/context sweep。
- Week 8 完整 `num_warps=2/4/8` sweep 显示：`num_warps=2` 在 28 个 dtype/case 组合中全部 p50 最优。
- 当前 paged decode 默认配置已调整为 `num_warps=2`。
- Week 8 有效带宽估算显示：长 context 下 kernel 更接近 K/V 访存主导。

## Profiling 计划

代表场景：

| 场景 | shape | 目的 |
| --- | --- | --- |
| small | `batch=1, context=128` | 观察固定开销和 launch overhead |
| medium | `batch=16, context=1024` | 观察常规 decode workload |
| large | `batch=16, context=8192` | 观察长 context 访存瓶颈 |

PyTorch profiler 命令：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

Chrome trace 命令：

```bash
python benchmarks/profile_paged_decode.py --case medium --repeat 10 --export-trace --output-dir benchmarks/profiles/week9_paged_decode_trace
```

## 待补充结果

上板后需要补充：

| 场景 | p50_ms | p90_ms | profiler CUDA time | 观察 |
| --- | ---: | ---: | ---: | --- |
| small | 待补充 | 待补充 | 待补充 | 待补充 |
| medium | 待补充 | 待补充 | 待补充 | 待补充 |
| large | 待补充 | 待补充 | 待补充 | 待补充 |

## 已验证有效优化

### `num_warps=4 -> 2`

证据：

- Week 8 full sweep 中，`num_warps=2` 在 28 个 dtype/case 组合中全部 p50 最优。
- `num_warps=2` 相比 `num_warps=4` 平均 p50 加速约 2.10x。
- `num_warps=2` 相比 `num_warps=8` 平均 p50 加速约 3.75x。

解释：

- 当前 kernel 每个 program 处理一个 `(sequence, q_head)`。
- 对当前 `block_size=16`、`head_dim=128` 的实现，更多 warps 没有带来足够收益，反而可能增加调度、同步或 register pressure。
- 较少 warps 更适合当前工作粒度。

## 需要验证的瓶颈

- K/V cache global memory load 是否占主导。
- block table 间接索引在长 context 下是否可见。
- `seq_len`、mask 和 logical block loop 的控制开销是否影响短 context。
- 当前 register 使用是否限制 occupancy。

## 下一步优化候选

1. KV cache layout 对比：
   - 当前：`[num_blocks, num_kv_heads, block_size, head_dim]`
   - 候选：`[num_blocks, num_kv_heads, head_dim, block_size]`
2. block size 对比：
   - 当前仅支持 `block_size=16`
   - 后续实验 `8/16/32`
3. profiler 指导下的 indexing 优化：
   - 减少 block table load。
   - 减少 mask 和 offset 计算。
4. 与 dense Triton baseline 做部分 shape 对比。
5. 条件允许时，与 FlashInfer 或 vLLM 公开实现做有限对比。
