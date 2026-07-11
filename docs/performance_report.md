# FlashDec 性能报告

本文记录 FlashDec paged decode attention 的性能结论、profiling 证据和后续优化方向。

## 当前结论

- Week 7 已完成真实 decode shape 覆盖：`head_dim=128`、FP16/BF16、GQA/MQA、batch/context sweep。
- Week 8 完整 `num_warps=2/4/8` sweep 显示：`num_warps=2` 在 28 个 dtype/case 组合中全部 p50 最优。
- 当前 paged decode 默认配置已调整为 `num_warps=2`。
- Week 8 有效带宽估算显示：长 context 下 kernel 更接近 K/V 访存主导。
- Week 9 PyTorch profiler 和 Chrome trace 进一步显示：large context 下总估算流量几乎全部来自 K/V 读取，kernel 时间随 context 近似线性增长。
- 当前 RTX 5070 WSL 环境缺少 `ncu` / `nsys`，Nsight 硬件计数暂未补充。

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

## PyTorch Profiler 结果

RTX 5070 上已完成 PyTorch profiler 三场景 profiling：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

| 场景 | profiler CUDA total | profiler CUDA avg/call | 观察 |
| --- | ---: | ---: | --- |
| small_b1_ctx128 | 74.285 us | 7.428 us | kernel 本体极短，固定开销和 launch overhead 更重要 |
| medium_b16_ctx1024 | 1.585 ms | 158.493 us | 常规 decode workload，适合作为后续 profiler 重点 |
| large_b16_ctx8192 | 12.524 ms | 1.252 ms | 长 context 下 kernel 时间明显上升，符合 K/V 访存主导判断 |

注意：

- 这里记录的是 PyTorch profiler 表中的 CUDA kernel time。
- CUDA event 的 p50/p90 已由脚本写入 `benchmarks/profiles/...txt` 和 `benchmarks/results/week9_summary.md`，本次粘贴日志未展开这些文件内容。
- 当前环境缺少 `ncu` / `nsys`，暂时无法补 Nsight Compute / Nsight Systems 的硬件计数，例如 memory throughput、achieved occupancy、register 使用。

## Chrome Trace 与 CUDA Event 结果

RTX 5070 上已补充 medium / large Chrome trace：

| 场景 | event mean | event p50 | event p90 | effective_total_gbps_p50 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| medium_b16_ctx1024 | 0.203555 ms | 0.202880 ms | 0.208352 ms | 1059.3716 | 常规 decode workload，适合作为 Week 10 baseline |
| large_b16_ctx8192 | 1.314675 ms | 1.309984 ms | 1.328576 ms | 1236.1614 | 长 context 下 K/V 读取占主导 |

large trace 关键估算：

| metric | value |
| --- | ---: |
| `estimated_kv_read_bytes` | 1,618,067,456 |
| `estimated_total_bytes` | 1,619,351,552 |
| `profiler CUDA avg/call` | 1.230 ms |
| `cuLaunchKernelEx avg/call` | 8.602 us |

解释：

- large 场景中 `estimated_kv_read_bytes` 与 `estimated_total_bytes` 几乎相同，说明总数据量主要来自 K/V cache 读取。
- context 从 `1024` 增到 `8192` 后，kernel 时间也接近按比例增长，支持 memory-bound 判断。
- large 场景下 launch overhead 相比 kernel 本体很小，优先优化方向应放在 KV layout、block size 和索引/访存路径，而不是 Python wrapper。

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

## Profiling 初步判断

- small shape：kernel 本体只有约 7.4 us/call，固定开销可能比算子本体更值得关注。
- medium shape：kernel 本体约 158 us/call，可作为后续 Nsight 分析的主力代表场景。
- large shape：kernel 本体约 1.25 ms/call，随 context 变长明显增长，继续支持 memory-bound 判断。
- PyTorch profiler 已经能确认主要 CUDA 时间集中在 `_paged_decode_attention_kernel`；但它还不能替代 Nsight 的硬件计数。
- 因当前环境没有 `ncu` / `nsys`，Week 10 先使用 CUDA event、PyTorch profiler 和逻辑带宽估算推进优化实验；后续环境具备时再补硬件计数。

## 下一步优化候选

1. KV cache layout 对比：
   - 当前：`[num_blocks, num_kv_heads, block_size, head_dim]`
   - 候选：`[num_blocks, num_kv_heads, head_dim, block_size]`
2. block size 对比：
   - `8/16/32` 已通过 RTX 5070 correctness：`36 passed in 6.17s`。
   - quick sweep 中 block32 在 10/10 个 dtype/case 组合中 p50 最优，相对 block16 p50 几何平均加速约 1.31x。
   - full sweep 尚未完成，当前仓库默认值仍为 16。
3. profiler 指导下的 indexing 优化：
   - 减少 block table load。
   - 减少 mask 和 offset 计算。
4. 与 dense Triton baseline 做部分 shape 对比。
5. 条件允许时，与 FlashInfer 或 vLLM 公开实现做有限对比。
