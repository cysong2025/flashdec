# FlashDec 性能实验记录

本文记录 FlashDec paged decode attention 的性能实验。目标不是只保存 benchmark 数字，而是把每次优化的假设、测量方法、结果和结论连起来。

## 指标口径

基础指标：

- `mean_ms` / `p50_ms` / `p90_ms`：CUDA event 计时得到的 latency。
- `speedup_vs_ref`：PyTorch paged reference 的平均耗时除以 Triton kernel 的平均耗时。
- `decode_tokens_per_s_mean`：按 batch 中每个 sequence 生成一个 token 估算的吞吐。
- `head_outputs_per_s_mean`：按 `(sequence, q_head)` 输出数量估算的吞吐。

估算访存指标：

- `estimated_kv_read_bytes`：按当前 kernel 模型估算的 K/V 读取字节数。
- `estimated_total_bytes`：估算的 Q 读取、K/V 读取、block table 读取、seq_lens 读取和输出写回总字节数。
- `effective_kv_gbps_mean` / `effective_total_gbps_mean`：用估算字节数除以平均 latency 得到的有效带宽。
- `effective_kv_gbps_p50` / `effective_total_gbps_p50`：用估算字节数除以 p50 latency 得到的有效带宽。

注意：这些带宽是逻辑估算值，用来比较不同 shape 和 kernel config 的趋势；真实硬件 memory transaction、cache 命中、replay 和 occupancy 需要通过 profiler 验证。

## E1：num_warps 参数实验

假设：

- 当前 paged decode kernel 默认使用 `num_warps=4`。
- 对短 context、小 batch 来说，更少 warps 可能降低调度和同步开销。
- 对长 context、大 batch 来说，更多 warps 可能提升并行度，但也可能增加 register pressure 或降低 occupancy。

实验脚本：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
```

可选加入 reference 对比：

```bash
python benchmarks/run_week8_paged_decode.py --quick --mode all --output benchmarks/results/week8_paged_decode_warps_quick_with_ref.csv
```

默认 sweep：

- `num_warps=2,4,8`
- dtype：FP16/BF16
- `head_dim=128`
- `num_q_heads=32`
- `num_kv_heads=8`
- `block_size=16`

需要记录：

- 每个 shape 的最优 `num_warps`。
- `num_warps=4` 是否仍适合作为默认值。
- 长 context 下有效带宽是否随 `num_warps` 明显变化。
- 是否存在小 shape 中 p90 明显抖动的情况。

当前状态：

- 脚本已实现。
- RTX 5070 实测结果待补充。

## E2：长 context 访存瓶颈分析

假设：

- 第七周完整 benchmark 已显示 context 从 128 增加到 8192 时，Triton latency 明显上升，`speedup_vs_ref` 下降到约 20x。
- 这说明长 context 下 K/V cache 读取量开始主导 latency，kernel 更接近 memory-bound。

实验方法：

- 使用 Week 8 脚本输出的 `estimated_kv_read_bytes`、`estimated_total_bytes` 和有效 GB/s。
- 重点比较 `context=128,1024,4096,8192`。
- 对 FP16 和 BF16 分别观察趋势。

需要回答：

- latency 是否近似随 `total_context_tokens` 线性增长。
- 有效带宽在长 context 下是否趋于稳定。
- 小 context 的速度是否主要受 launch overhead 和固定开销影响。

当前状态：

- 指标估算逻辑已实现。
- RTX 5070 实测结果待补充。

## E3：profiling 准备

Week 8 后半段或 Week 9 需要对三类场景做 profiling：

- 小 batch / 短 context：例如 `batch=1, context=128`。
- 中 batch / 中 context：例如 `batch=16, context=1024`。
- 大 batch / 长 context：例如 `batch=64, context=4096` 或 `batch=16, context=8192`。

优先记录：

- CUDA kernel time。
- memory throughput。
- achieved occupancy。
- register usage。
- block table load 和 K/V load 是否造成明显瓶颈。

当前状态：

- profiler 脚本待补。
- 先用 Week 8 CSV 指标决定 profiling 的重点 shape。
