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

- Week 8 初始 paged decode kernel 默认使用 `num_warps=4`。
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
- RTX 5070 quick sweep 与完整 sweep 均已完成。
- 完整 sweep 支持将默认 `num_warps` 从 4 调整为 2。

### Quick 结果（RTX 5070，2026-06-28）

运行命令：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
```

输出文件：

```text
benchmarks/results/week8_paged_decode_warps_quick.csv
```

结果摘要：

| dtype | case | p50 w2 | p50 w4 | p50 w8 | best | best effective_total_gbps_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| float16 | batch_b1_ctx1024 | 0.060704 | 0.061920 | 0.081824 | 2 | 206.0369 |
| float16 | batch_b16_ctx1024 | 0.206656 | 0.426976 | 0.758464 | 2 | 944.0942 |
| float16 | batch_b64_ctx1024 | 0.593024 | 1.454432 | 2.732800 | 2 | 1358.9605 |
| float16 | context_b16_ctx128 | 0.029120 | 0.058304 | 0.104928 | 2 | 892.4132 |
| float16 | context_b16_ctx4096 | 0.680000 | 1.638880 | 2.821824 | 2 | 1193.8966 |
| bfloat16 | batch_b1_ctx1024 | 0.060960 | 0.062304 | 0.082048 | 2 | 205.1717 |
| bfloat16 | batch_b16_ctx1024 | 0.208576 | 0.429088 | 0.753888 | 2 | 935.4035 |
| bfloat16 | batch_b64_ctx1024 | 0.590688 | 1.453280 | 2.711936 | 2 | 1364.3348 |
| bfloat16 | context_b16_ctx128 | 0.029184 | 0.057696 | 0.100064 | 2 | 890.4561 |
| bfloat16 | context_b16_ctx4096 | 0.681664 | 1.651424 | 2.804256 | 2 | 1190.9823 |

观察：

- quick sweep 的 10 个 dtype/case 组合中，`num_warps=2` 在 p50 上全部最优。
- 除了 `batch=1, context=1024` 这种小 batch case 外，`num_warps=2` 相比 `num_warps=4` 通常有约 2.0x-2.5x p50 优势。
- `num_warps=8` 在当前 kernel 形态下明显更慢，说明更多 warps 没有带来收益，反而可能增加调度、同步或 register pressure。
- `batch_b64_ctx1024` 和 `context_b16_ctx4096` 的有效带宽估算达到约 1.2-1.36 TB/s，说明更大 batch 或更长 context 更能摊薄固定开销。
- quick 结果已经提示默认 `num_warps=4` 可能不是最佳配置；完整 sweep 进一步确认了这一点。

### Full 结果（RTX 5070，2026-06-29）

运行命令：

```bash
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
```

输出文件：

```text
benchmarks/results/week8_paged_decode_warps.csv
```

完整 sweep 覆盖：

- dtype：FP16/BF16。
- batch sweep：`1,2,4,8,16,32,64,128`，固定 `max_seq_len=1024`。
- context sweep：`128,256,512,2048,4096,8192`，固定 `batch=16`。
- `num_warps=2/4/8`。
- 总计 84 条 Triton benchmark 记录。

最优配置统计：

| 指标 | 结果 |
| --- | ---: |
| dtype/case 组合数 | 28 |
| `num_warps=2` p50 最优次数 | 28 |
| `num_warps=4` p50 最优次数 | 0 |
| `num_warps=8` p50 最优次数 | 0 |
| `num_warps=2` 相对 `num_warps=4` 的 p50 加速范围 | 1.00x-2.99x |
| `num_warps=2` 相对 `num_warps=4` 的 p50 平均加速 | 2.10x |
| `num_warps=2` 相对 `num_warps=8` 的 p50 加速范围 | 1.33x-4.81x |
| `num_warps=2` 相对 `num_warps=8` 的 p50 平均加速 | 3.75x |

batch sweep 中 `num_warps=2` 结果：

| dtype | batch | p50_ms | p90_ms | mean_ms | total_context_tokens | effective_total_gbps_p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 1 | 0.060672 | 0.061664 | 0.065063 | 762 | 206.1456 |
| float16 | 2 | 0.060064 | 0.096576 | 0.066580 | 1306 | 356.9824 |
| float16 | 4 | 0.059360 | 0.088352 | 0.064257 | 2504 | 692.6059 |
| float16 | 8 | 0.124992 | 0.144896 | 0.128722 | 5698 | 748.4526 |
| float16 | 16 | 0.212768 | 0.218784 | 0.212847 | 13438 | 1036.6383 |
| float16 | 32 | 0.371712 | 0.377504 | 0.373450 | 23033 | 1017.3554 |
| float16 | 64 | 0.600288 | 0.611936 | 0.605295 | 49266 | 1347.2787 |
| float16 | 128 | 1.142720 | 1.162752 | 1.146318 | 99176 | 1424.7250 |
| bfloat16 | 1 | 0.061184 | 0.062880 | 0.065021 | 762 | 204.4205 |
| bfloat16 | 2 | 0.061440 | 0.101696 | 0.075761 | 1306 | 348.9875 |
| bfloat16 | 4 | 0.059232 | 0.086592 | 0.065114 | 2504 | 694.1026 |
| bfloat16 | 8 | 0.119968 | 0.145792 | 0.120962 | 5698 | 779.7962 |
| bfloat16 | 16 | 0.213792 | 0.219008 | 0.215164 | 13438 | 1031.6731 |
| bfloat16 | 32 | 0.375392 | 0.383040 | 0.375718 | 23033 | 1007.3822 |
| bfloat16 | 64 | 0.603648 | 0.609216 | 0.604226 | 49266 | 1339.7795 |
| bfloat16 | 128 | 1.153120 | 1.175040 | 1.190499 | 99176 | 1411.8753 |

context sweep 中 `num_warps=2` 结果：

| dtype | max_seq_len | p50_ms | p90_ms | mean_ms | total_context_tokens | effective_total_gbps_p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float16 | 128 | 0.029568 | 0.059904 | 0.041477 | 1559 | 873.3507 |
| float16 | 256 | 0.047584 | 0.052608 | 0.052138 | 3091 | 1070.5259 |
| float16 | 512 | 0.089376 | 0.132224 | 0.105509 | 6138 | 1128.8793 |
| float16 | 2048 | 0.359296 | 0.366496 | 0.361342 | 23757 | 1084.7567 |
| float16 | 4096 | 0.663072 | 0.679904 | 0.667005 | 51899 | 1283.5304 |
| float16 | 8192 | 1.322528 | 1.328288 | 1.321173 | 99883 | 1238.3688 |
| bfloat16 | 128 | 0.029952 | 0.035104 | 0.033959 | 1559 | 862.1538 |
| bfloat16 | 256 | 0.048992 | 0.075136 | 0.054050 | 3091 | 1039.7596 |
| bfloat16 | 512 | 0.087840 | 0.130496 | 0.105999 | 6138 | 1148.6193 |
| bfloat16 | 2048 | 0.361216 | 0.370176 | 0.362059 | 23757 | 1078.9908 |
| bfloat16 | 4096 | 0.664480 | 0.685536 | 0.668438 | 51899 | 1280.8107 |
| bfloat16 | 8192 | 1.322112 | 1.331520 | 1.325954 | 99883 | 1238.7585 |

结论：

- `num_warps=2` 在完整 sweep 中稳定胜出，可以作为当前默认配置。
- `num_warps=4` 和 `num_warps=8` 没有在任何 full sweep case 中取得 p50 最优。
- 长 context 下有效带宽稳定在约 1.1-1.3 TB/s，说明随着上下文变长，kernel 越来越接近 K/V 访存主导。
- batch 增大时有效带宽明显上升，说明较大 batch 能摊薄固定开销，并让 GPU 更充分工作。
- FP16 与 BF16 仍基本同量级，继续支持“瓶颈主要在访存和 launch/config，而不是 dtype 算力”的判断。
- 后续优化应继续围绕 K/V layout、block table 间接索引开销和 profiler 证据展开。

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
- Quick benchmark 已显示小 batch 有效带宽较低，而 batch/context 增大后有效带宽明显上升。
- 完整 context sweep 已显示 `context=8192` 下 p50 约 1.322 ms，有效带宽约 1.24 TB/s，长 context 更接近 memory-bound。

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

- profiler 脚本和 small / medium / large 三场景 RTX 5070 基线已完成，结果见 `docs/performance_report.md`。
- Nsight 硬件计数因当前 RTX 5070 WSL 环境缺少 `ncu` / `nsys` 仍待补充。

## E4：block size 8/16/32 对比

### 背景

block size 会同时影响 block table 项数、最后一个 block 的无效 token、单次循环的 K/V 读取工作量和编译期张量形状，因此不能只凭直觉选择。full sweep 前的默认值为 16；本节记录 8/16/32 的统一比较。

### 实验命令

快速 correctness 与性能冒烟：

```bash
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
```

完整 batch/context sweep：

```bash
python benchmarks/run_block_size_sweep.py --output benchmarks/results/week8_paged_decode_block_size.csv
```

默认固定：

- `block_size=8/16/32`
- `num_warps=2`
- `head_dim=128`
- `num_q_heads=32`
- `num_kv_heads=8`
- dtype：FP16/BF16

### 结果

- RTX 5070 correctness 已完成：`36 passed in 6.17s`。
- 覆盖 `block_size=8/16/32`、`head_dim=64/128`、FP16/BF16、MHA/GQA/MQA、zero seq len、自定义 scale、错误输入和公共 `flashdec.decode()` API。
- quick block-size sweep 已完成，30 条记录全部 `validated=True`。
- 固定 `num_warps=2` 时，block32 在 10 个 dtype/case 组合中 p50 全部最优。
- block32 相对 block16 的 p50 加速范围为 1.16x-1.49x，几何平均约 1.31x。
- block32 相对 block8 的 p50 加速范围为 1.68x-2.92x，几何平均约 1.99x。
- p90 中 block32 在 9/10 个组合最优；唯一例外是 FP16 `batch=16, context=128`，三个 block size 的 p90 接近且 quick run 抖动明显。
- 完整结果摘要见 `benchmarks/results/week8_block_size_summary.md`。

### Block Size × Num Warps Quick 结果

- block8：2 warps 在 10/10 个组合中 p50 最优。
- block32：2 warps 在 10/10 个组合中 p50 最优。
- block16：2 warps 在 8/10 个组合中最优；另外两个 batch=1 case 中 4 warps 表面最优，但其中 BF16 只领先约 1.7%，FP16 的 w2 出现单次 `0.172608 ms` 异常值，而独立重复运行约为 `0.060640 ms`。
- 重复 quick run 的 30 个 w2 对照中，28 个相对差异不超过 10%；两个异常点分别是 FP16 batch=1/block16 和 BF16 short-context/block8。
- 因此没有证据推翻当前 `num_warps=2` 结论，也不需要对 4/8 warps 做完整 sweep。

### Full 结果与决策

- full sweep 生成 84 条记录，全部 `validated=True`。
- block32 在 24/28 个 p50、25/28 个 p90、26/28 个 mean 组合中最优。
- block32 相对 block16 的 p50 加速范围为 0.77x-2.93x，几何平均约 1.31x；相对 block8 的 p50 几何平均约 1.95x。
- BF16 下 block32 在 14/14 个 p50 组合中获胜，几何平均约快 1.41x；FP16 下 block32 在 10 个组合中获胜、1 个持平、3 个小 shape 落后，几何平均约快 1.21x。
- block16 的例外只出现在 FP16 小工作量 case：batch=1/4、context=1024，或 batch=16、context=256/512。
- 决定：benchmark/profile 默认配置改为 `block_size=32, num_warps=2`；FP16 latency-critical small shape 仍可显式选择 block16。

### 上板后要记录

- 后续 KV layout 实验的 p50、p90、mean latency 和有效带宽。
- KV layout 与 block32 的访存/寄存器取舍。
- FP16 小 shape 是否值得单独引入 block16 dispatch。

### 下一步

- block size 实验已闭环，推进 KV layout 对比，形成第三个独立优化实验。
- 若 KV layout 对比不能稳定提升，优先用 Nsight/Profiler 验证 block table indexing、mask 与 register pressure。
