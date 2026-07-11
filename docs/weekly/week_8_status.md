# Week 8 状态记录

## 本周主题

优化第一轮：参数、布局、访存。

## 本周学习目标

- 学会用数据判断 kernel 默认配置，而不是凭感觉选择参数。
- 理解 `num_warps` 对 latency、occupancy、register pressure 和访存隐藏的影响。
- 学会把 latency 转成 tokens/s、有效带宽等更接近性能分析的问题。
- 能区分逻辑估算带宽和 profiler 里的真实硬件计数。
- 为 Week 9 profiling 报告准备 shape 和实验结论。

## 本周计划

### Day 1：补性能指标工具

- 新增 paged decode 字节量估算逻辑。
- 在 benchmark CSV 中记录：
  - decode tokens/s。
  - head outputs/s。
  - 估算 K/V 读取字节数。
  - 估算总字节数。
  - 有效 GB/s。

### Day 2：做 `num_warps` sweep

- 对 `num_warps=2,4,8` 做对比。
- 覆盖 batch sweep 和 context sweep。
- 观察默认 `num_warps=4` 是否仍合理。

### Day 3：整理第一版性能实验记录

- 新增 `docs/perf_experiments.md`。
- 记录实验假设、命令、指标口径和待上板结果。

### Day 4-5：RTX 5070 上板验证

- 跑 correctness 回归。
- 跑 quick benchmark。
- 跑完整 Week 8 benchmark。
- 根据 CSV 判断最优 `num_warps`。

### Day 6-7：决定下一步优化方向

- 如果长 context 有效带宽趋稳，优先做 K/V layout 或 block table 开销分析。
- 如果不同 `num_warps` 差异明显，考虑按 shape 选择默认配置。
- 如果 p90 抖动明显，准备 Week 9 profiler 分析。

## 当前已完成

- 新增性能估算模块：
  - `flashdec/perf.py`
- 新增性能指标单元测试：
  - `tests/test_perf_metrics.py`
- 新增 Week 8 benchmark 脚本：
  - `benchmarks/run_week8_paged_decode.py`
- 新增性能实验记录：
  - `docs/perf_experiments.md`
- RTX 5070 上完成性能指标 correctness：
  - `tests/test_perf_metrics.py tests/test_benchmark_helpers.py`：`6 passed in 0.02s`
- RTX 5070 上完成 paged decode correctness 回归：
  - `tests/test_paged_decode.py`：`14 passed in 3.66s`
- RTX 5070 上完成 Week 8 quick `num_warps` sweep。
- RTX 5070 上完成 Week 8 full `num_warps` sweep。
- paged decode 默认 `num_warps` 已从 4 调整为 2。
- paged decode kernel、wrapper 和 correctness 参数矩阵已扩展到 `block_size=8/16/32`。
- 新增 `benchmarks/run_block_size_sweep.py`，用于固定 `num_warps=2` 对比三种 block size。
- RTX 5070 block-size correctness 已通过：`36 passed in 6.17s`。
- RTX 5070 block-size quick sweep 与 block-size/warp 交叉 quick sweep 已完成。
- quick 结果中 block32/w2 在 10/10 个 dtype/case 组合中 p50 最优，相对 block16 p50 几何平均加速约 1.31x。

## 当前实现范围

`benchmarks/run_week8_paged_decode.py` 当前支持：

- `num_warps=2/4/8` sweep，当前默认配置为 `num_warps=2`。
- `head_dim=64/128`。
- FP16/BF16。
- `mode=triton` 或 `mode=all`。
- quick shape matrix 和完整 shape matrix。
- 输出 latency、tokens/s、估算字节数和有效 GB/s。
- 默认每个 Triton config 计时前做 reference correctness 校验。

暂不支持：

- KV cache physical layout 对比。
- Nsight / PyTorch profiler 自动摘要。
- 根据 shape 自动选择 kernel config。

## 当前环境限制

当前 Codex macOS 环境没有 PyTorch / pytest / CUDA / Triton，因此不能在本机直接运行 CUDA correctness 和 benchmark。

本机已完成：

```bash
python3 -m compileall flashdec tests benchmarks
```

结果：编译通过。

本机未完成：

```bash
python3 -m pytest -q tests/test_perf_metrics.py tests/test_benchmark_helpers.py
```

原因：当前 macOS Python 环境没有安装 `pytest`。

## Block Size 实验进展

correctness 与 quick benchmark 已完成：

```bash
pytest -vv tests/test_paged_decode.py tests/test_public_api.py
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
```

结果：

```text
36 passed in 6.17s
```

full sweep：

```bash
python benchmarks/run_block_size_sweep.py --output benchmarks/results/week8_paged_decode_block_size.csv
```

结果：84 条记录全部 `validated=True`。block32/w2 在 24/28 个 p50、25/28 个 p90、26/28 个 mean 组合中最优；p50 相对 block16 几何平均加速约 1.31x。

决定：benchmark/profile 默认改为 `block_size=32, num_warps=2`。FP16 的少数小 shape 保留 block16 作为可选对照，不单独增加自动 dispatch。

默认值调整后，API 推断 regression test 已在 RTX 5070 复跑：

```bash
python -m pytest -vv tests/test_paged_decode.py tests/test_public_api.py
```

结果：

```text
38 passed in 4.29s
```

新增测试确认：调用方省略 `block_size` 时，paged decode 可分别从 block16 与 block32 cache 正确推断 block size。

## RTX 5070 Block Size Quick 验证记录（2026-07-11）

验证提交：`419e903`。

环境：

- Python：3.12.3。
- pytest：9.1.1。
- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。

correctness：

```bash
python -m pytest -vv tests/test_paged_decode.py tests/test_public_api.py
```

结果：

```text
36 passed in 6.17s
```

quick benchmark：

```bash
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
python benchmarks/run_week8_paged_decode.py --quick --block-size 8 --num-warps 2 4 8 --output benchmarks/results/week8_block8_warps_quick.csv
python benchmarks/run_week8_paged_decode.py --quick --block-size 16 --num-warps 2 4 8 --output benchmarks/results/week8_block16_warps_quick.csv
python benchmarks/run_week8_paged_decode.py --quick --block-size 32 --num-warps 2 4 8 --output benchmarks/results/week8_block32_warps_quick.csv
```

四份 CSV 共 120 条记录，全部 `validated=True`。精简结果见 `benchmarks/results/week8_block_size_summary.md`，原始 CSV 和日志保留在本地结果目录。

full block-size CSV 额外包含 84 条 validated 记录，结论已合并到同一摘要。

## RTX 5070 验证记录（2026-06-28）

运行环境：

- OS：Linux / WSL2。
- Python：3.12.3。
- pytest：9.1.1。
- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。

性能指标与 benchmark helper correctness：

```bash
pytest -vv tests/test_perf_metrics.py tests/test_benchmark_helpers.py
```

结果：

```text
6 passed in 0.02s
```

paged decode correctness 回归：

```bash
pytest -vv tests/test_paged_decode.py
```

结果：

```text
14 passed in 3.66s
```

## RTX 5070 Quick Benchmark 记录（2026-06-28）

运行命令：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
```

输出文件：

```text
benchmarks/results/week8_paged_decode_warps_quick.csv
```

quick sweep 配置：

- dtype：FP16/BF16。
- `head_dim=128`。
- `num_q_heads=32`。
- `num_kv_heads=8`。
- `block_size=16`。
- `num_warps=2/4/8`。
- 每个 Triton config 计时前均完成 reference validation：`validated=True`。

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

- quick sweep 的所有 dtype/case 组合中，`num_warps=2` 在 p50 上都是最优。
- `batch=1, context=1024` 中 `num_warps=2` 和 `num_warps=4` 的 p50 很接近，属于小 shape 固定开销主导区间。
- 中大 shape 上 `num_warps=2` 优势非常明显：相比 `num_warps=4`，p50 通常快约 2.0x-2.5x；相比 `num_warps=8`，p50 通常快约 3.4x-4.6x。
- FP16 和 BF16 结果基本同量级，说明当前主要瓶颈不在 dtype 算力差异，而更像是 kernel 配置、访存和并行粒度。
- 有效带宽估算随 batch/context 增大明显上升，说明小 shape 受固定开销影响较大，大 shape 更接近访存主导。
- quick 结果已经提示 `num_warps=2` 是更好的候选配置；完整 sweep 进一步确认后，默认配置已调整为 `num_warps=2`。

## RTX 5070 Full Benchmark 记录（2026-06-29）

运行命令：

```bash
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
```

输出文件：

```text
benchmarks/results/week8_paged_decode_warps.csv
```

完整 sweep 配置：

- dtype：FP16/BF16。
- batch sweep：`1,2,4,8,16,32,64,128`，固定 `max_seq_len=1024`。
- context sweep：`128,256,512,2048,4096,8192`，固定 `batch=16`。
- `head_dim=128`。
- `num_q_heads=32`。
- `num_kv_heads=8`。
- `block_size=16`。
- `num_warps=2/4/8`。
- 总计 84 条 Triton benchmark 记录。

最优配置统计：

| 指标 | 结果 |
| --- | ---: |
| dtype/case 组合数 | 28 |
| `num_warps=2` p50 最优次数 | 28 |
| `num_warps=4` p50 最优次数 | 0 |
| `num_warps=8` p50 最优次数 | 0 |
| `num_warps=2` 相对 `num_warps=4` 的 p50 平均加速 | 2.10x |
| `num_warps=2` 相对 `num_warps=8` 的 p50 平均加速 | 3.75x |

batch sweep 中 `num_warps=2` 结果：

| dtype | batch | p50_ms | p90_ms | mean_ms | effective_total_gbps_p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 1 | 0.060672 | 0.061664 | 0.065063 | 206.1456 |
| float16 | 2 | 0.060064 | 0.096576 | 0.066580 | 356.9824 |
| float16 | 4 | 0.059360 | 0.088352 | 0.064257 | 692.6059 |
| float16 | 8 | 0.124992 | 0.144896 | 0.128722 | 748.4526 |
| float16 | 16 | 0.212768 | 0.218784 | 0.212847 | 1036.6383 |
| float16 | 32 | 0.371712 | 0.377504 | 0.373450 | 1017.3554 |
| float16 | 64 | 0.600288 | 0.611936 | 0.605295 | 1347.2787 |
| float16 | 128 | 1.142720 | 1.162752 | 1.146318 | 1424.7250 |
| bfloat16 | 1 | 0.061184 | 0.062880 | 0.065021 | 204.4205 |
| bfloat16 | 2 | 0.061440 | 0.101696 | 0.075761 | 348.9875 |
| bfloat16 | 4 | 0.059232 | 0.086592 | 0.065114 | 694.1026 |
| bfloat16 | 8 | 0.119968 | 0.145792 | 0.120962 | 779.7962 |
| bfloat16 | 16 | 0.213792 | 0.219008 | 0.215164 | 1031.6731 |
| bfloat16 | 32 | 0.375392 | 0.383040 | 0.375718 | 1007.3822 |
| bfloat16 | 64 | 0.603648 | 0.609216 | 0.604226 | 1339.7795 |
| bfloat16 | 128 | 1.153120 | 1.175040 | 1.190499 | 1411.8753 |

context sweep 中 `num_warps=2` 结果：

| dtype | max_seq_len | p50_ms | p90_ms | mean_ms | effective_total_gbps_p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 128 | 0.029568 | 0.059904 | 0.041477 | 873.3507 |
| float16 | 256 | 0.047584 | 0.052608 | 0.052138 | 1070.5259 |
| float16 | 512 | 0.089376 | 0.132224 | 0.105509 | 1128.8793 |
| float16 | 2048 | 0.359296 | 0.366496 | 0.361342 | 1084.7567 |
| float16 | 4096 | 0.663072 | 0.679904 | 0.667005 | 1283.5304 |
| float16 | 8192 | 1.322528 | 1.328288 | 1.321173 | 1238.3688 |
| bfloat16 | 128 | 0.029952 | 0.035104 | 0.033959 | 862.1538 |
| bfloat16 | 256 | 0.048992 | 0.075136 | 0.054050 | 1039.7596 |
| bfloat16 | 512 | 0.087840 | 0.130496 | 0.105999 | 1148.6193 |
| bfloat16 | 2048 | 0.361216 | 0.370176 | 0.362059 | 1078.9908 |
| bfloat16 | 4096 | 0.664480 | 0.685536 | 0.668438 | 1280.8107 |
| bfloat16 | 8192 | 1.322112 | 1.331520 | 1.325954 | 1238.7585 |

观察：

- 完整 sweep 进一步确认 `num_warps=2` 是当前 kernel 最优默认配置：28 个 dtype/case 组合中全部 p50 最优。
- `num_warps=2` 相比 `num_warps=4` 平均 p50 加速约 2.10x，相比 `num_warps=8` 平均 p50 加速约 3.75x。
- `context=8192` 时 FP16/BF16 p50 均约 1.322 ms，有效带宽约 1.24 TB/s，说明长 context 更接近 K/V 访存主导。
- batch 增加到 128 时，p50 约 1.14-1.15 ms，有效带宽约 1.41-1.42 TB/s，说明大 batch 能更好摊薄固定开销。
- FP16/BF16 曲线几乎重合，继续说明主要瓶颈不在 dtype 算力。
- 根据完整 sweep，已将 `flashdec.kernels.paged_decode.paged_decode_attention` 默认 `num_warps` 从 4 调整为 2。

## 需要在 RTX 5070 开发板完成

正确性回归：

```bash
pytest -vv tests/test_perf_metrics.py tests/test_benchmark_helpers.py
pytest -vv tests/test_paged_decode.py
```

已完成。

quick benchmark：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
```

已完成。

完整 benchmark：

```bash
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
```

已完成。

如需同时记录 reference 对比：

```bash
python benchmarks/run_week8_paged_decode.py --quick --mode all --output benchmarks/results/week8_paged_decode_warps_quick_with_ref.csv
```

## 上板后要记录

- 下一轮 profiling 的三类代表 shape。
- `num_warps=2` 默认配置在 profiler 中的 occupancy、memory throughput 和 register 使用。
- KV cache layout 是否值得作为下一轮优化方向。

## Week 8 当前完成判定

- 性能指标工具已实现。
- `num_warps` sweep 脚本已实现。
- 性能实验文档已建立。
- RTX 5070 quick correctness 和 quick benchmark 已完成。
- RTX 5070 完整 Week 8 benchmark 已完成。
- paged decode 默认配置已根据实测结果调整为 `num_warps=2`。
