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

## 当前实现范围

`benchmarks/run_week8_paged_decode.py` 当前支持：

- `num_warps=2/4/8` sweep。
- `head_dim=64/128`。
- FP16/BF16。
- `mode=triton` 或 `mode=all`。
- quick shape matrix 和完整 shape matrix。
- 输出 latency、tokens/s、估算字节数和有效 GB/s。
- 默认每个 Triton config 计时前做 reference correctness 校验。

暂不支持：

- `block_size=8/32` 的实际 kernel benchmark。
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
- `num_warps=2` 已成为候选默认配置，但仍应先跑完整 sweep，再决定是否修改 kernel wrapper 的默认 `num_warps`。

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

如需同时记录 reference 对比：

```bash
python benchmarks/run_week8_paged_decode.py --quick --mode all --output benchmarks/results/week8_paged_decode_warps_quick_with_ref.csv
```

## 上板后要记录

- 每个 shape 下 `num_warps=2/4/8` 的 p50/p90/mean。
- 每个 shape 的最优 `num_warps`。
- `effective_total_gbps_p50` 随 context 增长的变化。
- FP16 和 BF16 的 latency 是否仍基本同量级。
- 是否需要把默认 `num_warps` 从 4 改成按 shape 选择。

## Week 8 当前完成判定

- 性能指标工具已实现。
- `num_warps` sweep 脚本已实现。
- 性能实验文档已建立。
- RTX 5070 quick correctness 和 quick benchmark 已完成。
- 完整 Week 8 benchmark 待补充。
