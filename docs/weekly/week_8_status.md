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

## 需要在 RTX 5070 开发板完成

正确性回归：

```bash
pytest -vv tests/test_perf_metrics.py tests/test_benchmark_helpers.py
pytest -vv tests/test_paged_decode.py
```

quick benchmark：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
```

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
- RTX 5070 实测结果待补充。
