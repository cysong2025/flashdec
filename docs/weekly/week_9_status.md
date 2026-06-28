# Week 9 状态记录

## 本周主题

profiling 与对比。

## 本周学习目标

- 学会用 profiler 解释 kernel 为什么快或慢。
- 区分 CUDA event latency、PyTorch profiler kernel time 和 Nsight 硬件计数。
- 能从 occupancy、memory throughput、register 使用、K/V load 和 block table load 角度解释 paged decode 的瓶颈。
- 把 Week 8 的 `num_warps=2` 优化从 benchmark 结论推进到 profiler 证据。

## 本周计划

### Day 1：建立 paged decode profiling 脚本

- 新增 `benchmarks/profile_paged_decode.py`。
- 固定当前默认配置 `num_warps=2`。
- 覆盖三类代表场景：
  - small：`batch=1, context=128`。
  - medium：`batch=16, context=1024`。
  - large：`batch=16, context=8192`。
- 输出 PyTorch profiler 文本摘要和可选 Chrome trace。

### Day 2：RTX 5070 上板 profile

- 跑 small / medium / large 三类场景。
- 记录 CUDA event latency。
- 记录 profiler 中主要 CUDA kernel 时间。
- 观察是否有明显 CPU launch 或 Python wrapper 开销。

### Day 3-4：补 Nsight 方向证据

- 对 medium 和 large 场景优先跑 Nsight Compute 或 Nsight Systems。
- 重点看：
  - memory throughput。
  - achieved occupancy。
  - register usage。
  - global load efficiency。

### Day 5：整理性能报告

- 更新 `docs/performance_report.md`。
- 说明哪些优化有效、哪些实验暂时无效、下一步怎么做。

### Day 6-7：决定下一轮优化

- 如果 K/V load 是主要瓶颈，推进 KV layout 对比。
- 如果 block table load 或 mask/indexing 占比明显，推进索引路径优化。
- 如果 occupancy/register 是瓶颈，继续做 kernel config 或拆分实验。

## 当前已完成

- 新增 Week 9 profiling 脚本：
  - `benchmarks/profile_paged_decode.py`
- 新增性能报告草稿：
  - `docs/performance_report.md`
- 新增 Week 9 profiling summary 占位：
  - `benchmarks/results/week9_summary.md`
- RTX 5070 上完成 correctness 回归：
  - `tests/test_paged_decode.py tests/test_perf_metrics.py tests/test_benchmark_helpers.py`：`20 passed in 5.16s`
- RTX 5070 上完成 small profiling smoke。
- RTX 5070 上完成 small / medium / large 三场景 PyTorch profiler。

## 当前环境限制

当前 Codex macOS 环境没有 PyTorch / pytest / CUDA / Triton，因此不能在本机直接运行 CUDA profiler。

本机可完成：

```bash
python3 -m compileall flashdec tests benchmarks
```

## 需要在 RTX 5070 开发板完成

正确性回归：

```bash
pytest -vv tests/test_paged_decode.py tests/test_perf_metrics.py tests/test_benchmark_helpers.py
```

已完成：`20 passed in 5.16s`。

快速 profiling smoke：

```bash
python benchmarks/profile_paged_decode.py --case small --repeat 3 --output-dir benchmarks/profiles/week9_paged_decode_smoke
```

已完成。

完整三场景 profiling：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

已完成。

如需 Chrome trace：

```bash
python benchmarks/profile_paged_decode.py --case medium --repeat 10 --export-trace --output-dir benchmarks/profiles/week9_paged_decode_trace
```

## RTX 5070 Profiling 记录（2026-06-29）

运行环境：

- OS：Linux / WSL2。
- Python：3.12.3。
- pytest：9.1.1。
- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。
- dtype：FP16。
- `num_warps=2`。

correctness 回归：

```bash
pytest -vv tests/test_paged_decode.py tests/test_perf_metrics.py tests/test_benchmark_helpers.py
```

结果：

```text
20 passed in 5.16s
```

profiling smoke：

```bash
python benchmarks/profile_paged_decode.py --case small --repeat 3 --output-dir benchmarks/profiles/week9_paged_decode_smoke
```

结果摘要：

| case | repeat | profiler CUDA total | profiler CUDA avg/call | output |
| --- | ---: | ---: | ---: | --- |
| small_b1_ctx128 | 3 | 22.238 us | 7.413 us | `benchmarks/profiles/week9_paged_decode_smoke/small_b1_ctx128_float16_triton_w2.txt` |

完整三场景 profiling：

```bash
python benchmarks/profile_paged_decode.py --case all --repeat 10 --output-dir benchmarks/profiles/week9_paged_decode
```

结果摘要：

| case | repeat | profiler CUDA total | profiler CUDA avg/call | profiler observation |
| --- | ---: | ---: | ---: | --- |
| small_b1_ctx128 | 10 | 74.285 us | 7.428 us | kernel 本体很短，固定开销和 profiler overhead 更显眼 |
| medium_b16_ctx1024 | 10 | 1.585 ms | 158.493 us | GPU kernel 时间开始成为主要可解释对象 |
| large_b16_ctx8192 | 10 | 12.524 ms | 1.252 ms | 长 context 下 GPU kernel 时间占主导，符合 K/V 访存主导判断 |

观察：

- PyTorch profiler 中 `_paged_decode_attention_kernel` 是唯一主要 CUDA kernel，说明当前 profile 对象明确。
- small case 的 kernel avg 约 7.4 us/call，已经接近固定开销区域；这类 shape 更适合关注 launch / wrapper overhead。
- medium case 的 kernel avg 约 158 us/call，适合作为常规 decode workload 的 profiling 代表。
- large case 的 kernel avg 约 1.25 ms/call，和 Week 8 中长 context 有效带宽估算相互印证：context 变长后主要矛盾转向 K/V 读取。
- 日志中 PyTorch profiler 提示可设置 `acc_events=True`，脚本已更新以保留跨 cycle 事件，后续输出会更干净。

## 上板后要记录

- small / medium / large 的 p50、p90、mean latency。
- profiler 里 paged decode Triton kernel 的 CUDA time。
- CPU launch 和 wrapper overhead 是否明显。
- large context 下 memory-bound 判断是否和 Week 8 的有效带宽估算一致。
- 是否需要优先做 KV layout、block table indexing 或 kernel config 继续优化。

## Week 9 当前完成判定

- profiling 脚本已实现。
- profiling 文档入口已建立。
- RTX 5070 PyTorch profiler smoke 和三场景结果已补充。
- Nsight Compute / Nsight Systems 结果待补充。
