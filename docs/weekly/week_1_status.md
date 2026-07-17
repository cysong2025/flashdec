# Week 1 状态记录

## 当前已完成

- 新增 PyTorch reference：
  - `vector_add_ref`
  - `row_softmax_ref`
  - `rmsnorm_ref`
- 新增 Triton kernels：
  - `flashdec.kernels.vector_add`
  - `flashdec.kernels.softmax`
  - `flashdec.kernels.rmsnorm`
- 新增 benchmark helper：
  - CUDA event 计时。
  - p50 / p90 / mean latency。
  - CSV 输出。
- 新增 microbench 脚本：
  - `benchmarks/run_microbench.py`
- 新增 pytest：
  - `tests/test_triton_basics.py`
  - `tests/test_benchmark_helpers.py`
- 新增中文笔记：
  - `docs/notes/triton_basics.md`

## 当前环境限制

Codex 工作区是 macOS 环境，没有 NVIDIA GPU、CUDA、PyTorch、Triton。因此这里能完成代码实现和静态检查，真实 GPU correctness 与 benchmark 在 Windows + WSL2 Ubuntu 24.04 + RTX 5070 台式机上完成。

## RTX 5070 验证记录（2026-06-23）

### 环境

- OS：WSL2 Ubuntu 24.04，Linux-6.18.33.1-microsoft-standard-WSL2-x86_64。
- GPU：NVIDIA GeForce RTX 5070，11.94 GiB，sm_12.0。
- Driver：581.29，`nvidia-smi` 显示 CUDA Version 13.0。
- Python：3.12.3。
- PyTorch：2.11.0+cu128，PyTorch CUDA 12.8。
- Triton：3.6.0。
- NVCC：not found。Week 1 不阻塞，后续 CUDA extension 阶段需要补 CUDA Toolkit。

### Correctness

```bash
pytest tests/test_triton_basics.py
```

结果：24 passed in 3.68s。

### Benchmark

主 benchmark：

| op | shape | dtype | mean_ms | p50_ms | p90_ms | min_ms | max_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vector_add | size=1,000,000 | float16 | 0.019766 | 0.013088 | 0.041152 | 0.009312 | 0.114592 |
| row_softmax | 1024x1024 | float16 | 0.018436 | 0.013152 | 0.038816 | 0.009472 | 0.093600 |
| rmsnorm | 1024x1024 | float16 | 0.017897 | 0.013472 | 0.037536 | 0.010560 | 0.087040 |

补充 benchmark：

| op | shape | dtype | mean_ms | p50_ms | p90_ms | min_ms | max_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vector_add | size=1,000,000 | float32 | 0.020197 | 0.013952 | 0.036992 | 0.009664 | 0.094624 |
| row_softmax | 1024x1024 | float32 | 0.016077 | 0.013152 | 0.017312 | 0.009376 | 0.095328 |
| rmsnorm | 1024x1024 | float32 | 0.015974 | 0.013920 | 0.017600 | 0.010624 | 0.070112 |
| row_softmax | 4096x1024 | float16 | 0.033753 | 0.019328 | 0.059584 | 0.014848 | 0.077600 |

输出文件：

- `benchmarks/results/week1_microbench.csv`
- `benchmarks/results/week1_microbench_fp32.csv`
- `benchmarks/results/week1_softmax_4096x1024_fp16.csv`

### 结论

- Week 1 三个 Triton kernel 在 RTX 5070 上 correctness 全部通过。
- CUDA event benchmark helper 可以正常输出 CSV。
- 1024x1024 小 shape 下三个 kernel 的 p50 都在约 0.013 ms，说明当前测试主要覆盖 kernel 编译运行、计时链路和基础访存模式；后续 attention kernel 需要更系统的 shape sweep 与 profiling。
- `row_softmax` 从 1024 行扩展到 4096 行后，float16 p50 从 0.013152 ms 增加到 0.019328 ms，mean 从 0.018436 ms 增加到 0.033753 ms，符合 workload 增大后的趋势。

## 上板后要记录

- 已把 `scripts/check_env.py` 输出补到 `docs/environment.md`。
- 已把 benchmark CSV 保存在 `benchmarks/results/`，CSV 文件由 `.gitignore` 排除，GitHub 只提交文档总结。
- 已在本文件补充 correctness 与性能结论。

## Week 1 完成判定

- `tests/test_triton_basics.py` 在 RTX 5070 上通过：24 passed。
- `benchmarks/results/week1_microbench.csv` 生成。
- `docs/notes/triton_basics.md` 已覆盖并由对应实现验证以下概念：
  - program id
  - block size
  - mask load/store
  - stride
  - coalesced memory access
  - reduction
