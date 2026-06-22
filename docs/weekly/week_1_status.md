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

Codex 工作区是 macOS 环境，没有 NVIDIA GPU、CUDA、PyTorch、Triton。因此这里能完成代码实现和静态检查，但不能完成真实 kernel correctness 与性能测试。

## 今日 benchmark 记录（2026-06-20）

当前仓库尚未发现实际 benchmark 结果文件：

- `benchmarks/results/` 目录下只有占位文件，尚未生成 `week1_microbench.csv`。
- 当前 Codex macOS 环境不能运行 CUDA/Triton benchmark，因此不能在本机补出真实延迟数据。
- 今日 benchmark 结果状态：待 RTX 5070 开发板运行后补充。

需要在 RTX 5070 开发板执行：

```bash
python benchmarks/run_microbench.py --op all --dtype float16 --output benchmarks/results/week1_microbench.csv
python benchmarks/run_microbench.py --op all --dtype float32 --output benchmarks/results/week1_microbench_fp32.csv
python benchmarks/run_microbench.py --op softmax --dtype float16 --rows 4096 --cols 1024 --output benchmarks/results/week1_softmax_4096x1024_fp16.csv
```

上板后把 CSV 中每个 op 的 `mean_ms`、`p50_ms`、`p90_ms`、shape、dtype、GPU 名称和 CUDA/PyTorch 版本补到本节，再总结 vector add、row softmax、RMSNorm 三个 kernel 的相对耗时与可能瓶颈。

## 需要在 RTX 5070 开发板完成

```bash
python scripts/check_env.py
pytest tests/test_triton_basics.py
python benchmarks/run_microbench.py --op all --dtype float16
```

如果这些命令通过，Week 1 的工程目标基本完成。

建议再补两组 benchmark：

```bash
python benchmarks/run_microbench.py --op all --dtype float32 --output benchmarks/results/week1_microbench_fp32.csv
python benchmarks/run_microbench.py --op softmax --dtype float16 --rows 4096 --cols 1024 --output benchmarks/results/week1_softmax_4096x1024_fp16.csv
```

## 上板后要记录

- 把 `scripts/check_env.py` 输出补到 `docs/environment.md`。
- 把 benchmark CSV 保存在 `benchmarks/results/week1_microbench.csv`。
- 在本文件补充 correctness 与性能结论。
- 如果 pytest 失败，记录失败 kernel、shape、dtype、报错信息。
- 如果 benchmark 失败，先降低 `--rows` / `--cols`，确认是资源问题还是代码问题。

## Week 1 完成判定

- `tests/test_triton_basics.py` 在 RTX 5070 上通过。
- `benchmarks/results/week1_microbench.csv` 生成。
- 你能口头解释 `docs/notes/triton_basics.md` 中的 6 个概念：
  - program id
  - block size
  - mask load/store
  - stride
  - coalesced memory access
  - reduction
