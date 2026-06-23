# Week 2 状态记录

## 本周主题

matmul、访存、autotune、profiling 入门。

## 本周目标

- 建立 GPU kernel 性能直觉。
- 理解 compute-bound 与 memory-bound 的区别。
- 实现一个小型 FP16 Triton matmul kernel。
- 对比 fixed Triton、autotuned Triton 与 `torch.matmul`。
- 用 PyTorch profiler 生成一次 matmul profiling 摘要。

## 当前已完成

- 新增 PyTorch reference：
  - `matmul_ref`
- 新增 Triton kernels：
  - `flashdec.kernels.matmul.matmul`
  - `flashdec.kernels.matmul.matmul_autotuned`
- 新增 pytest：
  - `tests/test_matmul.py`
- 新增 benchmark 脚本：
  - `benchmarks/run_matmul_bench.py`
- 新增 profiler 脚本：
  - `benchmarks/profile_matmul.py`
- 新增中文笔记：
  - `docs/notes/gpu_memory_basics.md`

## 需要在 RTX 5070 台式机完成

先同步 GitHub 最新代码：

```bash
cd ~/work/flashdec
git pull
source .venv/bin/activate
```

运行 correctness：

```bash
pytest tests/test_matmul.py
```

运行 benchmark：

```bash
python benchmarks/run_matmul_bench.py --output benchmarks/results/week2_matmul.csv
```

如果完整 shape sweep 太慢，先跑小集合：

```bash
python benchmarks/run_matmul_bench.py --shape 128,128,128 --shape 256,256,256 --shape 512,512,512 --output benchmarks/results/week2_matmul_small.csv
```

运行 profiler：

```bash
python benchmarks/profile_matmul.py --impl fixed --m 1024 --n 1024 --k 1024 --output benchmarks/profiles/week2_matmul_profiler.txt
python benchmarks/profile_matmul.py --impl autotuned --m 1024 --n 1024 --k 1024 --output benchmarks/profiles/week2_matmul_autotuned_profiler.txt
```

## 上板后要记录

- `pytest tests/test_matmul.py` 是否通过。
- `benchmarks/results/week2_matmul.csv` 中每个 shape 的：
  - `torch_matmul`
  - `triton_matmul_fixed`
  - `triton_matmul_autotuned`
  - `mean_ms`
  - `p50_ms`
  - `p90_ms`
  - `speedup_vs_torch`
- profiler 中 CPU 时间和 CUDA kernel 时间的主要条目。
- fixed 配置和 autotuned 配置的性能差异。

## 预期判断

- `torch.matmul` 背后通常是 cuBLAS，性能大概率强于本周手写 Triton matmul。
- fixed Triton 版本的意义是理解 tile、mask、K 维循环和 accumulator。
- autotuned 版本的意义是观察 tile 参数如何影响性能，不要求大范围搜索。
- 如果某些大 shape 运行时间过长，先降低 repeat 或只保留小 shape sweep。

## Week 2 完成判定

- `tests/test_matmul.py` 在 RTX 5070 上通过。
- 生成 matmul benchmark CSV。
- 至少完成 5 组 M/N/K shape sweep。
- 有 fixed Triton 与 autotuned Triton 的对比。
- 有一份 profiler 文本摘要或截图说明。
- 你能解释：
  - global memory、shared memory、register 的区别。
  - compute-bound 与 memory-bound 的区别。
  - kernel launch time 与 GPU execution time 的区别。
  - 为什么 cuBLAS 很难被手写 matmul 打败。
  - 为什么 decode attention 常常比 prefill 更偏 memory-bound。
