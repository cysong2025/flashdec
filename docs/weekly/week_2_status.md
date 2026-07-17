# Week 2 状态记录

## 本周主题

matmul、访存、autotune、profiling 入门。

## 阶段目标

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

## RTX 5070 验证记录（2026-06-26）

### Correctness

```bash
pytest tests/test_matmul.py
```

结果：9 passed in 4.76s。

覆盖内容：

- fixed Triton FP16 matmul。
- autotuned Triton FP16 matmul。
- 规整 shape：`16,16,16`、`32,32,32`、`64,64,64`、`128,128,128`。
- 非规整 shape：`257,129,65`，用于验证 tile mask 是否正确。

### Benchmark

运行命令：

```bash
python benchmarks/run_matmul_bench.py --repeat 30 --output benchmarks/results/week2_matmul.csv
```

结果文件：

- `benchmarks/results/week2_matmul.csv`

| shape M,N,K | impl | mean_ms | p50_ms | p90_ms | speedup_vs_torch |
| --- | --- | ---: | ---: | ---: | ---: |
| 128,128,128 | torch | 0.024138 | 0.014208 | 0.050560 | 1.0000 |
| 128,128,128 | triton_fixed | 0.043434 | 0.039200 | 0.077248 | 0.5557 |
| 128,128,128 | triton_autotuned | 0.111694 | 0.117184 | 0.130592 | 0.2161 |
| 256,256,256 | torch | 0.014437 | 0.013472 | 0.016192 | 1.0000 |
| 256,256,256 | triton_fixed | 0.022393 | 0.017088 | 0.047104 | 0.6447 |
| 256,256,256 | triton_autotuned | 0.040596 | 0.041344 | 0.071712 | 0.3556 |
| 512,512,512 | torch | 0.020901 | 0.016768 | 0.021472 | 1.0000 |
| 512,512,512 | triton_fixed | 0.033837 | 0.018560 | 0.055232 | 0.6177 |
| 512,512,512 | triton_autotuned | 0.030061 | 0.016640 | 0.053696 | 0.6953 |
| 1024,1024,1024 | torch | 0.068077 | 0.047040 | 0.137376 | 1.0000 |
| 1024,1024,1024 | triton_fixed | 0.062363 | 0.052192 | 0.094112 | 1.0916 |
| 1024,1024,1024 | triton_autotuned | 0.062139 | 0.045376 | 0.071392 | 1.0956 |
| 1024,1024,256 | torch | 0.024242 | 0.019392 | 0.025664 | 1.0000 |
| 1024,1024,256 | triton_fixed | 0.030493 | 0.021728 | 0.060224 | 0.7950 |
| 1024,1024,256 | triton_autotuned | 0.050017 | 0.022496 | 0.082784 | 0.4847 |
| 4096,1024,1024 | torch | 0.178575 | 0.174080 | 0.213280 | 1.0000 |
| 4096,1024,1024 | triton_fixed | 0.240013 | 0.228960 | 0.292160 | 0.7440 |
| 4096,1024,1024 | triton_autotuned | 0.253466 | 0.267776 | 0.374016 | 0.7045 |

### Profiler

运行命令：

```bash
python benchmarks/profile_matmul.py --impl fixed --m 1024 --n 1024 --k 1024 --output benchmarks/profiles/week2_matmul_profiler.txt
python benchmarks/profile_matmul.py --impl autotuned --m 1024 --n 1024 --k 1024 --output benchmarks/profiles/week2_matmul_autotuned_profiler.txt
```

fixed profiler 摘要：

- `_matmul_kernel`：CUDA total 477.674 us，10 calls，CUDA time avg 47.767 us。
- `cuLaunchKernelEx`：CPU total 527.095 us，10 calls，CPU time avg 52.709 us。
- `aten::empty`：CPU total 724.640 us，10 calls。
- Self CPU time total：5.399 ms。
- Self CUDA time total：477.674 us。

autotuned profiler 摘要：

- `_matmul_autotuned_kernel`：CUDA total 363.707 us，10 calls，CUDA time avg 36.371 us。
- `cuLaunchKernelEx`：CPU total 104.181 us，10 calls，CPU time avg 10.418 us。
- `aten::empty`：CPU total 513.501 us，10 calls。
- Self CPU time total：3.803 ms。
- Self CUDA time total：363.707 us。

## 结论

- `torch.matmul` 背后通常是 cuBLAS，性能大概率强于本周手写 Triton matmul。
- 当前 benchmark 中，`torch.matmul` 在大多数 shape 上更快，这是预期结果。
- `1024,1024,1024` 这个规整大 shape 上，fixed 和 autotuned Triton 版本略快于 `torch.matmul`：`speedup_vs_torch` 分别为 1.0916 和 1.0956。这个结果只能说明该 shape 下可比较，不能泛化成 Triton 总是更快。
- `512,512,512` 上 autotuned 比 fixed 稍快，但仍慢于 `torch.matmul`。
- `128,128,128` 和 `256,256,256` 上 autotuned 明显慢于 fixed，说明当前 autotune 搜索空间很小，并不保证小 shape 上选到更优配置。
- profiler 中 fixed/autotuned 的 CUDA kernel 时间分别约为 47.767 us/call 和 36.371 us/call，说明 autotuned 在 `1024,1024,1024` profiler case 上 GPU execution time 更低。
- 小 shape 下 launch overhead、allocation 和尾部波动更明显，后续分析要同时看 mean、p50 和 p90。

## Week 2 完成判定

- `tests/test_matmul.py` 在 RTX 5070 上通过：9 passed。
- 生成 matmul benchmark CSV：`benchmarks/results/week2_matmul.csv`。
- 已完成 6 组 M/N/K shape sweep。
- 已完成 fixed Triton 与 autotuned Triton 的性能对比。
- 已生成 profiler 文本摘要：
  - `benchmarks/profiles/week2_matmul_profiler.txt`
  - `benchmarks/profiles/week2_matmul_autotuned_profiler.txt`
- 实验与笔记覆盖以下性能概念：
  - global memory、shared memory、register 的区别。
  - compute-bound 与 memory-bound 的区别。
  - kernel launch time 与 GPU execution time 的区别。
  - 为什么 cuBLAS 很难被手写 matmul 打败。
  - 为什么 decode attention 常常比 prefill 更偏 memory-bound。
