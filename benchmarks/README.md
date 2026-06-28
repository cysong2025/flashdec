# Benchmarks

这里放 benchmark 脚本和结果。

计划目录：

- `results/`：CSV 和 Markdown benchmark 输出。
- `profiles/`：Nsight / PyTorch profiler 输出。体积太大的 profiler 文件不要直接提交到 Git，可以只保留截图或摘要。

benchmark 记录至少包含：

- 测试日期。
- GPU 型号。
- PyTorch / Triton / CUDA 版本。
- shape。
- warmup 次数。
- 计时迭代次数。
- p50 / p90 / mean latency。
- 简单结论。

当前脚本：

- `run_microbench.py`：Week 1 小算子 benchmark。
- `run_matmul_bench.py`：Week 2 matmul shape sweep，对比 `torch.matmul`、fixed Triton 和 autotuned Triton。
- `profile_matmul.py`：Week 2 PyTorch profiler 文本摘要。
- `run_decode_reference.py`：Week 3 dense decode PyTorch reference baseline。
- `run_dense_decode.py`：Week 4 dense decode Triton kernel benchmark。
- `run_paged_decode.py`：Week 6 paged decode Triton kernel benchmark。
