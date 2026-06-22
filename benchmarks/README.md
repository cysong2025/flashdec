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
