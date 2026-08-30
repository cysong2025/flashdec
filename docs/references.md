# FlashDec References

本页列出直接影响 FlashDec 语义、实现或实验方法的 primary references。项目结论仍以仓库中的 reference implementation、测试和可复现实验为准；外部材料用于定义术语、核对 API 和理解设计来源。

## PagedAttention 与 LLM serving memory

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023。
- [vLLM documentation](https://docs.vllm.ai/) 与 [vLLM source](https://github.com/vllm-project/vllm)。

这些资料解释 logical/physical KV blocks、block table 和 serving 中的显存碎片问题。FlashDec 在此基础上进一步把 block ownership、transaction rollback 和 Scheduler commitment 写成独立、可测试的 runtime invariants；对应文档见[Paged KV](design_paged_kv.md)和[Scheduler](design_scheduler.md)。

## Attention 与 Online Softmax

- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018。
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022。
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023。

FlashDec 的 dense/paged decode kernels 使用分块 online softmax 和 FP32 accumulation。公式与空 context 语义见[Online Softmax 与 Decode Attention](concepts/online_softmax.md)。

## Triton

- [Triton documentation](https://triton-lang.org/main/index.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton source](https://github.com/triton-lang/triton)

相关实现包括 [`dense_decode.py`](../flashdec/kernels/dense_decode.py) 和 [`paged_decode.py`](../flashdec/kernels/paged_decode.py)。项目的 layout、block size、warps 与 staging 决策来自本仓库的 shape matrix，而不是从教程配置直接外推。

## PyTorch 与自定义算子

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [`torch.utils.cpp_extension`](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)

PyTorch reference 定义数值语义；C++/CUDA extension 提供独立和 fused append 路径；profiler 仅用于阶段归因，不替代 non-instrumented latency。对应设计见[CUDA append](design_cuda_kv_append.md)、[fused append](design_fused_rope_kv_append.md)与[性能报告](performance_report.md)。

## CUDA 与性能测量

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/)

FlashDec 区分逻辑 workload GB/s、CUDA-event latency、同步 wall time 与 profiler attribution。没有硬件计数时，不推断具体 occupancy、cache hit rate 或实测 DRAM bandwidth。

## FlashInfer 外部基线

- [FlashInfer documentation](https://docs.flashinfer.ai/)
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer)

外部基线只使用固定版本 FlashInfer 的公开 `BatchDecodeWithPagedKVCacheWrapper`，并在共同 paged-decode shape、输入、page table 和 CUDA-event timing scope 下比较。详细公平性契约与不可比项见[FlashInfer baseline 设计](design_flashinfer_baseline.md)。

## vLLM out-of-tree backend 与 benchmark

- [vLLM attention backend API](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backend/)
- [vLLM backend registry](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/registry/)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)
- [vLLM source](https://github.com/vllm-project/vllm)

R7 固定 `vLLM==0.25.1`，并以该安装版本的 registry、Triton metadata 和 CLI 源码作为最终 API 依据。FlashDec plugin 只替换 eligible decode attention；模型与 serving A/B 使用同一个 vLLM runtime。设计、fallback 与性能边界见 [vLLM backend 设计](design_vllm_backend.md)。

## 引用与证据规则

1. 数学与 API 以论文、官方文档和对应版本源码为准。
2. 第三方 benchmark 数字不直接移植到 FlashDec；性能结论必须由本仓库固定输入与 timing scope 的实验产生。
3. 不同 shape、layout、硬件、版本或计时边界的结果不合并为一个 speedup。
4. 正结果和负结果使用同一预注册矩阵与 strict summarizer。
5. 所有公开数字应能追溯到 commit、设备、版本、shape、dtype、seed、trial 和正式摘要。

项目的完整证据方法见[研究问题](research_questions.md)、[性能报告](performance_report.md)和[复现指南](reproducibility.md)。
