# FlashDec 技术参考资料

本文汇总与 FlashDec 设计直接相关的公开资料入口。资料用于核对术语、API 和系统设计；项目结论仍以仓库源码、测试和可复现实验为准。

## Triton

- Triton 中文文档：https://triton-lang.cn/main/index.html
- Triton 中文教程：https://triton-lang.cn/main/getting-started/tutorials/index.html

相关主题：program id、mask load/store、reduction、matmul、fused attention、autotune 和 debugging。

对应实现：

- `flashdec/kernels/dense_decode.py`
- `flashdec/kernels/paged_decode.py`
- `tests/test_dense_decode.py`
- `tests/test_paged_decode.py`

## PyTorch 与自定义算子

- PyTorch 中文站：https://pytorch.ac.cn/
- PyTorch 中文教程：https://docs.pytorch.ac.cn/tutorials/
- PyTorch 自定义 C++/CUDA 算子教程：https://docs.pytorch.ac.cn/tutorials/advanced/cpp_custom_ops.html

相关主题：reference tensor semantics、CUDA event、profiler、C++/CUDA extension 和 op registration。

对应实现：

- `flashdec/reference.py`
- `flashdec/paged_reference.py`
- `flashdec/csrc/`
- `flashdec/_cuda_kv_append.py`
- `flashdec/_fused_rope_kv_append.py`

## PagedAttention 与 KV Cache

- vLLM 中文文档：https://docs.vllm.com.cn/
- vLLM Paged Attention 设计：https://docs.vllm.com.cn/en/latest/design/paged_attention/

相关主题：logical/physical blocks、block table、GQA/MQA head mapping、paged KV layout 和请求生命周期。

对应设计：

- `docs/design_paged_kv.md`
- `docs/design_decode_engine.md`
- `docs/design_scheduler.md`
- `docs/design_multi_layer_kv_transaction.md`

## CUDA 与 GPU 性能

建议参考 NVIDIA 官方 CUDA Programming Guide、CUDA Best Practices Guide，以及公开的体系结构与 profiling 文档。关键主题包括：

- grid/block/thread/warp。
- global/shared/register memory。
- coalesced access、occupancy 和 register pressure。
- CUDA event 与 kernel launch overhead。
- NVCC、host compiler 和 PyTorch extension 构建。

FlashDec 的性能判断以 CUDA event、PyTorch profiler、固定 shape sweep 和显式 timing scope 为证据；没有硬件计数时不会推断具体 occupancy 或 cache hit rate。

## Attention 与 online softmax

相关主题：

- decode attention 的 Q/K/V shape。
- safe softmax 与 online softmax。
- FP32 accumulation。
- memory-bound decode 与 KV read traffic。
- 不 materialize 完整 attention matrix 的流式归约。

仓库笔记：

- `docs/notes/online_softmax.md`
- `docs/notes/gpu_memory_basics.md`
- `docs/design.md`

## 资料与实验的使用规则

1. API 和行为优先以官方文档与源码为准。
2. 第三方文章只作为设计线索，不直接作为性能结论。
3. 关键判断必须由最小测试或 benchmark 验证。
4. 性能数字必须记录硬件、版本、shape、dtype、计时范围和 commit。
5. 不同实现无法公平对齐时，明确标记为不可比。

完整实验方法见[复现指南](reproducibility.md)和[性能报告](performance_report.md)。
