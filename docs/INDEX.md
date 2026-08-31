# FlashDec Documentation

FlashDec 文档按“使用 → 架构 → 性能 → 深入实现”组织。首页只展示项目定位和代表性结果；完整协议、负结果与证据保留在这里和 [`benchmarks/results/`](../benchmarks/results/README.md)。

## 从这里开始

| 目标 | 文档 |
| --- | --- |
| 理解系统边界和数据流 | [Architecture](design.md) |
| 调用公开接口 | [API reference](API.md) |
| 查看性能收益和限制 | [Performance report](performance_report.md) |
| 配置 Python、PyTorch、CUDA 环境 | [Compatibility](compatibility.md) |
| 运行测试和 benchmark | [Reproducibility](reproducibility.md) |
| 查看审核后的原始结论 | [Canonical results](../benchmarks/results/README.md) |

## Runtime 与内存管理

- [Paged KV Cache](design_paged_kv.md)：physical blocks、block table、ownership 与 lifecycle。
- [DecodeEngine](design_decode_engine.md)：请求编排、状态校验和错误边界。
- [Block-aware scheduler](design_scheduler.md)：capacity commitment、公平与 progress。
- [Multi-layer token transaction](design_multi_layer_kv_transaction.md)：跨层 commit、abort 与 rollback。
- [Shared-prefix blocks](design_shared_prefix_blocks.md)：immutable full blocks、refcount 与 LRU。
- [Integrated runtime](design_integrated_scheduled_multi_layer.md)：scheduler、transaction、prefix 和 cleanup 的组合轨迹。

## Kernel 与 GPU 数据路径

- [Paged decode design](design.md)
- [Online softmax](concepts/online_softmax.md)
- [RoPE + KV append](design_rope_kv_append.md)
- [CUDA KV append](design_cuda_kv_append.md)
- [Fused RoPE + KV append](design_fused_rope_kv_append.md)
- [Kernel experiments](kernel_experiments.md)

## 外部集成与基线

- [vLLM out-of-tree backend](design_vllm_backend.md)：CUSTOM backend、eligibility、fallback 与 attestation。
- [FlashInfer baseline](design_flashinfer_baseline.md)：共同 shape、计时边界和公平比较规则。
- [Dynamic workload methodology](design_dynamic_workload.md)：kernel、完整 step 和 profiler 的不同测量范围。

## 研究与参考材料

- [Research questions](research_questions.md)：设计假设、证据链和不可外推边界。
- [Primary references](references.md)：PagedAttention、FlashAttention、Triton、vLLM 与 FlashInfer 资料。

贡献、问题报告和安全说明分别见[贡献指南](../CONTRIBUTING.md)、[支持说明](../SUPPORT.md)和[安全政策](../SECURITY.md)。
