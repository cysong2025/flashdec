# FlashDec 文档导航

FlashDec 的文档按“研究问题 → 系统设计 → 实验方法 → 可复现证据”组织。若只阅读一篇技术总览，请从[研究问题](research_questions.md)开始；接口使用者可直接进入[公开 API](API.md)。

## 推荐阅读顺序

1. [研究问题](research_questions.md)：六个核心问题、实现回答、正式证据与结论边界。
2. [总体设计](design.md)：attention、Paged KV、事务、调度和 kernel 的分层关系。
3. [公开 API](API.md)：张量约定、Cache、Engine、Scheduler 与 workload 接口。
4. [性能报告](performance_report.md)：默认选择、正负实验结果和不可外推边界。
5. [复现指南](reproducibility.md)：环境、correctness、benchmark 与严格摘要命令。

## 研究问题与证据

| 问题 | 主要设计 | 正式实验 |
| --- | --- | --- |
| Paged decode 的语义、地址映射与 kernel 配置如何独立验证？ | [Decode attention](design.md) · [Paged KV](design_paged_kv.md) · [Online softmax](concepts/online_softmax.md) | [Warp selection](../benchmarks/results/paged_decode_warp_selection_summary.md) · [Block size](../benchmarks/results/paged_decode_block_size_summary.md) · [KV layout](../benchmarks/results/paged_decode_kv_layout_summary.md) · [默认配置](../benchmarks/results/paged_decode_default_profile_summary.md) · [Staging 负结果](../benchmarks/results/paged_decode_staging_summary.md) |
| 动态请求下谁拥有 KV blocks、`seq_len` 与 lifecycle？ | [Paged KV](design_paged_kv.md) · [RoPE/KV append](design_rope_kv_append.md) · [DecodeEngine](design_decode_engine.md) | [Append backends](../benchmarks/results/rope_kv_append_backends_summary.md) · [Dynamic workload](../benchmarks/results/decode_engine_workload_trials3_summary.md) |
| 有限 KV 容量下如何保证 admission 安全、进展和公平？ | [Scheduler](design_scheduler.md) | [Policy matrix](../benchmarks/results/scheduler_capacity_progress_summary.md) |
| 多层 token 如何原子写入、提交或回滚？ | [Multi-layer transaction](design_multi_layer_kv_transaction.md) · [Integrated workload](design_integrated_scheduled_multi_layer.md) | [Multi-layer matrix](../benchmarks/results/multi_layer_transaction_summary.md) · [Trusted transaction](../benchmarks/results/trusted_transaction_summary.md) · [Persistent-metadata 负结果](../benchmarks/results/persistent_metadata_candidate_summary.md) · [Integrated lifecycle](../benchmarks/results/integrated_runtime_lifecycle_summary.md) |
| Shared prefix 如何改变 ownership、capacity 与 admission？ | [Shared Prefix Blocks](design_shared_prefix_blocks.md) | [8-trial confirmation](../benchmarks/results/shared_prefix_capacity_summary.md) |
| Kernel 优化能否传递到完整 step，外部基线如何公平比较？ | [Dynamic workload](design_dynamic_workload.md) · [FlashInfer baseline](design_flashinfer_baseline.md) | [Stage attribution](../benchmarks/results/decode_engine_stage_profile_summary.md) · [FlashInfer comparison](../benchmarks/results/flashinfer_paged_decode_baseline_summary.md) |

## 系统与数据路径设计

- [总体设计](design.md)
- [Paged KV Cache](design_paged_kv.md)
- [RoPE + Paged KV Append](design_rope_kv_append.md)
- [CUDA KV Append](design_cuda_kv_append.md)
- [Fused RoPE + Paged KV Append](design_fused_rope_kv_append.md)
- [DecodeEngine](design_decode_engine.md)
- [Block-aware Scheduler](design_scheduler.md)
- [Multi-layer KV Token Transaction](design_multi_layer_kv_transaction.md)
- [Shared Prefix Blocks](design_shared_prefix_blocks.md)
- [Integrated Scheduled Multi-layer Workload](design_integrated_scheduled_multi_layer.md)

## 实验、性能与复现

- [Paged Decode Kernel 实验](kernel_experiments.md)
- [性能报告](performance_report.md)
- [Dynamic Workload 方法](design_dynamic_workload.md)
- [FlashInfer 有限基线设计](design_flashinfer_baseline.md)
- [兼容性](compatibility.md)
- [复现指南](reproducibility.md)
- [Benchmark 命令](../benchmarks/README.md)
- [审核后的结果索引](../benchmarks/results/README.md)

## 概念与外部资料

- [Online Softmax 与 Decode Attention](concepts/online_softmax.md)
- [Primary references](references.md)

贡献、问题报告与安全边界分别见[贡献指南](../CONTRIBUTING.md)、[支持说明](../SUPPORT.md)和[安全政策](../SECURITY.md)。
