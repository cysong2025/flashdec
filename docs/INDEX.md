# FlashDec 文档索引

## 推荐入口

- [当前交付状态](DELIVERY_STATUS.md)：R1–R5 能力、canonical evidence、限制和暂停项的唯一总览。
- [当前状态与后续目标](NEXT_STEPS.md)：当前维护范围与已暂停的 `v0.1.0` release gate。
- [Benchmark 结果索引](../benchmarks/results/README.md)：正式摘要、历史结果和本地产物边界。
- [复现指南](reproducibility.md)：已验证环境、证据命令和 release-only clean-install 流程。

仓库当前为 private `0.0.0` development candidate；R1–R5 技术交付已完成，新环境复现、版本升级、公开设置和 tag 暂停。

## 系统概览

- [公开 API](API.md)
- [项目范围与系统分层](AI_INFRA_SCOPE.md)
- [项目演进与里程碑](PROJECT_PLAN.md)
- [路线图](ROADMAP.md)
- [当前状态与后续目标](NEXT_STEPS.md)
- [兼容性矩阵](compatibility.md)
- [Changelog](../CHANGELOG.md)

## 核心设计

- [Decode attention 语义](design.md)
- [Paged KV Cache](design_paged_kv.md)
- [RoPE + KV append](design_rope_kv_append.md)
- [CUDA KV append](design_cuda_kv_append.md)
- [Fused RoPE + KV append](design_fused_rope_kv_append.md)
- [DecodeEngine](design_decode_engine.md)
- [Block-aware Scheduler](design_scheduler.md)
- [Multi-layer KV Token Transaction](design_multi_layer_kv_transaction.md)
- [Shared Prefix Blocks](design_shared_prefix_blocks.md)
- [R4-C Integrated Scheduled Multi-layer Workload](design_integrated_scheduled_multi_layer.md)
- [R5 FlashInfer 有限公开基线](design_flashinfer_baseline.md)
- [Dynamic workload](design_dynamic_workload.md)

## 性能与复现

- [性能实验记录](perf_experiments.md)
- [性能报告](performance_report.md)
- [复现指南](reproducibility.md)
- [开发与验证清单](PREP_CHECKLIST.md)
- [Benchmark 命令](../benchmarks/README.md)
- [Benchmark 结果索引](../benchmarks/results/README.md)
- [脚本说明](../scripts/README.md)

## 技术背景

- [技术参考资料](CHINESE_RESOURCES.md)
- [Triton 基础笔记](notes/triton_basics.md)
- [GPU memory 基础](notes/gpu_memory_basics.md)
- [Online softmax](notes/online_softmax.md)
- [从 PagedAttention 到可解释 Decode Runtime](notes/from_paged_attention_to_decode_runtime.md)

## 工程历史

- [贡献与本地验证](../CONTRIBUTING.md)
- [阶段验证日志](weekly/README.md)
- [环境记录](environment.md)

历史日志用于追溯决策和实验，不代表当前 API 或默认配置；当前状态以[交付状态](DELIVERY_STATUS.md)为准。
