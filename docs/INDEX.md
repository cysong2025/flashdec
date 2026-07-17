# FlashDec 文档索引

## 系统概览

- [项目范围与系统分层](AI_INFRA_SCOPE.md)
- [项目演进与里程碑](PROJECT_PLAN.md)
- [路线图](ROADMAP.md)
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
- [Dynamic workload](design_dynamic_workload.md)

## 性能与复现

- [性能实验记录](perf_experiments.md)
- [性能报告](performance_report.md)
- [复现指南](reproducibility.md)
- [开发与验证清单](PREP_CHECKLIST.md)
- [Benchmark 命令](../benchmarks/README.md)
- [脚本说明](../scripts/README.md)

## 技术背景

- [技术参考资料](CHINESE_RESOURCES.md)
- [Triton 基础笔记](notes/triton_basics.md)
- [GPU memory 基础](notes/gpu_memory_basics.md)
- [Online softmax](notes/online_softmax.md)

## 工程历史

- [贡献与本地验证](../CONTRIBUTING.md)
- [阶段验证日志](weekly/README.md)
- [环境记录](environment.md)

历史日志用于追溯决策和实验，不代表当前 API 或默认配置；当前状态以 README、设计文档和复现指南为准。
