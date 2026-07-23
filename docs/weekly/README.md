# FlashDec 阶段验证日志

本目录保存实现演进、实验输出和问题修复记录。日志是历史证据，不作为当前 API、默认配置或 release 状态的唯一来源。

- [Stage 1](week_1_status.md)：Triton 基础 kernel 与 benchmark helpers
- [Stage 2](week_2_status.md)：matmul、autotune 与 profiling 基础
- [Stage 3](week_3_status.md)：dense decode reference
- [Stage 4](week_4_status.md)：dense decode Triton kernel
- [Stage 5](week_5_status.md)：Paged KV Cache reference/runtime
- [Stage 6](week_6_status.md)：paged decode kernel v1
- [Stage 7](week_7_status.md)：真实 decode shape、GQA/MQA 与 BF16
- [Stage 8](week_8_status.md)：warps、block size 与 layout 实验
- [Stage 9](week_9_status.md)：最终 kernel profiling
- [Stage 10](week_10_status.md)：配置冻结与 KV Runtime v2
- [Stage 11](week_11_status.md)：CUDA/fused append 与 DecodeEngine
- [Stage 12](week_12_status.md)：dynamic workload 与完整 step 证据
- [Stage 13](week_13_status.md)：Scheduler R1 与 Multi-layer R2
- [Stage 14](week_14_status.md)：Shared Prefix R3
- [Stage 15](week_15_status.md)：Trusted Transaction Fast Path R4
- [Stage 16](week_16_status.md)：Integrated Scheduled Multi-layer Workload R4-C
- [Stage 17](week_17_status.md)：FlashInfer 有限公开基线 R5

当前设计与结果入口见[文档索引](../INDEX.md)。
