# FlashDec 项目演进与里程碑

本文记录 FlashDec 从 decode attention kernel 演进为单 GPU decode runtime 原型的工程路径，并保留关键决策、交付物和验收证据。

## 1. 工程目标

FlashDec 研究 LLM decode 阶段的三个相互关联的问题：

1. 如何用可读 reference 定义 dense/paged decode attention 语义，并用 Triton 实现主要 shape。
2. 如何在动态请求生命周期中管理 Paged KV physical blocks、容量、回收与复用。
3. 如何把 RoPE/KV append、paged attention、调度和多层 token 状态组织成可验证的完整执行路径。

项目不实现完整模型、sampling、网络服务或分布式执行。所有功能必须同时具备语义说明、correctness、benchmark 边界和可复现证据。

## 2. 演进阶段

| 阶段 | 主要交付 | 验收证据 |
| --- | --- | --- |
| Foundation | package、reference、pytest/benchmark helpers、环境记录 | CPU/reference tests 与 CUDA 环境检查 |
| Dense decode | 数值稳定 PyTorch reference、Triton online-softmax kernel | shape/dtype correctness 与 latency benchmark |
| Paged decode | block table 语义、paged reference、Triton paged kernel | 变长 context、MHA/GQA/MQA、FP16/BF16 |
| Kernel optimization | `num_warps`、block size、layout、staging 受控实验 | 默认配置绑定完整 sweep 与负结果 |
| KV Runtime v2 | allocate/free/reuse、finish/cancel、capacity atomicity、metrics | request churn、block leak 与错误路径 tests |
| CUDA data path | native KV append、fused RoPE + append、PyTorch fallback | reference 对齐与 append-only benchmark |
| DecodeEngine | dynamic active batch、稳定 row mapping、backpressure | complete-step workload 与 profiler attribution |
| Scheduler R1 | lifetime commitment、FIFO + aging、stale decision 语义 | 36-row policy matrix 与 boundary-deadlock case |
| Multi-layer R2 | shared location、sequential layers、commit/rollback | 144-row正式矩阵与最终 RTX 完整回归 |
| Shared Prefix R3 | immutable full-block reuse、refcount/LRU、shared-aware admission | R3-A/R3-B WSL 回归已通过；R3-C benchmark 已实现，RTX 实测待完成 |
| Release | clean install、版本、tag、公开 release | 最终 release gate，留到项目收尾执行 |

## 3. 关键设计决策

### Reference 先于优化实现

PyTorch dense/paged reference 是所有 Triton/CUDA 路径的 correctness anchor。新实现不能只比较 shape 或运行成功，必须逐输出、状态和 allocator trajectory 对齐。

### Runtime 独占状态所有权

Scheduler 不持有 K/V tensor 或 physical blocks；kernel 不推进 request seq_len；benchmark 不修改 allocator。Paged KV Runtime 是 block ownership 与事务状态的唯一权威来源。

### 正式性能与 profiler 分离

正式 latency 来自 non-instrumented、同步后的 wall/CUDA-event 路径。Profiler 只解释 append、decode、host dispatch 与 event/launch 构成，不替代 release latency。

### 用负结果约束结论

- `num_stages=2` 的收益未达到 5% 门槛，因此保留 Triton implicit default。
- 独立 CUDA append 没有稳定优于 torch，因此默认 GPU policy 选择 fused append。
- Scheduler 的价值是容量安全与进展保证，不是所有普通 workload 下无条件更快。
- 小样本 p99 与少数 profiler attribution anomaly 保留范围，不包装成稳定收益。

## 4. 已冻结配置

- KV layout：token-major。
- Paged block size：32。
- Triton `num_warps=2`。
- Triton `num_stages=None`（implicit default）。
- GPU Engine append：`fused_cuda`。
- GPU Engine decode：`triton`。
- Scheduler：lifetime block commitment + FIFO aging。
- Multi-layer：单个 open batch transaction，按 layer 顺序执行，batch 原子 commit/abort。

## 5. 当前完成状态

- Kernel、Paged KV Runtime、DecodeEngine、Scheduler R1 和 Multi-layer R2 已完成。
- R3-A shared-prefix Cache ownership 与 R3-B Engine/scheduler integration 已通过 WSL focused/full 回归。R3-B focused 为 `56 passed, 14 subtests passed in 5.29s`，完整回归为 `352 passed, 25 subtests passed in 9.37s`。R3-C benchmark/summary 已实现，RTX 实测待完成。
- R2 正式结果绑定 commit `fa0f89a`；证据提交 `67bee15` 在 RTX 5070 完成 `337 passed, 25 subtests passed` 的无跳过回归。
- 当前仓库是 `0.0.0` release candidate。
- clean-install、版本更新与 tag 按要求留在最后 release 阶段。

## 6. 工程完成定义

每个阶段只有满足以下条件才视为完成：

1. 状态、所有权、失败原子性和范围边界有设计文档。
2. CPU/reference tests 覆盖正常与错误路径。
3. GPU 路径与 reference 对齐。
4. benchmark 固定 shape、dtype、warmup、repeat、trial、seed 和计时范围。
5. 正式结果与 profiler attribution 分离。
6. 至少保留一个负结果或未达到门槛的实验。
7. README、设计、性能和复现文档与当前实现一致。
8. 结果绑定可读 commit，并由严格 validator 检查完整性。

后续功能优先级与选择性扩展见[路线图](ROADMAP.md)，完整命令见[复现指南](reproducibility.md)。
