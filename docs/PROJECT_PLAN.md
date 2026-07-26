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
| Shared Prefix R3 | immutable full-block reuse、refcount/LRU、shared-aware admission | R3-D correctness 与 8-trial/64-row RTX confirmation 已完成 |
| Trusted Transaction R4 | checked public raw op、Cache-owned device-value-check-free path、persistent metadata、统一多层调度证据 | R4-A 完成并冻结；R4-B 稳定性门失败后恢复 materialized 默认；R4-C commit `6912894` 完成 RTX correctness 与 24-row 正式验证 |
| External Baseline R5 | 固定版本 FlashInfer paged-decode 公平对比、strict summary、对外技术文章 | commit `d7d4feb` 的 post-schema focused、quick、72-row formal、full 与 release check 全部通过；canonical evidence 已固化 |
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
- R3-A/R3-B ownership 与 Engine/scheduler integration、R3-C benchmark 以及 R3-D hot-path metadata cache 均已闭合。R3-D commit `fe72e27` 的 targeted/focused/full RTX correctness 分别为 `1 passed`、`61 passed, 8 subtests passed` 与 `361 passed, 25 subtests passed`。
- 优化后的 8-trial/64-row confirmation 继续确认 75% hit 将 context physical blocks 从 `64/64` 降至 `20/64`，节省 `68.8%`/`5.5 MiB`，并在固定 48-block pool 下将 admission 从 `9/16` 提高到 `16/16`。所有非零 hit-rate 的 complete、scheduler 与 Engine p50 range 均跨 1，最终性能结论冻结为 near-neutral/no stable direction；旧 24-row 结果保留为优化前基线。
- R4-A 针对 multi-layer fused transaction 每 layer 重复执行的 CUDA index reduction + `.item()` host sync。public raw primitive 保持完整检查；`PagedKVCache.begin_token()` 以纯 host invariant 证明 allocator 位置，公开 transaction API 根据 transaction id 回查该内部状态并调用 private trusted raw primitive，DecodeEngine 仍只依赖 Cache public API。commit `4018449` 已在 RTX 5070 完成 focused `73 passed, 23 subtests passed`、full `410 passed, 48 subtests passed` 与 160-row/80-pair 五轮正式矩阵。trusted complete-token p50/TPS 几何平均为 `1.7307x/1.7131x`，16/16 个 dtype/case p50 ranges 均稳定胜出，append inclusive CPU 为 `2.3612x`；7/16 个 p99 ranges跨 1，因此只冻结 p50 与 host-sync 归因，不声明稳定尾延迟或 device-kernel 加速。
- R4-B commit `8047a9c` 的 persistent metadata candidate 通过 focused `101 passed`、full `434 passed, 48 subtests passed` 与 160-row/80-pair 正式证据完整性校验。overall p50/TPS/append CPU 为 `1.2493x/1.2392x/3.0366x`，但只有 13/16 个分组的五轮 p50 最小值大于 1，未达到预注册 16/16 keep 门；因此保留正式负结果并恢复 R4-A/materialized 默认，不继续同线微调。
- rollback commit `36225d1` 已通过 focused `89 passed, 23 subtests passed`、full `410 passed, 48 subtests passed` 与 release evidence gate。随后 R4-C commit `6912894` 在 RTX 5070 通过 focused `60 passed, 17 subtests passed`、full `425 passed, 57 subtests passed`、FP16 quick 与 24-row/3-trial FP16/BF16 正式矩阵；dynamic mixed-prefix reference digest、multi-layer prompt transaction、failure rollback、prefix lifetime、block reuse 与最终零占用 cleanup 全部严格通过，R4 阶段完成。
- R5 commit `d7d4feb` 在固定 Python 3.12/Torch `2.11.0+cu128`/Triton `3.6.0`/CUDA 12.8/FlashInfer `0.6.15.post1` 环境完成 post-schema focused `93 passed, 37 subtests passed`、3-row quick、72-row/3-trial formal、full `453 passed, 94 subtests passed` 与 clean-tree release check。CUDA-core/tensor-core 的 8 组 p50 ratio 几何平均为 `1.2003x/1.2284x`，16/16 个三轮范围均高于 1；small-shape 幅度波动和 7/16 p99 range 重叠被保留，因此只冻结有限 kernel-only p50 观察，不外推端到端 runtime 或生产尾延迟。
- R2 正式结果绑定 commit `fa0f89a`；证据提交 `67bee15` 在 RTX 5070 完成 `337 passed, 25 subtests passed` 的无跳过回归。
- 当前仓库仍为 private `0.0.0` development candidate。
- clean-install、版本更新、公开与 tag 按所有者要求暂停在最后 release 阶段。

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
