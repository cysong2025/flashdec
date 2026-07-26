# FlashDec 交付状态

本文是 FlashDec 当前工程交付状态的唯一简明入口。阶段历史见[项目演进](PROJECT_PLAN.md)和[阶段日志](weekly/README.md)，详细数字见[性能报告](performance_report.md)，复现命令见[复现指南](reproducibility.md)。

## 状态快照

截至 2026-07-26：

- R1–R5 研究与工程交付均已完成，默认实现和正式证据已经冻结。
- 当前默认路径是 token-major KV、block size 32、2 warps、implicit stages、fused CUDA append、Triton decode、lifetime FIFO + aging scheduler，以及 R4-A materialized trusted transaction。
- 仓库保持 private `0.0.0` development candidate，不代表已经发布 `v0.1.0`。
- 项目整理只统一文档、索引、证据入口和结构门禁，不修改 kernel/runtime 行为，也不重新解释历史数据。
- 所有者已明确暂缓新环境复现；fresh virtualenv、版本升级、公开设置和 tag 均不在本轮范围内。

## 可交付成果

| 阶段 | 工程能力 | 主要实现 | 验证与正式证据 | 冻结边界 |
| --- | --- | --- | --- | --- |
| Foundation / R0 | PyTorch dense/paged reference、Triton paged decode、Paged KV lifecycle、CUDA/fused append、动态 DecodeEngine | [`flashdec/`](../flashdec)、[总体设计](design.md)、[公开 API](API.md) | [性能报告](performance_report.md)、[Week 10 staging](../benchmarks/results/week10_num_stages_summary.md)、[Week 11 append](../benchmarks/results/week11_rope_kv_append_summary.md)、[Week 12 workload](../benchmarks/results/week12_decode_engine_workload_trials3_summary.md) | 单 GPU、单 token decode；不包含完整模型或服务层 |
| R1 Scheduler | lifetime block commitment、FIFO + aging、fair runnable subset、stale decision 拒绝 | [`scheduler.py`](../flashdec/scheduler.py)、[`scheduled_workload.py`](../flashdec/scheduled_workload.py)、[设计](design_scheduler.md) | [36-row 正式摘要](../benchmarks/results/r1_scheduler_workload_trials3_summary.md) | 价值是容量安全和进展保证，不宣称普通 workload 无条件更快 |
| R2 Multi-layer | 多层共享 token 位置、顺序 layer write、单次 seq_len commit、失败 rollback | [`cache.py`](../flashdec/cache.py)、[`engine.py`](../flashdec/engine.py)、[设计](design_multi_layer_kv_transaction.md) | [144-row 正式摘要](../benchmarks/results/r2_multi_layer_engine_trials3_summary.md) | 是调用方提供 Q/K/V 的 token transaction，不是完整 Transformer executor |
| R3 Shared Prefix | immutable full-block 共享、refcount、private tail、inactive LRU、shared-aware admission | [`cache.py`](../flashdec/cache.py)、[`engine.py`](../flashdec/engine.py)、[设计](design_shared_prefix_blocks.md) | [8-trial/64-row confirmation](../benchmarks/results/r3_shared_prefix_workload_trials8_summary.md) | 稳定收益是 KV capacity/admission；latency 为 near-neutral/no stable direction |
| R4 Trusted / Integrated | Cache-owned trusted validation、R4-B candidate 评估/回滚、统一 scheduled multi-layer trajectory | [`_fused_rope_kv_append.py`](../flashdec/_fused_rope_kv_append.py)、[`integrated_workload.py`](../flashdec/integrated_workload.py)、[R4-C 设计](design_integrated_scheduled_multi_layer.md) | [R4-A](../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)、[R4-B 负结果](../benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)、[R4-C](../benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md) | R4-A/materialized 保持默认；R4-B 未过 16/16 keep gate，rollback `36225d1` 已通过 focused `89/23`、full `410/48` 与 release PASS |
| R5 External Baseline | 固定 FlashInfer 0.6.15.post1 的共同 paged-decode kernel-only 对比 | [`run_flashinfer_baseline.py`](../benchmarks/run_flashinfer_baseline.py)、[`summarize_flashinfer_baseline.py`](../benchmarks/summarize_flashinfer_baseline.py)、[设计](design_flashinfer_baseline.md) | [72-row/3-trial 正式摘要](../benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md) | 只比较共同 kernel scope；不比较 scheduler、KV ownership、transaction 或 serving |

## Canonical evidence

`python scripts/check_release.py --require-evidence` 要求以下当前证据存在：

- 默认配置与数据路径：[Week 8 block size](../benchmarks/results/week8_block_size_summary.md)、[Week 8 layout](../benchmarks/results/week8_layout_summary.md)、[Week 9 final default](../benchmarks/results/week9_final_default_summary.md)、[Week 10 staging](../benchmarks/results/week10_num_stages_summary.md)、[Week 11 append](../benchmarks/results/week11_rope_kv_append_summary.md)。
- 完整 step：[Week 12 multi-trial](../benchmarks/results/week12_decode_engine_workload_trials3_summary.md)、[Week 12 profiler](../benchmarks/results/week12_decode_engine_profile_summary.md)。
- 系统阶段：[R1](../benchmarks/results/r1_scheduler_workload_trials3_summary.md)、[R2](../benchmarks/results/r2_multi_layer_engine_trials3_summary.md)、[R3](../benchmarks/results/r3_shared_prefix_workload_trials8_summary.md)。
- R4：[trusted fast path](../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)、[persistent metadata 负结果](../benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)、[integrated workload](../benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md)。
- R5：[FlashInfer baseline](../benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md)。
- 综合解释：[性能报告](performance_report.md)。

`benchmarks/results/` 中受 Git 管理的 Markdown 是审核后的精简证据；同目录的 CSV、log、quick summary、`local_backups/` 以及 `benchmarks/profiles/` 默认忽略，只作为本地原始记录，不属于发布 artifact。项目整理不会删除这些本地文件。

## 已知负结果与限制

- Triton `num_stages=2` 的 p50 几何平均收益约 `1.0039x`，未达到 5% 门槛，默认保持 implicit stages。
- 独立 CUDA append 没有稳定优于 torch；默认 Engine 使用 fused CUDA append。
- R3 的内存和 admission 收益稳定，但非零 hit-rate latency range 全部跨 1，不声明稳定加速或回退。
- R4-B persistent metadata 只有 13/16 个 p50 分组稳定胜出，未通过预注册 keep gate；rollback commit `36225d1` 的 focused `89 passed, 23 subtests passed`、full `410 passed, 48 subtests passed` 与 release evidence check 均通过。
- R5 的 16/16 个 p50 range 方向有利于 FlashInfer，但绝对 p99 有 7/16 范围重叠；不声明生产尾延迟或端到端 runtime 胜负。
- 当前不执行模型 forward、tokenizer、sampling、网络服务、TP/PP、多机、swap/offload 或生产级抢占。
- Shared Prefix 只接收调用方已经构建的 immutable full blocks；R4-C 只事务性导入调用方提供的多层 context K/V，不执行模型 prompt/prefill forward。

## 验证基线

- 最新 GPU 证据代码 commit `d7d4feb`：post-schema focused `93 passed, 37 subtests passed`，full `453 passed, 94 subtests passed`，R5 formal 72 rows/3 trials。
- canonical evidence commit `01d8c7c`：`python scripts/check_release.py --require-clean --require-evidence` 为 `PASS`。
- 本轮项目整理只变更文档、索引、release artifact coverage 和对应 dependency-free test；documentation/release/compile/diff checks 通过，dependency-free suite 为 `145 passed, 94 subtests passed`，不需要重跑 GPU 性能矩阵。
- 本轮验证结果记录在 [Stage 18 状态](weekly/week_18_status.md)。

## 暂缓的 release 工作

以下内容尚未完成，因此项目不能称为已发布版本：

1. fresh WSL virtualenv editable install 与新环境 correctness/quick workload；
2. 审核 fresh-environment `pip freeze`；
3. `0.0.0 -> 0.1.0` 版本更新；
4. repository visibility/license 的最终决定；
5. `v0.1.0` tag 与 GitHub release。

这些步骤只有在所有者明确恢复 release gate 后执行。当前后续状态见[下一步](NEXT_STEPS.md)。
