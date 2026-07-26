# FlashDec 结果索引

本目录区分审核后的 Markdown 证据与本地原始输出。当前综合状态见[交付状态](../../docs/DELIVERY_STATUS.md)，实验方法与计时边界见[性能报告](../../docs/performance_report.md)和[复现指南](../../docs/reproducibility.md)。

## Release evidence

以下文件由 `scripts/check_release.py --require-evidence` 要求存在：

| 范围 | Canonical summary | 状态 |
| --- | --- | --- |
| Block size | [Week 8](week8_block_size_summary.md) | block size 32 作为通用默认 |
| KV layout | [Week 8](week8_layout_summary.md) | token-major 作为默认 |
| Final kernel default | [Week 9](week9_final_default_summary.md) | 2 warps、block 32、token-major 的最终 profiling |
| Triton staging | [Week 10](week10_num_stages_summary.md) | stage 2 未过 5% 门，保持 implicit default |
| RoPE/KV append | [Week 11](week11_rope_kv_append_summary.md) | fused CUDA 为默认 Engine append path |
| Dynamic DecodeEngine | [Week 12 multi-trial](week12_decode_engine_workload_trials3_summary.md) | complete-step 正式延迟/TPS |
| Complete-step attribution | [Week 12 profiler](week12_decode_engine_profile_summary.md) | profiler 只用于归因 |
| R1 Scheduler | [36-row matrix](r1_scheduler_workload_trials3_summary.md) | 已完成 |
| R2 Multi-layer | [144-row matrix](r2_multi_layer_engine_trials3_summary.md) | 已完成 |
| R3 Shared Prefix | [8-trial/64-row confirmation](r3_shared_prefix_workload_trials8_summary.md) | 已完成，latency 无稳定方向 |
| R4-A Trusted Transaction | [160-row/80-pair matrix](r4_fused_transaction_fast_path_trials5_summary.md) | accepted/frozen |
| R4-B Persistent Metadata | [160-row/80-pair negative result](r4_persistent_transaction_metadata_trials5_summary.md) | rejected；rollback `36225d1` 通过 focused `89/23`、full `410/48` 与 release PASS |
| R4-C Integrated Workload | [24-row matrix](r4_integrated_scheduled_multi_layer_trials3_summary.md) | 已完成 |
| R5 FlashInfer | [72-row/3-trial matrix](r5_flashinfer_paged_decode_trials3_summary.md) | 有限 kernel-only baseline 已完成 |

`docs/performance_report.md` 也是 release evidence，用于把上述结果放回统一的计时和不可外推边界。

## 历史与支撑结果

以下 tracked summaries 用于追溯默认配置或阶段演进，但不是当前阶段结论的唯一入口：

- [Week 9 早期 profiler](week9_summary.md)。
- [Week 12 首轮 workload](week12_decode_engine_workload_summary.md)。
- [R3 优化前 3-trial baseline](r3_shared_prefix_workload_trials3_summary.md)。

历史结果不会因后续优化而删除；跨 commit 数字不能直接当作同一 A/B。

## 本地原始输出

`.gitignore` 默认排除：

```text
benchmarks/results/*.csv
benchmarks/results/*.log
benchmarks/results/*_quick_summary.md
benchmarks/results/*_smoke.md
benchmarks/results/local_backups/
benchmarks/profiles/
```

这些文件可以保存原始 trial、JIT、quick、profiler 或回传备份，但都不进入 release artifact gate。结果目录顶层的 `*_quick_summary.md`、`*_smoke.md` 和 `local_backups/` 也不进入 canonical documentation scan；`benchmarks/profiles/` 虽由 Git 忽略，其中若出现 Markdown 仍接受链接检查。不要为了仓库整洁自动删除本地产物；正式结论只从审核后提交的 canonical Markdown 读取。
