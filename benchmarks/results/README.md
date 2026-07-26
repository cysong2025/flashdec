# FlashDec 结果索引

本目录区分审核后的 Markdown 证据与本地原始输出。研究问题与实现边界见[研究问题](../../docs/research_questions.md)，实验方法与计时边界见[性能报告](../../docs/performance_report.md)和[复现指南](../../docs/reproducibility.md)。

## Formal evidence

以下文件由 `scripts/check_release.py --require-evidence` 要求存在：

| 研究范围 | Summary | 关键结论或边界 |
| --- | --- | --- |
| Warp selection | [Warp sweep](paged_decode_warp_selection_summary.md) | 2 warps 在 28/28 个 full-sweep p50 分组中胜出；绑定当前 kernel mapping |
| Block size | [Block-size sweep](paged_decode_block_size_summary.md) | block size 32 作为通用默认 |
| KV layout | [KV-layout sweep](paged_decode_kv_layout_summary.md) | token-major 作为默认 |
| Final kernel default | [Default profile](paged_decode_default_profile_summary.md) | 2 warps、block 32、token-major 的最终 profiling |
| Triton staging | [Staging sweep](paged_decode_staging_summary.md) | stage 2 未过 5% 门，保持 implicit default |
| RoPE/KV append | [Backend comparison](rope_kv_append_backends_summary.md) | fused CUDA 为默认 Engine append path |
| Dynamic DecodeEngine | [Three-trial workload](decode_engine_workload_trials3_summary.md) | complete-step 正式延迟/TPS |
| Complete-step attribution | [Stage profile](decode_engine_stage_profile_summary.md) | profiler 只用于归因 |
| Scheduler progress | [36-row matrix](scheduler_capacity_progress_summary.md) | lifetime commitment 解决 boundary deadlock；不是普通 workload speedup |
| Multi-layer transaction | [144-row matrix](multi_layer_transaction_summary.md) | complete-token 收益主要来自 append/launch path |
| Shared-prefix capacity | [8-trial/64-row confirmation](shared_prefix_capacity_summary.md) | KV capacity/admission 改善；latency 无稳定方向 |
| Trusted transaction | [160-row/80-pair matrix](trusted_transaction_summary.md) | 移除重复 host scalar sync；p99 仍有范围重叠 |
| Persistent metadata candidate | [160-row/80-pair negative result](persistent_metadata_candidate_summary.md) | 仅 13/16 分组稳定，未采用 |
| Integrated runtime invariants | [24-row matrix](integrated_runtime_lifecycle_summary.md) | scheduler/transaction/prefix/rollback/reuse lifecycle 全部通过 |
| FlashInfer kernel baseline | [72-row/3-trial matrix](flashinfer_paged_decode_baseline_summary.md) | 共同 paged-decode kernel scope；不比较完整 runtime |

`docs/performance_report.md` 也是正式证据，用于把上述结果放回统一的计时和不可外推边界。

## 公开结果概览

README 中的研究结果图由受版本控制的 [`public_results_snapshot.json`](public_results_snapshot.json) 生成。该文件是从上表 summaries 审核后提取的 processed snapshot，不是原始 benchmark dataset；正式 Markdown 仍是最终权威来源。

```bash
python scripts/generate_public_results_chart.py --check
```

生成器会解析来源文件中的 validation metadata、目标 Markdown 表格和 optimization outcome，再确定性生成 [light](../../docs/assets/flashdec-results-overview-light.svg) / [dark](../../docs/assets/flashdec-results-overview-dark.svg) SVG。图中分别展示 scheduler progress、KV-pool capacity、transaction optimization、integrated lifecycle 和 kernel-only 外部基线；不同图块的 ratio 方向与 workload 不同，不能合并成一个总加速结论。

## 历史与支撑结果

以下 tracked summary 用于追溯 shared-prefix metadata 优化前的结果，但不是主要结论入口：

- [Metadata-cache 优化前 3-trial baseline](shared_prefix_pre_metadata_cache_summary.md)。

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

这些文件可用于本地 trial、JIT、quick 或 profiler 调试，但不进入正式证据和文档扫描。确认对应 Markdown summary 已保留后，可以安全删除这些本地产物。
