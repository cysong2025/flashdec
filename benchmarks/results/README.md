# FlashDec Canonical Results

本目录保存经过 strict summarizer 校验、绑定 commit 和环境的 Markdown 结果。首页图表和性能报告从这些文件提取结论；原始 CSV、日志、trace 和模型缓存不提交到 Git。

## 代表性结果

| 层次 | 结果 | Canonical summary |
| --- | --- | --- |
| Qwen2.5-3B 长上下文 `LLM.generate` | 延迟 `−4.58%`，output TPS `+4.80%` | [Long-context model](vllm_qwen_long_context_model_latency_summary.md) |
| Qwen decode-attention | B8/ctx1024、ctx2048 p50 `−19.75% / −20.74%` | [vLLM attention](vllm_qwen_attention_summary.md) |
| Cache-owned transaction | complete-token p50 `1.7307x` | [Trusted transaction](trusted_transaction_summary.md) |
| Shared-prefix capacity | physical context blocks `64 → 20`，admission `9/16 → 16/16` | [Shared prefix](shared_prefix_capacity_summary.md) |
| Scheduler progress | boundary completion `100% / 50% / 0%` | [Scheduler matrix](scheduler_capacity_progress_summary.md) |
| FlashInfer external check | FlashInfer 在共同 kernel matrix 中 p50 更低 | [FlashInfer baseline](flashinfer_paged_decode_baseline_summary.md) |

端到端结果限定为 RTX 5070、Qwen2.5-3B BF16、固定 B8 长上下文和 blocking `LLM.generate`；它不是在线 serving 或多 GPU 结论。不同表中的 ratio 方向、shape 和计时边界可能不同，不能相乘或直接横向排序。统一解释见[性能报告](../../docs/performance_report.md)。

## Kernel 配置与数据路径

| 范围 | Summary | 结论 |
| --- | --- | --- |
| Warp selection | [Warp sweep](paged_decode_warp_selection_summary.md) | 默认 2 warps |
| Block size | [Block-size sweep](paged_decode_block_size_summary.md) | 默认 block size 32 |
| KV layout | [KV-layout sweep](paged_decode_kv_layout_summary.md) | 默认 token-major |
| Default profile | [Paged-decode profile](paged_decode_default_profile_summary.md) | 冻结后的组合配置 |
| Triton staging | [Staging sweep](paged_decode_staging_summary.md) | 候选未过 5% 门，保持 implicit default |
| RoPE/KV append | [Backend comparison](rope_kv_append_backends_summary.md) | fused CUDA 为 Engine 默认写入路径 |
| Complete-step propagation | [DecodeEngine workload](decode_engine_workload_trials3_summary.md) | 区分 append-only 与完整 step 收益 |
| Stage attribution | [Stage profile](decode_engine_stage_profile_summary.md) | profiler 只用于归因 |

## Runtime、容量与一致性

| 范围 | Summary | 结论 |
| --- | --- | --- |
| Scheduler | [Capacity/progress matrix](scheduler_capacity_progress_summary.md) | lifetime commitment 解决 boundary deadlock |
| Multi-layer transaction | [144-row matrix](multi_layer_transaction_summary.md) | complete-token 收益主要来自 append/launch |
| Trusted dispatch | [160-row matrix](trusted_transaction_summary.md) | 移除重复 host scalar sync |
| Persistent metadata | [Negative result](persistent_metadata_candidate_summary.md) | 仅 13/16 分组稳定，未采用 |
| Shared prefix | [8-trial confirmation](shared_prefix_capacity_summary.md) | 容量和 admission 改善；latency 无稳定方向 |
| Integrated lifecycle | [24-row matrix](integrated_runtime_lifecycle_summary.md) | scheduler/transaction/prefix/rollback/cleanup 不变量通过 |

## vLLM 与外部实现

| 范围 | Summary | 结论 |
| --- | --- | --- |
| FlashInfer paged decode | [72-row baseline](flashinfer_paged_decode_baseline_summary.md) | FlashInfer p50 更低；kernel-only |
| vLLM attention | [50-row comparison](vllm_qwen_attention_summary.md) | B8 两个目标 shape 通过 |
| Qwen generation correctness | [Cross-backend correctness](vllm_qwen_model_correctness_summary.md) | 第一 token 8/8 一致，完整 rollout 只作描述 |
| Qwen short fixed-batch | [Model latency](vllm_qwen_model_latency_summary.md) | 小幅改善但冻结目标未通过 |
| Qwen long-context fixed-batch | [Model latency](vllm_qwen_long_context_model_latency_summary.md) | 3% 端到端门槛通过 |
| Qwen online serving | [Serving comparison](vllm_qwen_serving_summary.md) | TPOT 通过，throughput 门槛略失 |

正向和负向结果均保留。每份 summary 自带环境、case、trial、计时范围、commit 和结果边界；后续实现不能覆盖历史证据。

## README 图表

[`public_results_snapshot.json`](public_results_snapshot.json) 是从上述 summaries 提取的 processed snapshot，不是原始 benchmark dataset。生成器会重新解析权威表格并确定性生成 light/dark SVG：

```bash
python scripts/generate_public_architecture.py --check
python scripts/generate_public_results_chart.py --check
```

当前图表优先展示真实 Qwen/vLLM 收益，同时保留 KV capacity、transaction、scheduler 和 FlashInfer reality check。Canonical Markdown 始终是最终权威来源。

## 本地输出

`.gitignore` 排除 `*.csv`、`*.log`、quick/smoke summaries、profiles 和 `local_backups/`。它们用于本地调试和审计，不属于发布内容；确认相应 canonical summary 已生成后即可清理。
