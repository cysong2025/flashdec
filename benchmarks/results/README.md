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
| vLLM Qwen attention | [50-row/25-pair matrix](vllm_qwen_attention_summary.md) | B8 ctx1024/2048 external-kernel gate 通过 |
| Qwen cross-backend correctness | [8-prompt generation](vllm_qwen_model_correctness_summary.md) | 第一 token 8/8 一致；完整 rollout 5/8 只作描述 |
| Qwen fixed-batch model | [12-row/6-pair matrix](vllm_qwen_model_latency_summary.md) | 两个 case 小幅改善，但预注册 target 失败 |
| Qwen long-context fixed-batch model | [R8 16-row/8-pair matrix](vllm_qwen_long_context_model_latency_summary.md) | input8192/output4096 延迟降低 `4.58%`、TPS 提升 `4.80%`；3% 端到端门槛通过 |
| Qwen online serving | [6-row/3-server-pair matrix](vllm_qwen_serving_summary.md) | TPOT 目标通过，throughput 目标略失，整体 gate 失败 |

`docs/performance_report.md` 也是正式证据，用于把上述结果放回统一的计时和不可外推边界。

## R7 vLLM Qwen外部比较

R7 是与旧 FlashInfer comparison 不同的环境和问题：它固定 `vLLM==0.25.1`、PyTorch `2.11.0+cu130` 和 Qwen2.5-3B BF16，复用 vLLM 的模型、KV cache、scheduler 与 API server，只替换 eligible single-token decode attention。

- [attention kernel](vllm_qwen_attention_summary.md) 是可交付的性能通过项：B8/ctx1024 与 B8/ctx2048 p50 分别为 vLLM Triton 的 `0.8025x` 与 `0.7926x`。
- [model correctness](vllm_qwen_model_correctness_summary.md) 证明第一步 greedy top-1 为 8/8；长 rollout 的浮点分叉按原样保留。
- [fixed-batch model](vllm_qwen_model_latency_summary.md) 和 [online serving](vllm_qwen_serving_summary.md) 都保留冻结门槛失败，不能从 kernel 数字推导完整模型或服务的同幅度收益。

四份 summary 分别绑定其被测 clean commit。后续文档/summary 提交不会改变当时的二进制路径；若修改 `flashdec/vllm_backend.py`、paged-decode kernel 或协议，必须生成新的 evidence commit 和结果，不能覆盖这些历史文件。

## R8 Qwen 长上下文端到端门槛

R8 在同一 RTX 5070、Qwen2.5-3B BF16、`vLLM==0.25.1` 环境中继续优化 eligible split-decode，并将确认性目标冻结为 B8/input8192/output4096 的 `FlashDec/vLLM Triton <= 0.970x`。正式矩阵绑定 clean commit `3ba68e3f5317f1a9e2f0a2830697c55de6dfe9d0`，两个 case、每 case 4 个独立 backend process pairs，按 AB/BA 平衡顺序运行。

- [R8 fixed-batch long-context model](vllm_qwen_long_context_model_latency_summary.md) 中，目标 case 的 paired-median ratio 为 `0.9542x [0.9530,0.9560]`，对应 `4.58%` latency reduction 和 `4.80%` output-TPS uplift；预注册的至少 3% 端到端门槛通过。
- B8/input512/output2 的集成 guard 为 `1.0029x [0.9890,1.0100]`，满足 `<=1.05x`；它证明短路径没有超过冻结回退边界，不表示短 case 获得加速。
- 8/8 CUSTOM workers 都提供唯一、canonical 的 engine-process marker，记录 CUDA Graph capture 中成功的 8-split launch；worker/parent runner 读盘校验 marker，summarizer 对 CSV 中的 canonical JSON、SHA、路径唯一性、shape、commit 与 dataset 投影做 fail-closed 复核。
- 每个请求的跨 backend 最小共同前缀满足 `>=2`；长 case 最小共同前缀为 49 tokens/request、完整序列 7/8 一致。完整自回归 hash 仍只作描述性证据。

原始 CSV、run/summary logs、datasets、worker JSON、8 个 activation markers 和 commit-scoped vLLM cache 保留在 WSL 外置目录 `/home/user/flashdec_results/r8_3ba68e3_formal_trials4/`，不提交到 Git。顶层校验值为：

| 外置产物 | SHA-256 |
| --- | --- |
| `model_latency.csv` | `ae57b1788abb61847e1faa4ee1a6ab57de0fba309c2cb5317d660e4913d503e2` |
| 生成版 `model_latency_summary.md` | `f511af02757b66cc75007768c5df7e9180ae31f3ed34853d00d00038e9354520` |
| `run.log` | `8c1956118f877f4f006c4fba50f9c91c814dcc283457c84ebc3ec59723a4b7f7` |
| `summary.log` | `1c8b5e95716fbd1a84528e8d39d6420d1d0b7b3a9023403c854edba2a4509bc6` |
| `evidence_manifest.sha256` | `cf7ea96e39133ff4bf12959877b177c31547d9eb6f17273e94ab35d19946fd57` |

这里的“端到端”严格限定为离线、固定批量、阻塞式 `LLM.generate`：包含模型执行、scheduler、KV-cache、sampling 和 Python API 开销，排除模型启动/JIT，也不替代 R7 的在线 TTFT/TPOT/throughput 结果。R7 的较短 fixed-batch target 与 serving overall gate 失败仍作为历史负结果保留。

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
