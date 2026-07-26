# Week 17 状态记录

## 本周主题

R5 FlashInfer 有限公开基线与项目技术表达：在 R4 系统路径已闭合后，用共同 paged decode 语义对齐一个固定版本的公开 kernel 基线，并将项目整理为算法、kernel、allocator、scheduler 和实验方法五层证据链。

## 当前已完成

- 所有者明确启动 R5，项目整理和 release gate 顺延到 R5 验收后。
- 第三方依赖冻结为 `flashinfer-python==0.6.15.post1`，只使用官方 `BatchDecodeWithPagedKVCacheWrapper`。
- FlashInfer 路径固定 `backend="fa2"`，分别覆盖 `use_tensor_cores=False` 和 `True`；FlashDec 继续使用冻结的 Triton paged decode 配置。
- 公平性契约已冻结：三个 backend 共用 Q/K/V、physical page order、page table、`seq_lens`、`sm_scale`、dtype 和 seed。FlashDec token-major `[page, head, token, dim]` 直接对应 FlashInfer `HND`。
- 计时只用 CUDA event 包围 run/kernel dispatch；input construction、reference validation、JIT、FlashInfer `plan()`、workspace 构建与 metadata 适配全部排除。
- 预注册矩阵为 small/medium/large/large_batch、FP16/BF16、3 backends、3 trials，合计 72 rows。
- 新增 `run_flashinfer_baseline.py`：检查精确 FlashInfer 版本和 clean-worktree evidence 前提，按 trial 轮转 case/dtype/backend 顺序，复用同一输入对象和各自 128 MiB FlashInfer workspace，并记录带时区的单次 run timestamp、Python/PyTorch/Triton/CUDA、reference/cross-backend 误差与 CUDA-event 统计。
- 新增 `summarize_flashinfer_baseline.py`：严格检查矩阵完整性、formal `3/10/50` 与 quick `1/2/10` 采样强度、固定 geometry、配对 page-table digest、轮转顺序、版本、128 MiB workspace、clean worktree、normalized tolerance ratio 与派生吞吐；summary 绑定 runner command 并展示绝对 p50/p90/p99，只生成描述性比值，不定胜负门。逻辑 workload GB/s 只作共同 payload proxy，不解释为 DRAM bandwidth。
- 新增 dependency-free runner/summary tests 与 optional CUDA correctness test；RTX 安装/JIT/focused 可行性和 canonical post-schema 计数均已完成并在后续条目记录。
- 2026-07-26 在独立 RTX 5070 virtualenv 确认工作组合：Torch `2.11.0+cu128`、Triton `3.6.0`、CUDA Toolkit `12.8.1`、FlashInfer `0.6.15.post1`，并对 SM 12.0 显式设置 `FLASHINFER_CUDA_ARCH_LIST=12.0a`。commit `570b2cf` 的首次 targeted correctness 为 `2 passed in 162.02s`（含 JIT），随后 focused 为 `90 passed, 24 subtests passed in 8.69s`。
- 安装排障确认无 constraints 的 baseline extra 会把既有 cu128 环境升级到 Torch 2.13/CUDA 13；新增 `constraints/r5-cu128.txt`、runner import/JIT 前 environment gate、CUDA_HOME realpath/NVCC `12.8.93` probe、CSV 环境字段与 strict summary 校验。`nvidia-cuda-nvdisasm==13.3.73` 只是 CUTLASS DSL 工具依赖，不代表 runtime 升级。
- commit `d7d4feb` 的 post-schema focused 为 `93 passed, 37 subtests passed in 5.60s`；FP16 medium quick 生成 3 rows 并通过 strict summary。
- 同一 clean commit 的 formal 生成 72 rows/3 trials，run timestamp 为 `2026-07-26T15:28:08+08:00`；所有 reference/cross-backend parity、page-table pairing、环境和 timing invariant 均通过。随后 full 为 `453 passed, 94 subtests passed in 86.33s`，release check 为 `PASS`。
- CUDA-core/tensor-core 的 8 组 p50 ratio 几何平均为 `1.2003x/1.2284x`，16/16 个三轮范围高于 1；FP16 small 幅度波动明显，p99 有 7/16 范围重叠，因此只冻结有限 kernel-only p50 观察，不声明稳定尾延迟或端到端 runtime 胜负。
- 设计、命令和验收口径见 [R5 FlashInfer 基线设计](../design_flashinfer_baseline.md)；五层项目总览见[从 PagedAttention 到 Decode Runtime](../notes/from_paged_attention_to_decode_runtime.md)。

## 结果边界

- 当前本地文档与 dependency-free 检查环境不是 RTX 5070 CUDA 开发板，不用它产生或推断 FlashInfer 性能数字。
- 依赖安装、SM120a JIT、post-schema focused、quick、formal 和 full 已在目标板验证；旧 schema CSV 仍不作为正式证据。
- FlashInfer wrapper 需要 plan、workspace 和 CSR paged metadata；这些是真实 API 差异，但不在本次 kernel-only CUDA-event 时间中。
- 三轮 `[min,max]` 不是置信区间；R5 没有事后设置的性能 pass/fail 门，正式方向与波动均按原样保留。

## RTX 5070 验收完成

1. post-schema focused、FP16 medium quick 和 72-row/3-trial FP16/BF16 正式矩阵均已执行。
2. strict summarizer 已验证矩阵、输入 digest、固定 cu128 环境、`12.0a`、correctness 与有限 latency。
3. full pytest 和 clean-tree release evidence gate 均已通过。
4. 审核后的精简结果已固化为 [R5 canonical summary](../../benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md) 并纳入 release evidence。

## 上板后要记录

- 带时区的 run timestamp、FlashDec commit 与 clean 工作树状态；
- GPU 名称、PyTorch、Triton、PyTorch CUDA build、CUDA Toolkit、Python、128 MiB wrapper workspace 和 FlashInfer 实际 import 版本；
- 完整 focused/full pytest 计数、subtests、运行时间和 release check 结果；
- quick/formal 原始 CSV、runner log、summary log 与输出路径；
- 每个 case/dtype/backend 的 p50/p90/p99、吞吐、3-trial range 和比值方向；
- 任何不支持 shape、JIT 或 plan 错误、OOM、数值超差与 retry，包括未经预注册的临时改动；
- plan/JIT/input/reference 仍被排除的计时边界，以及不能与 FlashDec runtime workload 直接相除的限制。

## 下一步

R5 已验收完成。下一阶段按所有者要求进行项目整理和交付审查，统一检查文档状态、结果索引、对外表达与仓库结构；当前仍不提前升级版本、公开仓库或创建 tag。
