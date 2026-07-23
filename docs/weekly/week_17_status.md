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
- 新增 dependency-free runner/summary tests 与 optional CUDA correctness test；实际 RTX 通过计数仍待上板记录。
- 设计、命令和验收口径见 [R5 FlashInfer 基线设计](../design_flashinfer_baseline.md)；五层项目总览见[从 PagedAttention 到 Decode Runtime](../notes/from_paged_attention_to_decode_runtime.md)。

## 当前环境限制

- 当前本地文档与 dependency-free 检查环境不是 RTX 5070 CUDA 开发板，不用它产生或推断 FlashInfer 性能数字。
- `flashinfer-python==0.6.15.post1` 的 wheel/JIT 与当前 RTX 环境的 PyTorch/CUDA 组合仍需要在目标板实际验证。
- FlashInfer wrapper 需要 plan、workspace 和 CSR paged metadata；这些是真实 API 差异，但不在本次 kernel-only CUDA-event 时间中。
- 当前没有可引用的 R5 正式性能结果；不在 weekly status 中预写胜负。

## 需要在 RTX 5070 开发板完成

1. 在项目 virtualenv 通过 `.[dev,cuda-extension,baseline]` 安装 `flashinfer-python==0.6.15.post1`，保留 install log 并确认 import 版本。
2. 运行 R5 focused tests，确认 layout/metadata 适配、三 backend reference parity 和 strict summary 错误路径。
3. 运行 FP16 medium quick（warmup 2/repeat 10），要求 3 rows/1 trial 完整，且每个 row 的 `reference_validated`、`cross_backend_validated` 与 `validated_invariants` 全部为 `True`。
4. 运行 72-row/3-trial FP16/BF16 正式矩阵，不删除慢 case、失败 row 或负结果。
5. 用 strict summarizer 验证矩阵、输入 digest、环境、correctness 与有限 latency，然后运行 full pytest 与 release evidence gate。
6. 将通过验证的精简 formal summary 固化为 canonical evidence，再开始 README/ROADMAP/NEXT_STEPS 等项目整理。

## 上板后要记录

- 带时区的 run timestamp、FlashDec commit 与 clean 工作树状态；
- GPU 名称、PyTorch、Triton、PyTorch CUDA build、CUDA Toolkit、Python、128 MiB wrapper workspace 和 FlashInfer 实际 import 版本；
- 完整 focused/full pytest 计数、subtests、运行时间和 release check 结果；
- quick/formal 原始 CSV、runner log、summary log 与输出路径；
- 每个 case/dtype/backend 的 p50/p90/p99、吞吐、3-trial range 和比值方向；
- 任何不支持 shape、JIT 或 plan 错误、OOM、数值超差与 retry，包括未经预注册的临时改动；
- plan/JIT/input/reference 仍被排除的计时边界，以及不能与 FlashDec runtime workload 直接相除的限制。

## 下一步

先完成 R5 RTX 5070 quick/formal/correctness 证据闭环，对结果做保守解释并保留不可比项。R5 验收通过后再统一清理历史状态文档、索引、release evidence 与项目对外表达；当前不提前升级版本或创建 tag。
