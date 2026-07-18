# Changelog

本文记录 FlashDec 的公开版本变化。项目当前仍处于 `0.0.0` 开发版本；只有 release gate 全部通过后才创建 `v0.1.0` 条目和 Git tag。

## [Unreleased]

### Added

- PyTorch dense/paged decode reference 与数值稳定 softmax。
- Triton dense decode 和 paged decode kernels，支持 FP16/BF16、MHA/GQA/MQA、head_dim 64/128、变长 context 和非连续 physical blocks。
- PagedKVCache request lifecycle、block allocate/free/reuse、capacity atomicity、utilization/fragmentation metrics 和 invariant validation。
- PyTorch、独立 CUDA 与 fused CUDA RoPE + paged KV append 路径。
- DecodeEngine waiting/active/finished/cancelled 状态、动态 active batch、append -> paged decode 和显式 backpressure。
- short-churn、mixed-steady、long-pressure synthetic workloads 与 complete-step wall-clock 指标。
- multi-trial backend-order 交替、严格 trial CSV validator 和跨 trial stability summary。
- 可选 DecodeEngine profiler ranges、阶段 CPU/device totals、CUDA event count、Chrome trace，以及正式 12-case matrix/range-count validator。
- `scripts/check_release.py` release artifact/version/Git gate checker。
- `scripts/run_r0_validation.py` 分阶段验证编排器：CUDA/tracked-clean 预检、产物检查、dry-run 和 WSL 到 Windows 导出。
- Scheduler v2 设计规格与 R1-A 纯 Python planner：lifetime block commitment、logical/physical capacity 分离、FIFO + aging、公平 runnable batch、结构化 decision 和 dependency-free focused tests。
- Scheduler R1-B Engine/Cache 集成：scheduler-managed request submission、Engine/Cache `state_version`、authoritative snapshot、stale/forged decision 原子拒绝、显式 rejection、initial-context seeding 和 commitment metrics。
- Scheduler R1-C trace-driven workload：cancel/greedy/lifetime 三策略、boundary-deadlock 检测、等待/公平性/commitment/physical block 指标、执行 token 与有效 token 分离，以及 RTX benchmark CLI。
- Multi-layer KV Token Transaction 设计规格：committed/pending seq_len、shared location、逐层执行、batch commit/abort 和 rollback invariant。
- Multi-layer Cache transaction 与 DecodeEngine sequential layer API：2/4-layer shared location、单次 seq_len commit、异常自动 rollback、scheduler transaction 互斥和单层 compatibility wrapper。
- R2-C fused CUDA location-only transaction write：复用 transaction 预留的 block ids/offsets，保持 allocator、rollback 和 committed seq_len 由 Cache transaction 唯一管理，并增加 2-layer FP16/BF16、GQA、Triton 与失败回滚测试。
- R2-D multi-layer workload runner 与严格 trial summary：12-case layer/batch/context 矩阵、complete-token/per-layer latency、host begin/commit、独立 profiler append/decode/launch attribution、KV bytes、rollback probe 和 paired evidence validation。
- 公开 API、文档索引、贡献指南、dependency-free Markdown link checker 与 GitHub Actions 质量门禁。
- R3-A shared-prefix ownership core：opaque prefix id、immutable multi-layer full blocks、request reference counting、private tail、inactive LRU、容量失败原子性和 shared-memory metrics。
- R3-B Engine/scheduler integration：`RequestSpec.prefix_id`、authoritative prefix metadata、admission attach、global residency + request-private commitment accounting，以及共享请求 lifecycle/invariant tests。
- R3-C shared-prefix workload runner 与严格 trial summary：0%/25%/50%/75% hit rate、bounded-capacity admission、fixed-full-batch decode、block/byte savings、attach/registration/eviction latency、trial-order rotation 与 lifecycle/accounting validator。

### Changed

- 冻结通用 paged decode 配置为 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Python package 核心依赖只保留 torch/triton；pytest 移入 `dev` extra，Ninja 保留在 `cuda-extension` extra。
- GPU Engine 明确使用 fused CUDA append policy；公开 reference API 默认仍保持 PyTorch 路径。
- DecodeEngine workload CSV、multi-trial summary 和 profiler evidence 现在绑定生成时的 Git commit。
- Release artifact/evidence gate 现在同时要求 R1 Scheduler、R2 Multi-layer 与 R3 Shared Prefix runner、validator、focused tests 和最终 Markdown summary。

### Performance evidence

- Week 10 staging 最佳候选相对 implicit default 的 p50 几何平均仅约 1.0039x，未达到 5% 门槛，因此保留默认 staging。
- Week 11 append-only full benchmark：fused CUDA p50 几何平均为 1.2226x vs torch。
- Week 12 正式 36 行 multi-trial：fused p50/p90/TPS 几何平均为 1.0668x/1.0317x/1.0811x；全部 invariant、block accounting、pair trajectory 与 seed/order 校验通过。
- short-churn 的 FP16/BF16 p50 均跨 trial 穿过 1.0；p99 ratio 范围为 0.2444x-5.0578x，因此保留为系统级噪声/负结果，不声明稳定尾延迟收益。
- 12-case complete-step profiler 显示 fusion 将 CUDA event 数减少 21.8%-45.6%，而 paged decode device time 变化仅为 -1.7%-+1.1%；收益主要来自 append/launch/runtime 路径。
- R2-D commit `fa0f89a` 的正式 144 行 multi-layer 矩阵全部通过严格校验；fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`，per-layer append device 为 `1.6103x`，decode device 为 `1.0024x`，CUDA event ratio 为 `1.9784x`。
- 24 个 dtype/case 组合中 20 个三轮 p50 稳定胜出，4 个跨过 1.0，没有稳定 torch-faster case。BF16 `l4_b4_c128` 的独立 profiler append ratio 为 `0.4980x`，但正式 complete-token p50 三轮均胜出，因此记录为 instrumented attribution anomaly，不解释为正式 wall-clock 回退。
- R2-D 每轮仅 20 repeats，nearest-rank p99 实际接近该轮最大值；p99 范围只用于报告长尾波动，不声明生产级尾延迟收益。
- R3-C commit `fd36ed0` 的 RTX 5070 FP16 quick 4 行矩阵通过严格校验。75% hit 将 context physical blocks 从 `4/4` 降至 `2/4`、peak blocks 从 `8` 降至 `6`，bounded-pool admission 从 `2/4` 提高到 `3/4`；单轮 latency 非单调，不形成性能结论。
- R3-C commit `1d5d8d0` 的 RTX 5070 正式 24 行矩阵全部通过严格校验。75% hit 将 context physical blocks 从 `64/64` 降至 `20/64`，节省 `68.8%`/`5.5 MiB`；peak blocks 从 `80` 降至 `36`，bounded-pool admission 从 `9/16` 提高到 `16/16`。attach p50 低于 `0.8 us`，decode latency 跨 dtype 非单调，因此不声明稳定加速。
- R3-C paired latency：FP16 25% p50 三轮稳定更快；FP16/BF16 75% p50 三轮稳定更慢，ratio 分别为 `0.9377x [0.9298,0.9870]` 与 `0.9054x [0.8602,0.9816]`。该负结果保留并进入 scheduler/engine attribution，不用内存收益掩盖 latency trade-off。
- R3-D submission-time shared metadata cache：移除 scheduling snapshot、commitment 与 invariant 热路径中的重复 prefix registry lookup，同时保留 Cache shared-block/version authoritative cross-check；新增 lookup-count correctness test。
- R3-D commit `fe72e27` 的 RTX 5070 8-trial confirmation 共 64 行并通过严格校验。75% hit 仍将 context physical blocks 从 `64/64` 降至 `20/64`、节省 `68.8%`/`5.5 MiB`，并将 bounded-pool admission 从 `9/16` 提高到 `16/16`。所有非零 hit-rate 的 complete、scheduler 与 Engine p50 paired range 均跨过 1，因此冻结为 near-neutral/no stable direction，不声明稳定加速或回退。
- confirmation 保留离群点：BF16 trial 1 的 25% case 主要是 Engine 整行变慢，FP16 trial 7 的 25% case 是尾部尖峰；相同执行顺序的第二轮均未复现，端点 `nvidia-smi` 快照也不足以确定根因。

### Correctness evidence

- R2-A Cache transaction 完整回归：`313 passed, 20 subtests passed`。
- R2-B commit `a009b45` RTX 5070 focused：`71 passed, 8 subtests passed in 3.71s`；完整回归：`322 passed, 20 subtests passed in 6.62s`，无 skipped、warning 或 failure。
- R2-C commit `6afc89f` RTX 5070 focused：`131 passed in 6.21s`；完整回归：`326 passed, 20 subtests passed in 6.23s`，摘要无 skipped、warning 或 failure。
- R2-D 证据提交 `67bee15` RTX 5070 最终完整回归：`337 passed, 25 subtests passed in 5.82s`，无 skipped 或 failure。
- R3-A commit `e1bb6a8` 的 WSL focused 与完整回归均报告通过；本轮未提供精确计数，因此不增加新的定量 pytest 基线。
- R3-B commit `08d0414` RTX 5070 focused：`56 passed, 14 subtests passed in 5.29s`；完整回归：`352 passed, 25 subtests passed in 9.37s`。
- R3-D commit `fe72e27` RTX 5070 targeted hot-path test：`1 passed`；focused：`61 passed, 8 subtests passed`；完整回归：`361 passed, 25 subtests passed in 6.28s`。

### Pending before v0.1.0

- clean WSL venv editable install 和 quick workload 复现。
- 将 `pyproject.toml` 与 `flashdec.__version__` 同步更新为 `0.1.0`。
- 创建并验证 `v0.1.0` tag；当前不得提前标记 release。
- 仓库可见性继续保持 private；公开、版本升级和 tag 等待所有者明确启动 release gate。
