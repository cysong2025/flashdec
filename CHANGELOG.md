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
- R4-A trusted fused transaction boundary：public raw append 保留 CUDA index 值域检查，Cache 在 host allocator reservation 时验证 provenance，public transaction API 回查内部状态后使用 private trusted raw launch；新增 detached-view tampering、checked/trusted parity、rollback 与同 commit 五轮配对 benchmark/strict summary。
- R4-C integrated scheduled multi-layer workload：原子 caller-supplied multi-layer prompt transaction、terminal Engine-owned prefix cleanup、dynamic mixed-prefix reference trajectory、layer-failure rollback、released-block reuse、observed/reference SHA-256 digest、24-row CUDA runner 与 strict summary validator。
- R5 FlashInfer public baseline：固定 `flashinfer-python==0.6.15.post1`，在相同 Q/K/V、page table、HND physical layout 与 CUDA-event timing 下运行 FlashDec Triton、FlashInfer FA2 CUDA-core/tensor-core；新增 72-row runner、strict summary、optional CUDA correctness 与 dependency-free matrix tests。
- R5 CUDA 12.8/SM120a 复现护栏：新增 `constraints/r5-cu128.txt`，固定 Torch/Triton/CUDA Python packages/Ninja；runner 在 FlashInfer import/JIT 前验证 `CUDA_HOME` 与 `FLASHINFER_CUDA_ARCH_LIST=12.0a`，CSV/strict summary 记录并拒绝环境漂移。

### Changed

- 冻结通用 paged decode 配置为 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Python package 核心依赖只保留 torch/triton；pytest 移入 `dev` extra，Ninja 保留在 `cuda-extension` extra。
- GPU Engine 明确使用 fused CUDA append policy；公开 reference API 默认仍保持 PyTorch 路径。
- DecodeEngine workload CSV、multi-trial summary 和 profiler evidence 现在绑定生成时的 Git commit。
- Release artifact/evidence gate 现在同时要求 R1 Scheduler、R2 Multi-layer、R3 Shared Prefix、R4-A Trusted Transaction 与 R4-C Integrated Workload 的 runner、validator、focused tests 和最终 Markdown summary。
- R4-A profiler attribution 改用 CPU-only WARMUP→active schedule；active CPU user annotation 提供 inclusive host time，并严格验证 checked/trusted scalar extraction。少记 range/scalar 可用相同 seed、全新 probe 最多重采集三次并记录 attempt count，多记立即失败。paired trial 先完成两条 path 的正式 wall，再运行 attribution/rollback，避免 retry 介入配对计时。未稳定关联的 append/decode device 与 CUDA-activity 字段从 strict schema 删除。
- R4-B persistent metadata candidate 未通过预注册 16/16 分组稳定性门，生产主线恢复 R4-A/materialized 默认；candidate commit 与正式负结果保留用于追溯，不继续同线微调。

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
- R4-A commit `4e18f5d` 的 RTX 5070 FP16 `l2_b4_c32` quick 保留 provisional complete-token p50/TPS `1.7856x/1.8755x`、append CPU `2.3751x` 与 item/local-scalar `20/20 -> 0/0`。旧 append/decode device 与 CUDA-event 字段来自非强契约的 CUDA user spans，已撤回；单 trial 不构成稳定 speedup 或尾延迟结论。
- R4-A commit `5d2f9c0` 的 l4 stress 在三次全新 probe 中都得到首个 decode CPU range 的正 host time（`61.332/71.163/69.992 us`）和零 correlated device time，随后 fail closed 且未写 CSV。该重复负结果使 R4 strict attribution 改为 CPU-only；不以 prime、补值或增加 retry 次数掩盖 profiler 关联缺口。
- R4-A commit `4018449` 的 RTX 5070/CUDA 12.8 五轮正式矩阵共 160 rows、80 paired trials并通过严格校验；16/16 个 `dtype x case` p50 分组均为 `trusted_faster`，且每组五轮最小值都大于 1。overall complete-token p50/p90/p99、TPS 与 append CPU/layer ratio 为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`，R4-A 据此冻结。
- 7/16 个 R4-A 分组的 p99 range 穿过 1；overall p99 几何平均只作为聚合统计保留，不能声明稳定尾延迟收益。CPU-only attribution 证明移除了 scalar extraction 等待，不代表 kernel device execution 本身加速。
- R4-B commit `8047a9c` 的 160-row/80-pair RTX 正式矩阵通过完整性校验，overall complete-token p50/TPS 与 append CPU/layer 为 `1.2493x/1.2392x/3.0366x`；只有 13/16 个分组的五轮 p50 最小值大于 1，正式 keep gate 失败。BF16 `l2_b4_c1024`、FP16 `l2_b16_c128` 与 FP16 `l4_b16_c128` 保留跨 1 范围，不删除失败样本。
- R4-C commit `6912894` 的 RTX 5070 formal matrix 为 FP16/BF16、2/4 layers、64/128 context、3 trials，共 24 rows；8 个分组的 complete-step p50 median 为 `1.360588–2.371724 ms`，TPS median 为 `43.070–126.641`。全部 strict lifecycle gate 通过；p90/p99 受有限 trace 的 private context-write steps 主导，不声明 steady-state tail 或 shared-prefix latency speedup。

### Correctness evidence

- R2-A Cache transaction 完整回归：`313 passed, 20 subtests passed`。
- R2-B commit `a009b45` RTX 5070 focused：`71 passed, 8 subtests passed in 3.71s`；完整回归：`322 passed, 20 subtests passed in 6.62s`，无 skipped、warning 或 failure。
- R2-C commit `6afc89f` RTX 5070 focused：`131 passed in 6.21s`；完整回归：`326 passed, 20 subtests passed in 6.23s`，摘要无 skipped、warning 或 failure。
- R2-D 证据提交 `67bee15` RTX 5070 最终完整回归：`337 passed, 25 subtests passed in 5.82s`，无 skipped 或 failure。
- R3-A commit `e1bb6a8` 的 WSL focused 与完整回归均报告通过；本轮未提供精确计数，因此不增加新的定量 pytest 基线。
- R3-B commit `08d0414` RTX 5070 focused：`56 passed, 14 subtests passed in 5.29s`；完整回归：`352 passed, 25 subtests passed in 9.37s`。
- R3-D commit `fe72e27` RTX 5070 targeted hot-path test：`1 passed`；focused：`61 passed, 8 subtests passed`；完整回归：`361 passed, 25 subtests passed in 6.28s`。
- R4-A commit `1169cb8` RTX 5070 focused CUDA suite：`40 passed in 2.34s`。第一次 quick CSV 因 profiler CPU/CUDA 分组选择缺陷未通过严格 summary，性能证据作废并等待修复后重跑。
- R4-A commit `4ee5fab` 的第二次 quick 在 `flashdec::paged_decode` CPU annotation 记录 `116.108 us` host time，但其自身 device total 为 0；runner 在写 CSV 前严格失败。该结果用于修正 CPU/CUDA range 归因契约，不构成性能结论。
- R4-A commit `4e18f5d` 的第三次 quick strict summary 已通过；当前 commit 的完整 RTX 回归仍待执行，因此 correctness 完成状态仍以 `1169cb8` 的 focused CUDA suite 为限。
- R4-A commit `e88900a` 的 formal 在一个 l4 probe 捕获 8 个 CPU append ranges、7 个同名 CUDA user annotations 后于写 CSV 前严格停止；该结果证明 profiler peer 契约错误，不表示 Engine 少执行 layer，也不产生正式性能数据。
- R4-A commit `5d2f9c0` 的 l4 stress 三次均在首个 decode CPU range 得到零 correlated device time并停止；没有 CSV/summary，不能形成性能结论。
- R4-A commit `4018449` RTX 5070 focused：`73 passed, 23 subtests passed`；完整回归：`410 passed, 48 subtests passed`。正式 160-row 数据、transaction/Engine trajectory、parity、rollback、CPU range 与 scalar extraction evidence 均通过 strict validator。
- R4-B commit `8047a9c` RTX 5070 focused：`101 passed`；完整回归：`434 passed, 48 subtests passed`。formal exact parity、transaction/block/Engine trajectory、rollback、metadata lifecycle 与 zero-resident cleanup 均通过。
- R4-B rollback commit `36225d1` RTX 5070 focused：`89 passed, 23 subtests passed in 4.24s`；完整回归：`410 passed, 48 subtests passed in 6.36s`；release evidence check 为 `PASS`。
- R4-C commit `6912894` RTX 5070 focused：`60 passed, 17 subtests passed in 3.09s`；完整回归：`425 passed, 57 subtests passed in 6.52s`；FP16 quick 与 24-row formal matrix 的 reference digest、rollback、prefix lifetime、block reuse 和 final zero-used cleanup 全部通过。

### Pending before v0.1.0

- clean WSL venv editable install 和 quick workload 复现。
- 将 `pyproject.toml` 与 `flashdec.__version__` 同步更新为 `0.1.0`。
- 创建并验证 `v0.1.0` tag；当前不得提前标记 release。
- 仓库可见性继续保持 private；公开、版本升级和 tag 等待所有者明确启动 release gate。
