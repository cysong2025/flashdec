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

### Changed

- 冻结通用 paged decode 配置为 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Python package 核心依赖只保留 torch/triton；pytest 移入 `dev` extra，Ninja 保留在 `cuda-extension` extra。
- GPU Engine 明确使用 fused CUDA append policy；公开 reference API 默认仍保持 PyTorch 路径。
- DecodeEngine workload CSV、multi-trial summary 和 profiler evidence 现在绑定生成时的 Git commit。

### Performance evidence

- Week 10 staging 最佳候选相对 implicit default 的 p50 几何平均仅约 1.0039x，未达到 5% 门槛，因此保留默认 staging。
- Week 11 append-only full benchmark：fused CUDA p50 几何平均为 1.2226x vs torch。
- Week 12 正式 36 行 multi-trial：fused p50/p90/TPS 几何平均为 1.0668x/1.0317x/1.0811x；全部 invariant、block accounting、pair trajectory 与 seed/order 校验通过。
- short-churn 的 FP16/BF16 p50 均跨 trial 穿过 1.0；p99 ratio 范围为 0.2444x-5.0578x，因此保留为系统级噪声/负结果，不声明稳定尾延迟收益。
- 12-case complete-step profiler 显示 fusion 将 CUDA event 数减少 21.8%-45.6%，而 paged decode device time 变化仅为 -1.7%-+1.1%；收益主要来自 append/launch/runtime 路径。

### Pending before v0.1.0

- clean WSL venv editable install 和 quick workload 复现。
- 将 `pyproject.toml` 与 `flashdec.__version__` 同步更新为 `0.1.0`。
- 创建并验证 `v0.1.0` tag；当前不得提前标记 release。
