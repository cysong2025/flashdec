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
- 可选 DecodeEngine profiler ranges、阶段 CPU/device totals、CUDA event count 和 Chrome trace。
- `scripts/check_release.py` release artifact/version/Git gate checker。
- Scheduler v2 设计规格：lifetime block commitment、logical/physical capacity 分离、FIFO + aging、公平 runnable batch 与 stale-decision gate。

### Changed

- 冻结通用 paged decode 配置为 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Python package 核心依赖只保留 torch/triton；pytest 移入 `dev` extra，Ninja 保留在 `cuda-extension` extra。
- GPU Engine 明确使用 fused CUDA append policy；公开 reference API 默认仍保持 PyTorch 路径。

### Performance evidence

- Week 10 staging 最佳候选相对 implicit default 的 p50 几何平均仅约 1.0039x，未达到 5% 门槛，因此保留默认 staging。
- Week 11 append-only full benchmark：fused CUDA p50 几何平均为 1.2226x vs torch。
- Week 12 首轮 complete-step full benchmark：fused p50/p90/TPS 几何平均为 1.0537x/1.0588x/1.0674x；p99 为 0.9641x，仍需 3-trial 复验。
- short-churn 首轮 p50 未获得稳定 fusion 收益，作为系统级负结果保留。

### Pending before v0.1.0

- RTX 5070 新提交 focused/full regression。
- 36-row multi-trial CSV、validator 和稳定性摘要。
- complete-step profiler ranges、CUDA event count 与 trace 验证。
- clean WSL venv editable install 和 quick workload 复现。
- 将 `pyproject.toml` 与 `flashdec.__version__` 同步更新为 `0.1.0`。
- 创建并验证 `v0.1.0` tag；当前不得提前标记 release。
