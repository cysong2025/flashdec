# Week 14 状态记录

## 本周主题

Shared Prefix Blocks：从 Cache ownership core 扩展到 DecodeEngine 与 block-aware scheduler 的 shared-aware admission。

## 当前已完成

### R3-A Cache ownership core

- `PagedKVCache.register_prefix()` 接收 immutable multi-layer full blocks。
- 多请求共享同一 physical block ids；request tail 保持私有。
- active refcount 与 prefix residency 分离，finish/cancel 只释放 private tail。
- inactive LRU、capacity failure atomicity、saved blocks/bytes metrics 和 shared-aware invariant validation 已实现。
- 2026-07-17 WSL focused 与完整回归均报告通过；本轮未提供精确通过数量，因此不记录新的定量 pytest 基线。

### R3-B Engine/scheduler integration

- `RequestSpec` 增加 optional opaque `prefix_id`。
- `DecodeEngine.register_prefix()` 在 request submission 前同步 Cache registry。
- Engine 从 Cache 派生 `shared_prefix_blocks`，并要求 prefix 覆盖完整 initial context。
- admission 自动执行 request 注册与 prefix attach，不逐 token 复制相同上下文。
- Scheduler 将 resident prefix 作为全局物理占用计一次，将每个 request 的 future decode blocks 作为 private lifetime commitment。
- 两个 request 共享 prefix 时，logical block tables 可以重复引用同一 physical ids，而 snapshot physical accounting 不重复计费。
- 2026-07-17 RTX 5070 WSL focused：`56 passed, 14 subtests passed in 5.29s`。
- 2026-07-17 RTX 5070 WSL 完整回归：`352 passed, 25 subtests passed in 9.37s`。

### R3-C Benchmark harness

- 0%/25%/50%/75% hit-rate runner 已实现，FP16/BF16 与相邻 trial case-order 轮转均写入 CSV。
- bounded-capacity admission 与 fixed-full-batch decode 使用独立 probe，避免把 batch 下降误写成 latency 收益。
- 计时前验证 shared/private context materialization 一致，计时后验证 shared K/V immutable。
- strict summary validator 检查 matrix、seed、capacity monotonicity、block/byte accounting、prefix lifecycle 与 cleanup。
- Mac dependency-free 新增测试 `8 tests` 通过。
- commit `fd36ed0` 的 RTX 5070 FP16 quick 共 4 行，matrix、capacity commitment、block/byte accounting、materialized context、immutable prefix、eviction 与 cleanup 全部通过严格校验。
- quick 从 0% 到 75% hit 将 context physical blocks 从 `4/4` 降至 `2/4`、peak blocks 从 `8` 降至 `6`；bounded pool admission 从 `2/4` 提高到 `3/4`。
- quick latency 在四档间非单调，每档只有 3 次正式 step 采样，因此不形成性能结论。
- commit `1d5d8d0` 的 RTX 5070 FP16/BF16 三轮正式矩阵共 24 行，全部通过严格校验。
- 正式 75% hit 将 context physical blocks 从 `64/64` 降至 `20/64`，节省 `68.8%`/`5.5 MiB`；包含 private tail 的 peak blocks 从 `80` 降至 `36`，减少 55%。
- 固定 48-block bounded pool 下，admission 随 hit rate 从 `9/16`、`12/16`、`15/16` 提高到 `16/16`。
- 非零 hit-rate 的 attach p50 均小于 `0.8 us`。paired p50 中只有 FP16 25% 三轮稳定更快；FP16/BF16 75% 均三轮稳定更慢，ratio 分别为 `0.9377x` 与 `0.9054x`，TPS ratio 为 `0.9569x` 与 `0.9022x`。

## 当前环境限制

macOS Codex 工作区没有项目 torch/pytest/CUDA 环境，只能执行 dependency-free unittest、`compileall`、文档检查与静态 diff 检查。动态 Engine correctness 继续在 RTX 5070 WSL 环境验证。

## 需要在 RTX 5070 完成

1. 将 CSV、log 与正式 summary 回传 Mac 并归档。
2. 使用 CSV 的 `scheduler_p50_ms` 与 `engine_step_p50_ms` 分离 75% hit 回退来源。
3. 根据 attribution 做有边界优化或冻结 trade-off，再完成完整 pytest 回归。

## 下一步

R3-B admission/commitment 语义已冻结。下一步执行 R3-C RTX 实测；clean install、版本与 tag 仍保留到最后。
