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
- dependency-free scheduler、文档与静态检查在 Mac 工作区通过；PyTorch/RTX 回归待执行。

## 当前环境限制

macOS Codex 工作区没有项目 torch/pytest/CUDA 环境，只能执行 dependency-free unittest、`compileall`、文档检查与静态 diff 检查。动态 Engine correctness 继续在 RTX 5070 WSL 环境验证。

## 需要在 RTX 5070 完成

1. R3-B scheduler/Engine focused tests。
2. 完整 pytest 回归，确认现有 R1/R2/CUDA 路径无回退。
3. R3-C 0%/25%/50%/75% hit-rate benchmark 与严格 summary。

## 下一步

R3-B 回归通过后冻结 admission/commitment 语义，随后实现 R3-C benchmark。clean install、版本与 tag 仍保留到最后。
