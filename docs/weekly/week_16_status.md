# Week 16 状态记录

## 本周主题

R4-C Integrated Scheduled Multi-layer Workload：在冻结的 R4-A/materialized 路径上组合验证 Scheduler、Shared Prefix、multi-layer prompt/transaction、rollback 和完整 lifecycle cleanup。

## 当前已完成

- 确认 rollback commit `36225d1` 工作树干净；RTX 5070 的 R4-B 边界 focused 回归为 `89 passed, 23 subtests passed in 4.24s`，完整回归为 `410 passed, 48 subtests passed in 6.36s`，`check_release.py --require-evidence` 为 `PASS`。这组结果证明 R4-B candidate 已移除且 R4-A/materialized 基线完整，不改变 R4-B 正式负结果。
- 新增 `DecodeEngine.prefill_request_layers()`：调用方提供一个 prompt token 的所有 layer K/V，成功时只 commit 一次，任一 layer 失败则 rollback。
- 新增 terminal `DecodeEngine.evict_prefix()`：只有 waiting/active lifecycle 全部结束后才允许清理 inactive fixed prefix，并维持 Engine/Cache version 同步。
- 新增 dependency-free standard trace/reference trajectory：动态到达、mixed prefix hit/miss、admission/defer、显式 cancel、layer-1 failure、finish 与 block reuse 均有确定预期。
- runner 逐 step 比对 reference，并从实际观测重新计算 SHA-256 trajectory digest；随机 tensor 构建、prefix registration 与 terminal eviction 不混入 logical-step wall。
- 新增 RTX runner、24-row formal matrix 和 strict summarizer；validator 会重建 reference 并拒绝 matrix、计数、digest、reuse 或 zero-used cleanup 漂移。

## 本地验证

- R4-C config/runner/summary 与既有 R2/R4-A dependency-free tests：`55 tests` 通过。
- `python3 -m compileall -q flashdec tests benchmarks scripts`：通过。
- `python3 scripts/check_docs.py`：`Documentation check: PASS (73 files)`。
- `python3 scripts/check_release.py --require-evidence`：private `0.0.0` tree gate 通过；开发工作树预期为 dirty。
- runner/summary `--help` 与 `git diff --check`：通过。

## 当前环境限制

Mac 工作区没有 Torch/CUDA，因此只能运行 dependency-free config/runner/summary tests、编译与文档检查。CPU/reference Engine integration、fused CUDA/Triton correctness 和正式矩阵必须在 WSL RTX 5070 环境完成。

## 需要在 RTX 5070 开发板完成

1. 运行 R4-C targeted/focused suite，覆盖 multi-layer prompt rollback、完整 trace、fused transaction 与 strict summary。
2. 运行 FP16 `l2_c32` quick 并通过 dependency-free summarizer。
3. 运行 2/4 layers、64/128 context、FP16/BF16、3 trials 的 24-row formal matrix。
4. 运行完整 pytest 与 release evidence check，保存 commit、环境、CSV、log 和 summary。

## 上板后要记录

- targeted/focused/full pytest 精确计数与耗时；
- quick/formal row 数、trajectory digest、transaction/prefix 计数和 final cleanup；
- 每个 dtype/case 的 p50/p90/p99/TPS median `[min,max]`；
- 任何 fail-closed 原始日志，不删行或手工补 evidence。

## 下一步

先完成本地全套 dependency-free gates 并提交实现；随后在 RTX 5070 按[复现指南](../reproducibility.md)执行 quick、focused、formal 与 full validation。正式证据返回前，R4-C 状态保持“实现就绪、等待上板”。
