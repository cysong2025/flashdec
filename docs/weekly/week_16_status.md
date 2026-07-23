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
- commit `6912894` 已在 RTX 5070 完成最终验证：focused `60 passed, 17 subtests passed in 3.09s`，完整回归 `425 passed, 57 subtests passed in 6.52s`。
- FP16 `l2_c32` quick 为 1 row/1 trial，complete-step p50 `1.341569 ms`、TPS `209.887`；预注册 formal matrix 为 24 rows/3 trials，8 个 dtype/case 分组全部通过严格校验。

## 本地验证

- R4-C config/runner/summary 与既有 R2/R4-A dependency-free tests：`55 tests` 通过。
- `python3 -m compileall -q flashdec tests benchmarks scripts`：通过。
- `python3 scripts/check_docs.py`：`Documentation check: PASS (73 files)`。
- `python3 scripts/check_release.py --require-evidence`：private `0.0.0` tree gate 通过；开发工作树预期为 dirty。
- runner/summary `--help` 与 `git diff --check`：通过。

## RTX 5070 正式验证

- 环境：NVIDIA GeForce RTX 5070、PyTorch `2.11.0+cu128`、CUDA Toolkit 12.8、commit `6912894`。
- focused：`60 passed, 17 subtests passed in 3.09s`。
- full：`425 passed, 57 subtests passed in 6.52s`。
- quick：FP16 `l2_c32`，1 row/1 trial；p50 `1.341569 ms`，TPS `209.887`。
- formal：2/4 layers、64/128 context、FP16/BF16、3 trials，共 24 rows。8 个分组的 complete-step p50 median 范围为 `1.360588–2.371724 ms`，TPS median 范围为 `43.070–126.641`。
- strict validator：reference digest、dynamic trajectory、rollback、transaction/prefix 计数、prefix lifetime、released-block reuse 与 final zero-used cleanup 全部通过。

formal trace 只有 10 个 logical steps，p90/p99 受 private context-write admission steps 主导，不能解释为 steady-state decode tail。context seeding 是 caller-supplied K/V 写入；随机构建、prefix registration 与 terminal eviction 均不在 logical-step wall 内。完整数据与边界见[R4-C 正式摘要](../../benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md)。

## 下一步

R4-C 与 R4 阶段已经闭合。下一步保持 private `0.0.0` 维护状态；只有所有者明确启动后，才进入可选 v0.1.0 release gate 或 R5 公开基线工作。
