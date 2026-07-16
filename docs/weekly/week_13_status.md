# Week 13 状态记录

## 本周主题

Block-aware Scheduler 正式证据闭环，以及 Multi-layer KV Token Transaction 的 Cache reference 基础。

## R1 Scheduler 正式结果

正式命令：

```bash
python benchmarks/run_scheduler_workload.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r1_scheduler_workload_trials3.csv

python benchmarks/summarize_scheduler_workload.py \
  --input benchmarks/results/r1_scheduler_workload_trials3.csv \
  --output benchmarks/results/r1_scheduler_workload_trials3_summary.md
```

- 设备：NVIDIA GeForce RTX 5070。
- commit：`16de9d4`。
- 36/36 行 case/dtype/policy/trial 矩阵与策略不变量通过。
- boundary-deadlock：
  - `cancel_on_backpressure`：完成率 50%，强制取消 1 个请求。
  - `greedy_step_only`：完成率 0%，确定检测到 1 次 resource deadlock。
  - `lifetime_fifo_aging`：完成率 100%，无取消、无死锁；第二个请求 admission wait p90 为 64 steps。
- finite-queue：三种策略均完成 6/6 请求，无取消、无死锁。
- lifetime scheduler p50 为约 0.024-0.035 ms，约占完整 step p50 的 2%-3%。
- 三轮 TPS/p99 仍存在明显波动，因此 R1 只冻结“容量安全与进展保证”结论，不宣称默认策略在所有普通 workload 都更快。

详细结果见 `benchmarks/results/r1_scheduler_workload_trials3_summary.md`。

## R2-A 当前实现

- `PagedKVCache` 允许 `num_layers > 1`，但多层 cache 禁止使用 legacy `append()` 绕过事务。
- `begin_token(request_ids)` 对整个 batch 统一做容量检查，并只预留一次 physical block/offset。
- `write_token_layer(tx, layer_idx, k, v)` 强制按 `0..N-1` 顺序写入，所有 layer 使用相同位置。
- `commit_token(tx)` 在全部 layer 成功后让 committed `seq_len` 只增长一次。
- `abort_token(tx)` 归还本事务新分配的 block，清除 in-flight marker；已经写入的 partial bytes 位于 committed length 之外，对普通 reader 不可见。
- metrics 增加 transaction begin/commit/abort、pending requests、reserved blocks/bytes、layer writes、rollback blocks 和真实 `bytes_per_block`。
- 新增 2/4-layer、mixed tail/boundary、容量失败、partial write rollback、越序/漏层/double terminal 和 single-layer compatibility 测试。

## 当前验证状态

- Mac：`compileall` 与 `git diff --check` 通过。
- R2-A RTX 5070 WSL 完整回归：`313 passed, 20 subtests passed in 5.83s`。
- R2-A 无 skipped、warning 或 failure。
- R2-B 验证 commit：`a009b45`，Git worktree clean。
- R2-B 验证环境：WSL2 Ubuntu、Python 3.12.3、PyTorch 2.11.0+cu128、Triton 3.6.0、CUDA Toolkit/NVCC 12.8.93、Ninja 1.13.0、NVIDIA GeForce RTX 5070 11.94 GiB、sm_12.0。
- R2-B focused：`71 passed, 8 subtests passed in 3.71s`。
- R2-B 完整回归：`322 passed, 20 subtests passed in 6.62s`。
- R2-B focused/full 均无 skipped、warning 或 failure。

## R2-B 当前实现

- `DecodeEngine.begin_step()` 固定本 token 的 request rows，并调用 Cache transaction 一次性预留位置。
- `step_layer()` 按 layer 顺序执行 RoPE、reference transaction write 和对应 layer 的 paged decode。
- `commit_step()` 在全部 layer 成功后提交，并只把 completed step/token counters 增加一次。
- `step_layer()` 的输入、写入或 decode 异常会自动 abort 整个 token。
- open transaction 期间禁止 submit/admit、scheduler snapshot/decision、prefill、finish/cancel 和另一条 `step()`。
- scheduler-managed begin/commit/abort 都推进 Engine/Cache version，使旧 decision 明确 stale。
- 单层 `append_backend="torch"` 的 `step()` 已包装为同一 transaction 语义；现有 CUDA/fused 单层 fast path 保持兼容。
- R2-B 验证时 multi-layer transaction 固定使用 `append_backend="torch"`；R2-C 在同一 transaction 语义上增加 fused CUDA location-only write。
- 新增 2/4-layer per-layer reference、row order、自动 rollback、early commit、lifecycle/scheduler 互斥和单层兼容测试。

R2-B 已完成 RTX 5070 WSL focused/full correctness，冻结 sequential layer API、异常 rollback、scheduler/open transaction 互斥和单层 compatibility 结论。本轮 pytest 总耗时只作为工程回归记录，不生成性能结论。

## R2-C 当前实现

- 复用已有 fused RoPE + KV append CUDA primitive；该 primitive 原本就只消费调用方提供的 physical block ids/offsets，不持有 allocator 或 request lifecycle。
- `PagedKVCache.write_token_layer_fused_cuda()` 使用 `begin_token()` 已预留的共享位置写入指定 layer，kernel 成功返回后才推进 transaction `next_layer_idx`。
- allocator、boundary block rollback、committed `seq_len` 和 transaction commit/abort 仍完全由 Python Cache transaction 管理。
- `DecodeEngine.begin_step()` 现在允许 `append_backend="fused_cuda"`；`step_layer()` 复用同一 transaction 执行 fused RoPE/KV write，再调用 reference 或 Triton paged decode。
- 原有单层 fused `DecodeEngine.step()` fast path 未修改；非 fused 的独立 CUDA append 仍不进入 multi-layer transaction。
- 新增 GPU tests 覆盖 2-layer、两个连续 token、position 1、partial RoPE、GQA、FP16/BF16、fused CUDA + Triton 对齐，以及第二层输入失败后的 block rollback。
- Mac `compileall` 与 `git diff --check` 通过。
- R2-C 验证 commit：`6afc89f`。
- R2-C RTX 5070 focused：`131 passed in 6.21s`。
- R2-C RTX 5070 完整回归：`326 passed, 20 subtests passed in 6.23s`。
- focused/full 摘要均无 skipped、warning 或 failure；R2-C correctness 验收完成。pytest 总耗时仅作为工程回归记录，不形成性能结论。

## R2-D 当前实现

- 新增 `run_multi_layer_engine.py`，正式矩阵固定为 1/2/4 layers、batch 4/16、context 128/1024、FP16/BF16、torch/fused CUDA 和 3 trials。
- 同一 trial 的 torch/fused 使用相同 seed、shape 和 request trajectory；相邻 trial 反转 backend 顺序。
- complete-token 正式计时包含 `begin_step -> all step_layer -> commit_step`，排除输入生成、context seed、JIT、profiler 与 rollback probe。
- CUDA event 记录完整 token、每层 combined append+decode 和所有 layer 合计 device time；begin/commit 单独记录 host dispatch 时间。
- 独立 profiler probe 记录每层 append/decode device time 和 CUDA event count；instrumented 数据只做归因。
- 独立 rollback probe 在 layer 0 成功后向 layer 1 注入非法 Q，验证自动 abort、block rollback 和 request state 不可见。
- CSV 记录 KV write bytes/token、cache capacity bytes、transaction counts、block accounting 和 timing scope。
- 新增严格 summary，拒绝缺行、case/shape 不一致、pair trajectory 漂移、transaction/profile/rollback 计数错误及 seed/order 错误。
- 新增 9 个 dependency-free tests；Mac `compileall`、`git diff --check` 与 tests 均通过。
- 当前尚未运行 RTX 5070 quick/formal workload，因此没有 R2-D 性能结论。
- RTX 3-trial quick 已通过严格 6-row summary；正式矩阵首次启动时发现 PyTorch profiler cycle 默认清理事件，导致首个 `begin` range 丢失，而 append/decode 与 correctness 轨迹完整。
- profiler probe 已启用 `acc_events=True`，跨内部 cycle 累积 range；该修复等待 RTX 正式矩阵重跑验证，不改变 non-instrumented 性能计时。

## 下一步

1. RTX 5070 先运行单个 2-layer FP16 quick case，验证 CSV、profiler attribution、rollback 和严格摘要。
2. quick 通过后运行 144 行正式 multi-trial，记录稳定收益、跨 1.0 场景、p99 范围和至少一个负结果。
3. 正式证据完成后执行 clean-machine install、版本升级和 release tag；不重新 sweep kernel，也不并行启动 shared prefix。
