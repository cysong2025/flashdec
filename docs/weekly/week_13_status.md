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
- Mac 未安装 PyTorch，因此不能把 transaction tensor tests 记录为已通过。
- RTX 5070 WSL focused/full：待执行。

## 下一步

1. 在 WSL 运行 `tests/test_multi_layer_transaction.py`、相关 Cache/Engine focused tests 和完整回归。
2. 验证通过后实现 R2-B Engine sequential layer API。
3. R2-B 通过后才重构 fused CUDA 为 location-only write，不重新做已冻结 kernel 参数 sweep。
