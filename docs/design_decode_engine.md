# DecodeEngine 与 Multi-layer Transaction 设计说明

## 研究问题

`DecodeEngine` 回答的问题是：在 request 持续到达、完成或取消时，如何把 Paged KV ownership、单层兼容 step、多层 sequential token transaction 和调用方提供的 prompt/context K/V 原子导入组织成一条可验证的数据路径。它不处理模型投影、采样、模型 prefill forward 或网络服务。

每一步把 active request rows 按固定顺序送入。默认 active batch 使用稳定的提交顺序；调用方显式传入 `request_ids` 时保留其 row order：

```text
Q/K/V
  |
RoPE + paged KV append (torch / cuda / fused_cuda)
  |
block_tables + seq_lens
  |
paged decode attention (reference / Triton)
  |
output rows in the same request-id order
```

## Request 状态机

```text
add_request
    |
 waiting --admit--> active --finish_request--> finished
                       |
                       +--cancel_request--> cancelled
                       |
                       +--step backpressure--> active (no mutation)
```

- `waiting`：Engine 已知但尚未创建 cache request，不占 physical block。
- `active`：已调用 `PagedKVCache.add_request()`，可进入 batch。
- `finished/cancelled`：调用 cache lifecycle API 后释放 block，不能重新激活。

## API

```python
engine = flashdec.DecodeEngine(
    cache,
    append_backend="fused_cuda",  # GPU engine policy
    decode_backend="triton",
    num_warps=2,
)

engine.add_request("r1")
engine.admit(["r1"])

result = engine.step(q, k, v, request_ids=["r1"])
if result.status == engine.STEP_BACKPRESSURE:
    # no block ownership / seq_len / lifecycle mutation happened
    handle_backpressure(result.needed_new_blocks, result.free_blocks)
else:
    out = result.output

engine.finish_request("r1")
```

`DecodeStepResult` 在成功时返回 output、pre-append positions、block tables 与 post-append seq lens。backpressure 时不返回 output，但包含 request ids、`needed_new_blocks`、`free_blocks` 和原因 `insufficient_physical_blocks`。

Multi-layer transaction API：

```python
tx = engine.begin_step(request_ids)
try:
    layer0 = engine.step_layer(tx, 0, q0, k0, v0)
    layer1 = engine.step_layer(tx, 1, q1, k1, v1)
    result = engine.commit_step(tx)
except Exception:
    # step_layer 的数据路径异常已经自动 abort；调用方只需丢弃中间输出。
    raise
```

调用方也可以在尚未发生 layer 错误时显式执行 `engine.abort_step(tx)`。每个 `step_layer()` 使用相同 physical block/offset，但读取对应 layer cache；attention 使用 pending `effective_seq_lens`，committed `seq_len` 只在 `commit_step()` 增长一次。

## Backpressure 与原子性

Engine 在执行 RoPE/KV append 前根据每个 active request 的当前 `seq_len` 和 cache free block 数计算本轮所需的新 block。容量不足则返回 backpressure，不调用 cache append，因此不创建所有权、不增加 seq_len、不修改 lifecycle。

这与 PagedKVCache 的 capacity preflight 形成两层保护：Engine 提供可调度的结果，Cache 保持底层 allocator 的原子性。

## Backend 策略

- `append_backend="torch"` + `decode_backend="reference"`：默认、可在 CPU 上用于可读 correctness 测试。
- `append_backend="fused_cuda"` + `decode_backend="triton"`：RTX 5070 GPU path。append-only 三路径 benchmark 的 p50 几何平均为 1.2226x vs torch，因此 Engine 显式选择该组合，而不改变公开 RoPE API 的默认值。

## 机制边界

- single-layer `step()` 与 multi-layer sequential transaction 均为每个 request 每 step 一 token。
- Q/K/V 由调用方提供；不执行模型 forward、sampling、tokenizer 或 prompt/prefill forward。
- `prefill_request_layers()` 可以原子导入调用方已经构建的多层 prompt/context K/V；FlashDec 不负责从 token/model 生成这些 K/V。
- multi-layer transaction 支持 `append_backend="torch"` 与 `"fused_cuda"`；独立非 fused CUDA append 不进入 transaction。
- 不实现 priority、preemption、prefix 内容构建、admission-time prefix eviction 或 CPU offload。

Scheduler 不直接持有 physical blocks；它在 admission 时建立 lifetime logical commitment，再由 Cache 按 append 进度惰性分配。`apply_scheduler_decision()` 要求显式提供生成 decision 的 scheduler 与原始 snapshot；Engine 在 lifecycle mutation 前重建权威 snapshot，并用该 config 重跑 canonical policy 做 exact compare。详细状态所有权、deadlock 反例与 decision-binding 语义见 `docs/design_scheduler.md`。

Shared-prefix admission 保持同一所有权边界：Engine 从 Cache registry 验证 `RequestSpec.prefix_id`，admission 时挂载 immutable full blocks；Scheduler 只接收 derived `shared_prefix_blocks`，不接触 K/V 或 physical ids。global prefix residency 只计一次，每个 request 只承诺 private tail。完整定义见 `docs/design_shared_prefix_blocks.md`。

## 验证

CPU/reference tests 覆盖动态 admission、batch row order、finish/cancel、block reuse 与 backpressure 无 mutation。RTX 5070 额外覆盖：

```text
fused_cuda append + Triton paged decode == PyTorch paged reference
```

Multi-layer API 的 2/4-layer per-layer reference、异常自动 rollback、scheduler/open transaction 互斥和单层 compatibility tests，在 commit `a009b45` 的 RTX 5070 WSL 上通过 focused `71 passed, 8 subtests passed` 和完整回归 `322 passed, 20 subtests passed`。Fused CUDA location-only sequential path 在 commit `6afc89f` 通过 focused `131 passed` 和完整回归 `326 passed, 20 subtests passed`，覆盖 2-layer FP16/BF16、GQA、Triton 与 rollback。

Commit `fa0f89a` 的 144 行正式 multi-layer workload 已通过 matrix、pair trajectory、transaction、block、rollback、profiler、seed/order 校验。fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`；append device 为 `1.6103x`，decode device 为 `1.0024x`，说明系统收益主要来自 append 与 launch 减少，而不是 attention kernel 变化。证据提交 `67bee15` 的 RTX 5070 完整回归为 `337 passed, 25 subtests passed in 5.82s`，无 skipped 或 failure。详细结果见 `benchmarks/results/multi_layer_transaction_summary.md`。

Shared-prefix integration 在 commit `08d0414` 的 RTX 5070 WSL 通过 focused `56 passed, 14 subtests passed in 5.29s` 与完整 `352 passed, 25 subtests passed in 9.37s` 回归。性能协议只把 non-instrumented fixed-full-batch step 作为 decode latency，registration、attach 与 bounded-capacity admission 分别报告。

`submit_request()` 将从 resident registry 验证得到的 shared block 数缓存为 immutable request metadata。`scheduling_snapshot()`、commitment accounting 和 invariant 热路径使用该缓存，避免每个 step 重复构造 `prefix_state()` snapshot；active request 仍必须与 Cache authoritative `shared_block_count` 一致，Cache `state_version` 仍保护 resident set 不被外部静默修改。

Commit `fe72e27` 的 lookup-count targeted test 与 RTX focused/full regression 已通过。8-trial/64-row confirmation 的所有非零 Engine p50 paired range 都跨过 1；包括旧 BF16 75% 回退在内，没有形成新的稳定方向。该结果不证明跨 commit 因果 speedup，只验证 Engine correctness 与 metadata-cache invariant；完整 step 性能保持近中性并带有未归因离群波动。
