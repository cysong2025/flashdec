# DecodeEngine v1 设计说明

## 目标

`DecodeEngine` 是 FlashDec 从数据路径走向 AI Infra runtime 的第一层执行编排。它固定验证单 layer、单 token decode，不处理模型投影、采样、prefill 或网络服务。

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

## Backpressure 与原子性

Engine 在执行 RoPE/KV append 前根据每个 active request 的当前 `seq_len` 和 cache free block 数计算本轮所需的新 block。容量不足则返回 backpressure，不调用 cache append，因此不创建所有权、不增加 seq_len、不修改 lifecycle。

这与 PagedKVCache 的 capacity preflight 形成两层保护：Engine 提供可调度的结果，Cache 保持底层 allocator 的原子性。

## Backend 策略

- `append_backend="torch"` + `decode_backend="reference"`：默认、可在 CPU 上用于可读 correctness 测试。
- `append_backend="fused_cuda"` + `decode_backend="triton"`：RTX 5070 GPU path。Week 11 append-only benchmark 的 p50 几何平均为 1.2226x vs torch，因此 Engine 显式选择该组合，而不改变公开 RoPE API 的默认值。

## 当前限制

- 仅单 layer、每个 request 每 step 一 token。
- Q/K/V 由调用方提供；不执行模型 forward、sampling 或 prefill。
- admission 不预留 physical block；资源不足在 `step()` 以 backpressure 表示。
- 不实现 continuous batching scheduler、priority、preemption、prefix cache 或 CPU offload。

后续 Scheduler v2 不会让 Scheduler 直接持有 physical blocks，而是在 admission 时建立 lifetime logical commitment，再由 Cache 按 append 进度惰性分配。详细状态所有权、deadlock 反例与 stale-decision 语义见 `docs/design_scheduler.md`。

## 验证

CPU/reference tests 覆盖动态 admission、batch row order、finish/cancel、block reuse 与 backpressure 无 mutation。RTX 5070 额外覆盖：

```text
fused_cuda append + Triton paged decode == PyTorch paged reference
```

验证命令将在 Engine 提交后记录到 Week 11 状态文档。
