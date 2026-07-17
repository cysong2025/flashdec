# FlashDec 公开 API

本文给出 `flashdec` 顶层公开接口的用途、张量约定和状态边界。当前版本仍为 `0.0.0` release candidate；接口在 `v0.1.0` 发布前允许做兼容性整理，但所有行为变更都必须同步测试与 Changelog。

## 张量约定

| 对象 | Shape | 说明 |
| --- | --- | --- |
| Q | `[batch, num_q_heads, head_dim]` | 当前 decode token 的 query |
| K/V input | `[batch, num_kv_heads, head_dim]` | 当前 token 的 key/value |
| token-major cache | `[num_layers, num_blocks, num_kv_heads, block_size, head_dim]` | `PagedKVCache` physical storage |
| block tables | `[batch, max_blocks_per_seq]` | logical block 到 physical block 的映射 |
| seq_lens | `[batch]` | committed logical length |
| attention output | `[batch, num_q_heads, head_dim]` | 单 token attention output |

`num_q_heads` 必须能被 `num_kv_heads` 整除。Paged Triton kernel 支持 FP16/BF16、`head_dim=64/128` 和 `block_size=8/16/32`；默认性能配置是 token-major、block 32、2 warps 和 implicit stages。

## Attention

```python
flashdec.dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=None)
flashdec.paged_decode_attention_ref(
    q, k_cache, v_cache, block_tables, seq_lens,
    sm_scale=None, kv_layout="token_major",
)
flashdec.decode(
    q, k_cache, v_cache, block_tables, seq_lens,
    sm_scale=None, block_size=None, num_warps=2,
    kv_layout="token_major", num_stages=None,
)
```

- `dense_decode_attention_ref`：dense PyTorch correctness anchor。
- `paged_decode_attention_ref`：paged PyTorch correctness anchor，支持 token-major 与 dim-major。
- `decode`：`paged_decode_attention` 的顶层别名，调用 Triton kernel。

Reference 与 Triton 路径具有相同的 request-row、GQA/MQA 和有效长度语义。Triton 路径要求 CUDA tensor。

## Paged KV Cache

```python
cache = flashdec.PagedKVCache(
    num_layers=2,
    num_kv_heads=8,
    head_dim=128,
    block_size=32,
    max_blocks=256,
    dtype=torch.float16,
    device="cuda",
    prefix_cache_capacity_blocks=64,
)
```

主要接口：

- lifecycle：`add_request()`、`finish_request()`、`cancel_request()`、`request_state()`。
- shared prefix：`register_prefix()`、`attach_prefix()`、`prefix_state()`、`evict_prefix()`。
- metadata：`block_tables()`、`seq_lens_tensor()`、`request_block_ids()`。
- observability：`metrics()`、`validate_invariants()`。
- legacy single-layer append：`append()`、`append_cuda()`、`append_fused_cuda()`。
- multi-layer transaction：`begin_token()`、`write_token_layer()`、`write_token_layer_fused_cuda()`、`commit_token()`、`abort_token()`。

Cache 是 request seq_len、physical block ownership、shared-prefix residency 和 transaction 状态的唯一事实来源。容量失败必须发生在 mutation 前；finish/cancel 释放 request-private blocks 并减少 prefix reference；multi-layer transaction 只在全部 layer 完成后增长一次 seq_len。

R3-A shared-prefix API 接收已经构建的 immutable full blocks，shape 为 `[num_layers, num_prefix_blocks, num_kv_heads, block_size, head_dim]`。多个 request 可以共享同一组 physical ids，prefix 后的 tail 始终私有；无 active owner 的 prefix 才能被显式或 LRU 淘汰。`prefix_cache_capacity_blocks=0` 是默认值，表示关闭该功能。DecodeEngine/scheduler commitment integration 属于 R3-B，R3-A 会拒绝在已有 resident prefix 的 cache 上启用 scheduler-managed mode。完整所有权说明见[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)。

## RoPE 与 Append

`flashdec.apply_rope()` 提供 split-half rotary embedding reference，支持 FP16/BF16/FP32 和 partial `rotary_dim`。

```python
result = flashdec.rope_paged_kv_append(
    cache,
    layer_idx=0,
    request_ids=request_ids,
    q=q,
    k=k,
    v=v,
    rotary_dim=128,
    append_backend="torch",  # "torch" | "cuda" | "fused_cuda"
)
```

`result` 包含 rotated Q、pre-append positions、block tables 和 committed seq_lens。公开 helper 默认使用 torch 语义基线；GPU Engine 显式选择 fused CUDA。该 helper 属于 legacy single-layer append，multi-layer 路径由 Engine transaction 管理。

低层 `cuda_kv_append()` 与 `fused_rope_kv_append()` 只接收已经验证的 tensor、physical block ids 和 offsets，不拥有 allocator 或 request lifecycle。对应的 `load_*_extension()` 显式触发 lazy JIT，主要用于环境检查和测试；普通调用方应优先使用 Cache 或 Engine API。

## DecodeEngine

```python
engine = flashdec.DecodeEngine(
    cache,
    append_backend="fused_cuda",
    decode_backend="triton",
    num_warps=2,
)
```

Unscheduled lifecycle：

```text
add_request -> admit -> step -> finish_request | cancel_request
```

Scheduler-managed lifecycle：

```text
submit_request(RequestSpec)
  -> scheduling_snapshot
  -> BlockAwareScheduler.plan
  -> apply_scheduler_decision
  -> step
```

Multi-layer token：

```python
tx = engine.begin_step(request_ids)
for layer_idx, (q, k, v) in enumerate(zip(q_by_layer, k_by_layer, v_by_layer)):
    layer = engine.step_layer(tx, layer_idx, q, k, v)
result = engine.commit_step(tx)
```

Layer 必须按 `0..N-1` 顺序执行。`step_layer()` 的输入、写入或 decode 异常会自动 abort；显式放弃使用 `abort_step()`。Open transaction 期间不能并行修改 request lifecycle 或应用 scheduler decision。

## Scheduler

`RequestSpec` 固定 initial context、最大生成 token 数和 submission order；`SchedulerConfig` 固定 active/batch 上限、reserve blocks 与 aging threshold。`BlockAwareScheduler.plan(snapshot)` 是纯函数式规划步骤，不修改 Engine 或 Cache。

`SchedulerDecision` 必须由同一 `state_version` 的 Engine 应用。过期或伪造 decision 会在任何部分 mutation 前被拒绝。

## Workload

- `run_synthetic_workload()`：single-layer dynamic DecodeEngine workload。
- `run_scheduler_workload()`：有限 trace 下的 cancel、greedy 与 lifetime scheduler 对照。
- `WorkloadResult` / `SchedulerWorkloadResult`：complete-step latency、吞吐、完成率、公平性与内存指标。

Multi-layer 正式 workload 位于 `benchmarks/run_multi_layer_engine.py`，它是证据生成工具，不属于顶层 runtime API。

## 错误与边界

- 参数或状态错误使用 `TypeError`、`ValueError` 或 `RuntimeError`，不静默修正 request 轨迹。
- Legacy `DecodeEngine.step()` 只支持 `num_layers=1`；多层 Cache 必须使用 transaction API。
- Q/K/V 由调用方提供；API 不包含模型投影、prefill engine、sampling 或网络服务。
- CUDA extension 使用 lazy JIT；首次构建时间不得计入稳态 benchmark。

状态机和失败原子性的完整说明见[总体设计](design.md)、[DecodeEngine 设计](design_decode_engine.md)、[Multi-layer transaction 设计](design_multi_layer_kv_transaction.md)与[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)。
