# FlashDec 公开 API

本文给出 `flashdec` 顶层公开接口的用途、张量约定和状态边界。`0.0.0` 表示研究原型 API：文档中的行为有测试约束，但不承诺语义版本稳定性；任何接口行为变更仍必须同步测试与 Changelog。

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

`begin_token()` 返回的是 detached snapshot。Cache 在 host allocator 完成 reservation 后会验证 position、block offset、physical block id 与 request block table 的一致性；`write_token_layer_fused_cuda()` 根据 transaction id 回查该内部状态，不使用调用方可修改的 snapshot 位置 tensor。这样 Cache-owned fused path 可以跳过 raw CUDA index 的 device reduction + `.item()`，但公开低层 `fused_rope_kv_append()` 仍保留完整的值域检查。private trusted raw primitive 不属于 `flashdec` 顶层 API。

只有 open transaction 会在 Cache 内保留完整 allocator/device metadata。`commit_token()` / `abort_token()` 先成功构造返回 view，再发布或回滚状态，并立即删除完整内部 state；Cache 仅保留最多 256 条 `transaction_id/cache_version/request_ids/state` 轻量 tombstone，用于近期 double commit/abort 的明确诊断。超过上限的旧 handle 变为 `unknown token transaction`；`transaction_view()` 只接受 open handle，终态 snapshot 由 commit/abort 的返回值提供。调用方自己长期保存 detached view 仍会保存其中 tensor，这不属于 Cache 内部 retention。

`metrics()` 通过 `retained_transaction_state_count`、`terminal_transaction_history_count` 和 `terminal_transaction_history_limit` 暴露该边界；`validate_invariants()` 要求完整 transaction state 至多一个且只能为 open，终态历史不得超过固定上限。

Shared-prefix API 接收已经构建的 immutable full blocks，shape 为 `[num_layers, num_prefix_blocks, num_kv_heads, block_size, head_dim]`。多个 request 可以共享同一组 physical ids，prefix 后的 tail 始终私有；无 active owner 的 prefix 才能被显式或 LRU 淘汰。`prefix_cache_capacity_blocks=0` 是默认值，表示关闭该功能。完整所有权说明见[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)。

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

低层 `cuda_kv_append()` 与 `fused_rope_kv_append()` 接收调用方提供的 tensor、physical block ids 和 offsets，自行执行结构与索引值检查，但不拥有 allocator 或 request lifecycle。对应的 `load_*_extension()` 显式触发 lazy JIT，主要用于环境检查和测试；普通调用方应优先使用 Cache 或 Engine API。

## DecodeEngine

```python
engine = flashdec.DecodeEngine(
    cache,
    append_backend="fused_cuda",
    decode_backend="triton",
    num_warps=2,
)
```

Scheduler-managed prefix：

```python
engine.register_prefix("system-v1", prefix_k_blocks, prefix_v_blocks)
engine.submit_request(
    flashdec.RequestSpec(
        request_id="request-1",
        initial_context_tokens=prefix_k_blocks.shape[1] * cache.block_size,
        max_new_tokens=128,
        submission_order=0,
        prefix_id="system-v1",
    )
)
```

prefix 必须在任何 request submission 之前 resident，并覆盖完整 `initial_context_tokens`。Snapshot 将 prefix residency 作为全局物理占用计一次，将每个 request 的 decode tail 作为 private lifetime commitment。scheduler-managed mode 启动后不能绕过 Engine 修改 prefix registry。

Private miss 的 caller-supplied multi-layer prompt token：

```python
# k_layers/v_layers: [num_layers, num_kv_heads, head_dim]
engine.prefill_request_layers(request_id, k_layers, v_layers)
```

该调用为一个 prompt token 建立 Cache transaction，按 layer 顺序写入，并只在全部成功后发布一次 `seq_len`；失败会自动 rollback。所有 request 进入 finished/cancelled/rejected terminal state 后，可用 `engine.evict_prefix(prefix_id)` 清理 inactive fixed prefix；active workload 中禁止调用。

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

`RequestSpec` 固定 initial context、最大生成 token 数、submission order 和可选 opaque `prefix_id`；`SchedulerConfig` 固定 active/batch 上限、reserve blocks 与 aging threshold。`BlockAwareScheduler.plan(snapshot)` 是纯函数式规划步骤，不修改 Engine 或 Cache。

```python
snapshot = engine.scheduling_snapshot(
    logical_step,
    waiting_wait_steps=wait_steps,
    waiting_skip_counts=skip_counts,
    active_service_wait_steps=service_wait_steps,
)
decision = scheduler.plan(snapshot)
engine.apply_scheduler_decision(
    decision,
    scheduler=scheduler,
    snapshot=snapshot,
)
```

`SchedulerDecision` 携带生成它的完整 K/V-free metadata `SchedulingSnapshot` 与 `SchedulerConfig`。应用时必须显式提供同一 value-equal snapshot 和 config-equal scheduler；Engine 会保留调用方负责的 fairness counters，同时从当前 Engine/Cache 重建所有权威字段，再用 `BlockAwareScheduler(scheduler.config)` 重跑 canonical policy 并 exact compare。stale、错 snapshot/config、被修改 decision，以及被覆写 scheduler 产生的非 canonical decision，都会在任何 admission/rejection mutation 前被拒绝。这是单次 apply 的完整性约束，不是线程安全或不可信进程的安全认证；该 API 变更不兼容旧的单参数调用。

## Workload

- `run_synthetic_workload()`：single-layer dynamic DecodeEngine workload。
- `run_scheduler_workload()`：有限 trace 下的 cancel、greedy 与 lifetime scheduler 对照。
- `run_integrated_workload()`：dynamic arrival、mixed prefix、multi-layer transaction、rollback 与 cleanup reference trace。
- `WorkloadResult` / `SchedulerWorkloadResult` / `IntegratedWorkloadResult`：complete-step latency、吞吐、完成率、公平性、轨迹与内存指标。

Multi-layer 正式 workload 位于 `benchmarks/run_multi_layer_engine.py`，它是证据生成工具，不属于顶层 runtime API。

## 可选 vLLM integration

`flashdec.vllm_plugin` 通过 vLLM general-plugin entry point 注册 `CUSTOM` attention backend。它不是 `import flashdec` 顶层 API，也不会让核心包强制 import vLLM。使用固定兼容环境安装后，通过配置选择：

```bash
export VLLM_PLUGINS=flashdec
vllm serve /home/<user>/models/Qwen2.5-3B-Instruct \
  --attention-backend CUSTOM
```

vLLM 继续拥有 KV cache、metadata、prefill、model runner、scheduler 和 API server；FlashDec 只处理 eligible single-token decoder attention，其他调用回退原生 Triton。支持条件、WSL 环境变量、split policy 和版本边界见 [vLLM backend 设计](design_vllm_backend.md)。

## 错误与边界

- 参数或状态错误使用 `TypeError`、`ValueError` 或 `RuntimeError`，不静默修正 request 轨迹。
- Legacy `DecodeEngine.step()` 只支持 `num_layers=1`；多层 Cache 必须使用 transaction API。
- Q/K/V 由调用方提供；`prefill_request_layers()` 只提交已构建的 prompt K/V，不包含模型投影、prompt 内容构建、sampling 或网络服务。
- CUDA extension 使用 lazy JIT；首次构建时间不得计入稳态 benchmark。

状态机和失败原子性的完整说明见[总体设计](design.md)、[DecodeEngine 设计](design_decode_engine.md)、[Multi-layer transaction 设计](design_multi_layer_kv_transaction.md)、[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)与[组合 workload 设计](design_integrated_scheduled_multi_layer.md)。
