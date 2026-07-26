# Integrated Scheduled Multi-layer Workload

## 1. 研究问题与范围

这项 workload 研究 scheduler、multi-layer token transaction、shared prefix 和 trusted fused append 在同一动态 request trajectory 中组合后，状态所有权、rollback 和 block reuse 是否仍能由独立 reference 完整验证。受测执行配置为：

- `append_backend="fused_cuda"`；
- `decode_backend="triton"`；
- Cache-owned trusted dispatch；
- `metadata_policy="materialized"`；persistent metadata candidate 未通过预注册稳定性门，不进入该组合路径；
- token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。

Workload 只使用运行前注册的固定 resident prefix。在线 prefix registration/eviction、CUDA Graph、kernel 参数 sweep、完整模型 forward、sampling 和网络服务都不在研究范围内。

## 2. 组合执行链

```text
dynamic arrivals
  -> lifetime scheduler admission/defer
  -> fixed resident prefix attach or caller-supplied private context
  -> begin_step
  -> step_layer(0..L-1)
  -> commit_step or automatic abort
  -> finish/cancel
  -> released private block reuse
  -> terminal inactive-prefix eviction and zero-used cleanup
```

Shared-prefix hit 在 admission 时只增加引用，不复制 prefix physical blocks。Private miss 通过 `DecodeEngine.prefill_request_layers()` 逐 token 提交调用方提供的 `[num_layers, num_kv_heads, head_dim]` K/V；每个 prompt token 只有所有 layer 写入成功后才发布一次 `seq_len`。任一层失败时复用 Cache transaction rollback，不能留下部分 layer、pending token 或 boundary block。

固定 prefix 在所有 request lifecycle 结束前禁止清理。`DecodeEngine.evict_prefix()` 只允许在 waiting/active request 都归零后调用，并同步 Engine/Cache version；这避免 benchmark 绕过 scheduler-managed Engine 直接修改 Cache。

## 3. 标准 trace 与 reference trajectory

标准 trace 包含四个请求：

| request | 到达 step | context | decode tokens | prefix | terminal |
| --- | ---: | ---: | ---: | --- | --- |
| `hit-a` | 0 | 64/128 | 4 | fixed resident hit | finish |
| `miss-a` | 0 | 64/128 | 3 | private miss | finish |
| `hit-cancel` | 1 | 64/128 | 5 | fixed resident hit | step 3 cancel |
| `miss-b` | 2 | 64/128 | 5 | private miss | finish |

Scheduler 使用 `max_active_requests=3`、`max_batch_requests=2` 和 FIFO + aging。step 4 在 layer 1 注入 shape error：layer 0 已写入 pending token，layer 1 失败后必须自动 abort；下一 step 的 position 必须与失败前相同。前一批 finish/cancel 释放的 private block 必须被后续 `miss-b` context 或 decode boundary 再次分配。

`build_integrated_reference()` 不导入 Torch，只使用 immutable request metadata、纯 Scheduler planner 和逻辑 block accounting 生成逐 step reference：

- submitted/cancelled/admitted/rejected ids；
- runnable/deferred/completed ids；
- transaction positions 与 abort 标志；
- committed/used/free block trajectory。

reference 与实际执行分别序列化为 canonical JSON，并计算 SHA-256。runner 在每一步先比较具体字段，终态再要求两个独立 digest 完全一致。标准 64-token trace 为 10 个 logical steps、9 个成功 token transaction、1 个 abort、13 个完成 request-token；完成顺序为 `miss-a -> hit-a -> miss-b`，`hit-cancel` 单独进入 cancelled 集合。

## 4. 证据 schema

每个 CSV row 是一个完整 trace，不是单一 token 或单一 backend pair。strict validator 重新构建 dependency-free reference，并检查：

- case/dtype/trial 矩阵、连续 seed 与轮转 case order；
- frozen backend 与 `materialized` metadata policy；
- reference/observed trajectory digest；
- completion、cancellation、rollback 与 block reuse；
- transaction begin/commit/abort/layer-write 计数；
- prefix registration/hit/eviction 计数；
- peak blocks/bytes、terminal resident-only 状态和 eviction 后 `final_used_blocks=0`；
- 所有 latency/TPS 字段正且有限。

两个 private miss 各写入 `context_tokens` 个 multi-layer prompt token。标准 failure 位于 layer 1，因此预期计数为：

```text
Cache begin   = 2 * context_tokens + 9 successful decode + 1 abort
Cache commit  = 2 * context_tokens + 9 successful decode
Cache abort   = 1
Cache writes  = (2 * context_tokens + 9) * num_layers + 1 pre-failure layer
Engine layers = 9 * num_layers + 1 pre-failure layer
```

Random prefix、prompt 和 decode tensor 在计时前构建。complete logical-step wall 是 scheduler、private context writes、fused transaction/decode 与 finish/cancel 的和；prefix registration 和最终 inactive-prefix eviction 不进入 latency，但都进入 correctness/accounting gate。

## 5. CUDA 矩阵与验证规则

正式矩阵为：

```text
2 layer counts x 2 contexts x 2 dtypes x 3 trials = 24 rows
layers: 2, 4
contexts: 64, 128
dtypes: FP16, BF16
```

每轮将四个 case 轮转一次，seed 连续。该协议不设置 shared-prefix latency speedup 门，也不与 rejected persistent metadata candidate 相除；必须满足：

1. focused/full correctness 无 failure；
2. 24-row strict validator 完整通过；
3. 每 row trajectory、rollback、reuse、prefix lifecycle 与 zero-used cleanup 通过；
4. p50/p90/p99/TPS 报告绝对值和跨 trial `[min,max]`，不把有限 trace 的 p99 包装成稳定尾延迟；
5. 失败时保留原始 CSV/log，不通过删行、补值或增加非预注册 retry 修饰结果。

## 6. RTX 5070 命令

先运行 quick：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case l2_c32 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/integrated_runtime_lifecycle_quick.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_quick.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_quick_summary.md \
  --expected-trials 1 \
  --expected-cases l2_c32 \
  --expected-dtypes float16
```

quick 与 focused correctness 通过后运行正式矩阵：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/integrated_runtime_lifecycle_trials3.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_trials3.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_summary.md \
  --expected-trials 3
```

Commit `6912894` 的 RTX 5070 证据包括 focused `60 passed, 17 subtests passed`、full `425 passed, 57 subtests passed`，以及通过 strict validator 的 FP16 quick 和预注册 24-row/3-trial FP16/BF16 正式矩阵。Canonical evidence 见[Integrated workload 正式摘要](../benchmarks/results/integrated_runtime_lifecycle_summary.md)。
