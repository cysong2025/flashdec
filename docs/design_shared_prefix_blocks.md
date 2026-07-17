# Shared Prefix Blocks 设计

本文定义 R3 的共享前缀边界、所有权模型和分阶段验收标准。R3 只处理调用方已经构建好的 KV full blocks；tokenizer、模型 prefill、prefix 内容寻址和网络服务不在范围内。

## 1. 问题

多个请求可能包含完全相同的 system prompt 或固定上下文。当前 `PagedKVCache` 为每个请求独立分配 physical blocks，因此相同 KV 会被重复保存。R3 的目标是让这些请求共享不可变的完整块，同时保证：

- 一个请求结束不会破坏其他请求仍在读取的数据；
- shared prefix 后的 tail 始终由请求私有，不会覆盖共享内容；
- capacity failure 不产生部分注册、部分挂载或错误引用计数；
- physical memory、logical tokens 和节省量分别统计，不混用口径。

## 2. 输入与范围

调用方提供 opaque `prefix_id`，以及已经按缓存布局构建的 K/V：

```text
[num_layers, num_prefix_blocks, num_kv_heads, block_size, head_dim]
```

`num_prefix_blocks` 必须大于 0。每个输入块都是 full block，因此 prefix token 数固定为 `num_prefix_blocks * block_size`。注册时 K/V 必须与目标 cache 的 layer 数、KV head 数、head dim、block size、dtype 和 device 完全一致。

第一版明确不支持：

- partial tail block 共享；
- shared block 原地更新；
- 自动计算或验证两个 `prefix_id` 的内容是否相同；
- 跨 `PagedKVCache` 实例共享 physical storage。

## 3. 双层所有权

R3 将“块仍驻留在 prefix cache”与“当前有请求使用该 prefix”分开：

```text
Prefix registry residency
    └── 保证 prefix blocks 不进入 free list

Active request reference
    └── 保证 prefix 有 active owner 时不能被 LRU 淘汰
```

每个 prefix registry entry 保存：

- `prefix_id`；
- immutable physical block ids；
- `active_refcount`；
- LRU 顺序。

每个挂载 prefix 的 request 保存：

- `prefix_id`；
- block table 开头的 shared block 数；
- prefix 之后的 request-private tail blocks。

请求 finish/cancel 时只释放 private tail，并将 `active_refcount` 减一。prefix blocks 继续驻留，直到没有 active owner 且被显式或自动 LRU eviction。这样“最后一个请求结束”和“物理块回到 free list”是两个独立事件。

## 4. 状态转换

### 注册

```text
missing prefix_id + valid full blocks
  -> capacity preflight
  -> evict inactive LRU entries when necessary
  -> allocate physical blocks
  -> copy K/V
  -> resident prefix(active_refcount=0)
```

重复 `prefix_id` 被拒绝，避免在同一个 id 下静默替换内容。容量不足必须在 registry、free list 和 request ownership 发生变化前报错。

### 挂载

```text
empty active request + resident prefix
  -> request.block_ids starts with prefix.block_ids
  -> request.seq_len = prefix_blocks * block_size
  -> prefix.active_refcount += 1
```

只允许空 request 挂载一次。由于 prefix 结束位置总在 block boundary，下一次 append 必然分配 request-private block，不需要 copy-on-write。

### 关闭与淘汰

```text
finish/cancel request
  -> release private tail
  -> decrement active_refcount

evict inactive prefix
  -> remove registry entry
  -> return prefix blocks to deterministic free list
```

有 active owner 的 prefix 不能被淘汰。自动 LRU 只选择 `active_refcount == 0` 的 entry。

## 5. 容量与统计口径

`prefix_cache_capacity_blocks` 是共享前缀可占用的 physical block 上限；默认值为 0，表示功能关闭，保持现有缓存行为。prefix blocks 与普通 request blocks 使用同一个 `max_blocks` 物理池。

指标分为三类：

- physical residency：resident prefixes、resident prefix blocks、allocated KV bytes；
- logical use：active request tokens、active prefix references；
- reuse benefit：hit/miss、shared physical blocks、saved blocks/bytes、evictions。

`saved_blocks` 定义为：

```text
sum((active_refcount - 1) * prefix_block_count)
```

它表示相对于每个请求各存一份 prefix，当前避免的重复 physical blocks。内部碎片按唯一 physical data 计算，不能用包含重复共享 token 的 logical active tokens 直接相减。

## 6. 不变量

`validate_invariants()` 必须检查：

1. free blocks、resident prefix blocks 和 request-private blocks 覆盖完整物理池，且互不重叠；
2. 一个 physical block 最多属于一个 prefix entry；
3. 多个 request 只能通过同一个 prefix entry 重复引用 shared blocks；
4. request 的 shared block 前缀与 registry entry 完全一致；
5. registry `active_refcount` 等于实际挂载的 active request 数；
6. terminal request 不保留 shared 或 private ownership；
7. open token transaction 只能新增或回滚 private tail block。

## 7. 分阶段实现

### R3-A：Cache ownership core

- `register_prefix()`、`attach_prefix()`、`evict_prefix()`、`prefix_state()`；
- full-block 校验、引用计数、inactive LRU 和容量失败原子性；
- shared-aware metrics 与 invariant validation；
- CPU/reference correctness tests。

验证状态：2026-07-17 的 WSL focused 与完整回归均报告通过。本轮未提供精确通过数量，因此只记录通过事实，不形成新的定量回归基线。

### R3-B：DecodeEngine 与 scheduler integration

`RequestSpec` 增加 opaque `prefix_id`。调用方不能提供 shared block 数；DecodeEngine 从 Cache registry 派生 prefix 长度，并要求：

```text
prefix.token_count == RequestSpec.initial_context_tokens
```

因此 R3-B 的 prefix 必须覆盖完整 initial context。admission 时 Engine 执行 `add_request + attach_prefix`，不再走逐 token prefill。`SchedulingSnapshot` 分别携带全局 `resident_prefix_blocks` 和每个 request 的 `shared_prefix_blocks`。

容量口径调整为：

```text
resident_prefix_blocks
  + sum(active request-private lifetime commitments)
  <= max_blocks - reserve_blocks

used physical blocks
  = resident_prefix_blocks
  + sum(active request-private physical blocks)
```

同一 prefix 被多个 request 引用时，resident blocks 只计一次；每个 request 的 future decode tail 独立 commitment。`needed_physical_blocks_now` 只统计 admission 后尚未存在的 private context blocks，以及本轮跨 boundary 的 private tail blocks。

prefix 必须在 request submission 前通过 `DecodeEngine.register_prefix()` 注册，或在构造 Engine 前已经存在于 Cache。scheduler-managed mode 启动后，外部 register/evict/attach 会使 Cache version 与 Engine snapshot 不一致并被拒绝。第一版不会为了 admission 主动淘汰 inactive prefix；scheduler 将当前 resident set 视为固定物理占用。

验证状态：2026-07-17 RTX 5070 WSL focused 为 `56 passed, 14 subtests passed in 5.29s`，完整回归为 `352 passed, 25 subtests passed in 9.37s`。R3-B admission、commitment 与 stale-mutation 语义由此冻结。

### R3-C：Benchmark 与 RTX evidence

固定 request 数和 context，构造 0%/25%/50%/75% hit rate，报告 physical blocks/bytes、saved blocks/bytes、admission、lookup/attach latency、decode latency 和 eviction count。验收重点是显存节省与所有权正确性，不宣称共享 prefix 会加速 attention kernel。

runner 将实验分为两个互不混淆的探针：

- fixed bounded capacity：所有 hit-rate 使用相同小容量，只记录第一次调度可接纳的请求数和 lifetime commitment；
- fixed full batch：使用足以容纳无共享基线的相同容量，保证四档 hit-rate 的 decode batch 完全一致，再比较物理块、字节和完整 step 延迟。

shared 与 private request 使用相同 K/V context。正式计时前逐行 materialize 并对齐；计时后再次检查 resident prefix 内容未被 private tail 改写。CSV validator 严格验证 matrix、trial 轮转、seed、容量单调性、block/byte 公式、prefix lifecycle 与最终 cleanup。

commit `fd36ed0` 的 RTX 5070 FP16 quick 共 4 行，全部通过严格校验：

| hit rate | admitted | context physical/logical | context saved | peak blocks | saved MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 2/4 | 4/4 | 0% | 8 | 0.000 |
| 25% | 2/4 | 4/4 | 0% | 8 | 0.000 |
| 50% | 3/4 | 3/4 | 25% | 7 | 0.125 |
| 75% | 3/4 | 2/4 | 50% | 6 | 0.250 |

25% 只有一个 hit request，没有第二个 owner 可以复用，因此相对私有存储没有净节省。50%/75% 从第二个 owner 开始分别节省 1/2 个 context blocks，并在相同 bounded pool 下多接纳一个请求。quick 每档只有 3 次正式 step 采样，p50/TPS 非单调，只验证链路，不形成 latency 结论。FP16/BF16 三轮正式证据待执行。

commit `1d5d8d0` 的 RTX 5070 正式矩阵覆盖 4 hit rates、2 dtypes、3 trials，共 24 行并全部通过严格校验。两种 dtype 的 block/byte 结果一致：

| hit rate | admitted | context physical/logical | context saved | peak blocks | peak reduction vs 0% | saved MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 9/16 | 64/64 | 0.0% | 80 | 0.0% | 0.000 |
| 25% | 12/16 | 52/64 | 18.8% | 68 | 15.0% | 1.500 |
| 50% | 15/16 | 36/64 | 43.8% | 52 | 35.0% | 3.500 |
| 75% | 16/16 | 20/64 | 68.8% | 36 | 55.0% | 5.500 |

context saving 只计算重复 prefix；peak blocks 还包含每个请求不可共享的 private decode tail，因此 75% 的 context saving 是 68.8%，完整 peak reduction 是 55.0%。prefix attach p50 在所有非零 hit-rate case 中低于 `0.8 us`，相对约 `1.6-2.0 ms` 的 complete step 很小。跨轮中位 latency 对 hit rate 不单调，FP16/BF16 方向也不一致；在 paired trial range 完成归档前不形成 latency 收益结论。

## 8. 验收测试

```text
two requests attach same prefix -> physical ids shared
one request finishes -> other request dense data unchanged
private tail append -> shared K/V unchanged
last owner finishes -> prefix remains resident
inactive eviction -> blocks return and are reusable
active prefix eviction -> rejected without mutation
capacity failure -> registry/refcount/free list unchanged
multi-layer prefix -> every layer materializes correctly
```
