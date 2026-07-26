# Block-aware Scheduler 设计

## 1. 目标与范围

Block-aware Scheduler 在有限 KV 容量下做可解释的 admission 与 runnable-set 决策。它只回答三个问题：

1. waiting request 何时可以进入 active 集合。
2. active request 中哪些 request 在本轮执行一个 decode token。
3. 容量不足、batch 有上限或请求长期等待时，如何给出确定且可观测的决策。

Scheduler 不生成 Q/K/V，不调用 Triton/CUDA kernel，也不直接分配或释放 physical block。`DecodeEngine` 负责执行已接受的决策，`PagedKVCache` 继续拥有 physical block 与 KV 数据。

调度单位固定为单 GPU 上每个 request 每次执行一个 token；layer 数不改变 token-level commitment。priority API、swap/offload、生产级抢占和多线程并发不在本文范围，shared prefix 的容量扩展见第 15 节。

## 2. 为什么不能只看“本 step 需要几个 block”

仅根据当前 `free_blocks` 选择 runnable subset，不能保证系统最终有进展。考虑：

```text
block_size = 32
max_blocks = 16
active requests = 16
每个 request 已持有 1 个 block，seq_len = 32
free_blocks = 0
每个 request 的下一个 token 都需要新 block
```

此时任意 request 都无法跨过 block boundary，runnable subset 为空；又因为没有 request 能继续执行到完成，也没有 block 会被释放。系统进入资源死锁。aging 只能改变选择顺序，不能创造容量。

因此默认策略不能把“下一步能不能执行”当作唯一 admission 条件。它必须在 admission 时为 active request 的完整剩余生命周期承诺容量。

## 3. 两种不同的 block 概念

### 3.1 Logical commitment

请求提交时必须给出确定的 `initial_context_tokens` 和 `max_new_tokens`。其生命周期最大 block 数为：

```text
request_total_blocks = ceil(
    (initial_context_tokens + max_new_tokens) / block_size
)
```

请求只有在以下条件成立时才能 admission：

```text
sum(active_request_commitments)
    + request_total_blocks
    <= max_blocks - reserve_blocks
```

commitment 是调度器账本中的逻辑容量承诺，不对应预先选定的 physical block id，也不要求 admission 时立即分配全部显存块。

### 3.2 Physical allocation

physical allocation 仍由 `PagedKVCache` 按实际 prefill/append 进度惰性执行。对任一 active request 都应满足：

```text
physical_blocks_owned <= committed_blocks
```

全局满足：

```text
physical_blocks_used <= committed_blocks <= schedulable_blocks
```

这种分离保留 Paged KV 的按需分配与复用，同时保证一个已 admission 的有限请求能够走到完成，不需要依赖强制取消回收容量。

### 3.3 设计取舍

生命周期 commitment 是保守策略：如果请求提前结束，一部分承诺容量可能从未物理分配；与按 step 贪心 admission 相比，active concurrency 也可能降低。因此实验同时报告 committed utilization、physical utilization、等待时间和完成吞吐。

## 4. 数据模型

实现位于 `flashdec/scheduler.py`，核心类型如下：

```python
@dataclass(frozen=True)
class RequestSpec:
    request_id: Hashable
    initial_context_tokens: int
    max_new_tokens: int
    submission_order: int


@dataclass(frozen=True)
class WaitingRequestMetadata:
    spec: RequestSpec
    wait_steps: int = 0
    skip_count: int = 0


@dataclass(frozen=True)
class SchedulerConfig:
    max_active_requests: int
    max_batch_requests: int
    reserve_blocks: int = 0
    aging_threshold_steps: int = 8
    policy: str = "lifetime_fifo_aging"


@dataclass(frozen=True)
class SchedulingSnapshot:
    state_version: int
    logical_step: int
    block_size: int
    max_blocks: int
    free_blocks: int
    waiting: tuple[WaitingRequestMetadata, ...]
    active: tuple[ActiveRequestMetadata, ...]


@dataclass(frozen=True)
class SchedulerDecision:
    snapshot_version: int
    admit_ids: tuple[Hashable, ...]
    runnable_ids: tuple[Hashable, ...]
    deferred_ids: tuple[Hashable, ...]
    waiting_ids: tuple[Hashable, ...]
    rejected_ids: tuple[Hashable, ...]
    committed_blocks_before: int
    committed_blocks_after: int
    needed_physical_blocks_now: int
    free_blocks_before_step: int
    drain_for_request_id: Hashable | None
    reasons: tuple[str, ...]
```

`ActiveRequestMetadata` 至少包含 request spec、当前 `seq_len`、剩余 decode token、已持有 block、commitment 和 service-wait 计数。`deferred_ids` 只表示已 active 但本轮未运行的请求；未 admission 的请求由 `waiting_ids` 表示，永久无法放入 schedulable capacity 的请求由 `rejected_ids` 表示。字段只描述调度所需元数据，不暴露 K/V tensor。

## 5. 状态与所有权

| 组件 | 拥有的状态 | 不允许拥有 |
| --- | --- | --- |
| Scheduler | admission/runnable policy、wait/skip/service 计数 | K/V tensor、physical block id、Engine lifecycle mutation |
| DecodeEngine | immutable request spec、waiting/active/terminal 状态、commitment 派生、稳定 row mapping、`state_version`、决策执行 | allocator free list |
| PagedKVCache | physical block、request block list、seq_len、KV storage、allocator invariant、ownership `state_version` | admission/fairness policy |
| Workload/Caller | Q/K/V 输入、arrival、请求 token budget | 直接伪造 scheduler/cache 内部状态 |

Scheduler 的 `plan(snapshot)` 不修改 Engine 或 Cache。只有 Engine 接受并应用 decision 后，Scheduler 才通过明确的 outcome 更新 commitment 与 fairness 计数。

## 6. 单线程一致性与 stale decision

执行模型使用单线程 event loop，但仍需防止“基于旧容量做出的决策”被延迟执行：

- Engine 维护单调递增的 `state_version`。
- Cache 维护独立 ownership/seq_len `state_version`；scheduler-managed Engine 记录最后一次已观察版本，直接绕过 Engine 修改 Cache 会使 decision 失效。
- submit、admit、成功 append、finish、cancel 都递增版本。
- `SchedulingSnapshot` 和 `SchedulerDecision` 携带版本。
- Engine 应用 decision 前必须验证 `decision.snapshot_version == engine.state_version`。
- 版本不匹配时拒绝 decision，记录 `stale_decision_count`，重新 snapshot/plan；不能尝试部分执行。

Scheduler-managed 接口为：

```python
engine.submit_request(spec)
snapshot = engine.scheduling_snapshot(
    logical_step,
    waiting_wait_steps=...,
    waiting_skip_counts=...,
    active_service_wait_steps=...,
)
decision = scheduler.plan(snapshot)
engine.apply_scheduler_decision(decision)
```

旧 `add_request()` / `admit()` / `step()` 继续作为 unscheduled compatibility path；同一个 Engine 不能混用两种 submission 模式。scheduler-managed step 必须严格使用最后一次已应用 decision 的 `runnable_ids`。永久超过容量的 request 进入 Engine-only `rejected` 终态，不创建 Cache ownership。

带 initial context 的新 admission 先通过 `prefill_request()` 在正式 decode 计时外写入 prompt K/V；prefill 会递增版本并清除 pending decision，因此必须重新 snapshot/plan 后才能执行第一个 decode token。

这不是线程安全承诺。多线程锁、异步 CUDA stream ownership 和分布式状态一致性不在 v2 范围。

## 7. 默认算法：lifetime FIFO + aging

每个逻辑 step 按以下顺序执行。

### 7.1 清理与对账

1. 根据 Engine terminal 状态释放对应 commitment。
2. 检查所有 active request 的 physical ownership 不超过 commitment。
3. 计算 `schedulable_blocks = max_blocks - reserve_blocks`。

### 7.2 Admission

1. waiting queue 以稳定 `submission_order` 为基础。
2. 只有完整 lifetime commitment 能放入剩余 committed capacity 的请求才可 admission。
3. 可跳过暂时放不下的队首请求，让较小请求利用空余容量。
4. 每次跳过增加 wait/skip 计数。
5. 当某请求达到 `aging_threshold_steps` 后进入 drain barrier：暂停 admission 会进一步占用其所需容量的年轻请求，让现有 active 集合自然完成并释放 commitment。
6. 单请求 commitment 大于 `schedulable_blocks` 时立即报告 `request_exceeds_capacity`，不能让它永久停留在 waiting queue。

aging 不能抢占 active request，但对有限 workload，它能阻止年轻小请求无限绕过较大的老请求。

### 7.3 Runnable batch

1. 从 active request 中选择最多 `max_batch_requests` 个 request。
2. 优先选择 service wait 最大的 request，再以 submission order 打破并列。
3. 未选中的 active request 进入 `deferred_ids` 并增加 service wait。
4. default lifetime policy 下，跨 block boundary 所需容量已由 commitment 保证；`needed_physical_blocks_now` 仍用于 invariant 和观测，而不是重新进行一次过量 admission。
5. 如果 commitment invariant 成立但本 step 仍出现 physical backpressure，应将其视为实现错误或外部 allocator mutation，不能静默取消 request。

### 7.4 应用结果

1. Engine 验证 snapshot version。
2. 按 decision admission，并立即为 workload request 执行既有 initial-context seeding。
3. Caller 仅为 `runnable_ids` 生成/选择对应 Q/K/V rows。
4. Engine 执行一个 token 的 append -> paged decode。
5. finish/cancel 后立即释放 physical blocks 与 commitment。
6. Scheduler 根据实际 outcome 更新 wait/service 计数；失败路径不能保留虚假的 admission commitment。

Scheduler 不实现多 token prefill kernel。Synthetic workload 的 context seeding 位于正式 decode-step 计时边界之外。

## 8. 对照策略

Benchmark protocol 保留三种可解释策略：

| policy | 行为 | 用途 |
| --- | --- | --- |
| `cancel_on_backpressure` | 全 active batch 失败后取消最老请求 | 对照取消恢复进展的代价 |
| `greedy_step_only` | 只看本 step block 需求，不做 lifetime commitment | 展示更高并发及潜在 deadlock/stall |
| `lifetime_fifo_aging` | 默认；完整生命周期 commitment + 公平 runnable batch | 容量安全与进展策略 |

`greedy_step_only` 必须检测连续零进展并输出 `resource_deadlock`/`stalled_steps`，不能让 benchmark 无限循环。

## 9. 正确性 invariant

每次 plan、apply 和 lifecycle transition 后至少检查：

```text
0 <= reserve_blocks < max_blocks
sum(active commitments) <= max_blocks - reserve_blocks
per-request physical blocks <= per-request commitment
used physical blocks <= sum(active commitments)
terminal request has no commitment and owns no physical block
waiting request owns no physical block
decision does not mutate Engine/Cache
decision version matches before apply
request/order decisions are deterministic for the same snapshot/config
```

默认 policy 还必须证明：对所有 `request_total_blocks <= schedulable_blocks` 的有限请求集合，在没有调用方取消且请求最终停止到达时，每个请求最终完成。

## 10. 指标

除 step latency、tokens/s 和 cache metrics 外，还记录：

- waiting queue mean/max depth。
- admission wait steps p50/p90/max。
- runnable/deferred request 数。
- per-request 最大 service wait。
- committed blocks、committed-but-unallocated blocks。
- commitment utilization、physical utilization 及二者差值。
- scheduler decision p50/p90 wall time。
- stale decisions、stalled steps、resource deadlocks。
- completed requests/tokens、forced cancellations、backpressure steps。

性能表必须同时报告完成率、公平性和资源效率，不能只用 p50 或瞬时 active batch 判断策略优劣。

## 11. 测试覆盖

### 11.1 纯 Scheduler 测试

```text
oversubscribed waiting queue -> capacity-safe admission
single request larger than schedulable capacity -> explicit rejection
same snapshot/config -> deterministic decision
plan(snapshot) -> no Engine/Cache mutation
max_batch limit -> fair runnable/deferred rotation
aged large request -> younger admissions stop until capacity drains
stale version -> decision rejected without partial apply
```

### 11.2 Engine/Cache 集成测试

```text
admission commitment -> physical blocks never exceed commitment
finish/cancel -> physical blocks and commitment both released
released capacity -> oldest eligible waiting request admitted
all rows at block boundary -> lifetime policy still makes progress
unexpected physical backpressure -> invariant failure, no forced cancel
request churn -> no leaked block or commitment
```

### 11.3 Workload 测试

```text
finite oversubscribed workload -> every admissible request completes
greedy boundary case -> deterministic deadlock detection
cancel baseline -> cancellation count preserved
same seed/config -> lifecycle and scheduler metrics reproducible
```

## 12. Benchmark 设计

在现有 `short_churn`、`mixed_steady`、`long_pressure` 之外增加一个 adversarial `boundary_deadlock` 场景，使多个请求同时到达 block boundary。

固定相同 arrival、request spec、seed、dtype、backend 和 cache capacity，对三种 policy 报告：

- completed requests/tokens 与 completion rate。
- forced cancellations、stalled/backpressure steps。
- TPS、complete-step p50/p90/p99。
- admission wait 与最大 service wait。
- physical/commitment utilization。
- scheduler decision overhead。

解释结果时不要求 lifetime policy 的所有 latency 指标都最好。它可能降低瞬时并发或增加等待；核心判断是能否消除容量死锁和强制取消，并给出可解释的完成率/公平性收益。

## 13. 实现与验证

### 纯策略层

- immutable metadata、snapshot、decision 和 config。
- 实现 commitment 计算、FIFO/aging admission、fair runnable selection。
- 不依赖 torch/CUDA 的纯 Python tests。

实现位于 `flashdec/scheduler.py`，核心测试位于 `tests/test_scheduler.py`。planner 与基础 tests 不依赖 torch；standard-library `unittest`、随机合法 snapshot invariant、WSL pytest 与完整回归共同覆盖纯策略层。

### Engine/Cache 集成

- 增加 `state_version` 和 scheduler-facing snapshot。
- 应用 decision 时进行 stale check。
- lifecycle 同步释放 commitment。
- 保持 Cache 为唯一 physical allocator。

验证覆盖 CPU/PyTorch 状态路径、RTX fused/Triton 数值对齐和 stale-decision 原子拒绝。

### Workload 与对照实验

- runner 支持三种 policy。
- boundary-deadlock workload 与 scheduler metrics。
- CPU quick 验证生命周期轨迹，再在 RTX 5070 跑 FP16/BF16 正式矩阵。

trace-driven runner 包含三策略 event loop、确定性 deadlock cutoff、等待/服务公平性、commitment/physical block 和有效 token 指标。CPU/PyTorch 对抗路径与 scheduled fused CUDA + Triton 数值路径均被覆盖；commit `16de9d4` 的 RTX 5070 36-row policy matrix 通过严格摘要校验。

### 默认策略的证据要求

- 所有 correctness/invariant 通过。
- 默认 pressure workload 不依赖强制取消恢复进度。
- 有限 admissible 请求最终完成，无 starvation。
- 报告 lifetime reservation 相对 greedy/cancel baseline 的收益与代价。
- 默认选择必须同时满足上述 correctness、进展和可解释性证据。

## 14. 实验结论

lifetime policy 在 boundary-deadlock 中完成 2/2 请求且无取消/死锁；cancel baseline 完成 1/2，greedy baseline 完成 0/2 并触发确定 deadlock。finite queue 中三种策略均完成 6/6，因此稳定结论是容量安全与进展保证，而不是无条件 latency/TPS 优势。[正式矩阵](../benchmarks/results/scheduler_capacity_progress_summary.md)

## 15. Shared Prefix 容量扩展

基础不变量 `sum(active physical_blocks) == used_blocks` 建立在每个 request 独占 block 的前提上。Shared Prefix 允许多个 request 的 block table 重复引用同一组 immutable prefix ids，因此增加两层派生元数据：

- `SchedulingSnapshot.resident_prefix_blocks`：Cache registry 当前持有的唯一 physical prefix blocks，只计一次。
- `WaitingRequestMetadata/ActiveRequestMetadata.shared_prefix_blocks`：单个 request 的 logical block table 中有多少前导块来自 prefix。

`RequestSpec` 只增加 opaque `prefix_id`，不接受调用方提供 prefix 长度。DecodeEngine 从 Cache registry 派生 shared block 数并验证 prefix 覆盖完整 initial context，防止伪造容量节省。

扩展后的不变量为：

```text
used_blocks
  == resident_prefix_blocks
   + sum(active physical_blocks - active shared_prefix_blocks)

committed_blocks
  == resident_prefix_blocks
   + sum(active request-private lifetime commitments)

used_blocks <= committed_blocks <= schedulable_blocks
```

waiting request 的 admission 只消耗 private lifetime commitment；prefix residency 已经包含在全局基数中。admission 后 Engine 将同一 prefix ids 挂到 request block table 开头，下一 token 位于 full-block boundary 并分配私有 tail。

边界：scheduler-managed mode 不在 decision 内注册或淘汰 prefix。resident set 必须在 request submission 前建立，后续外部 Cache mutation 会触发 version mismatch。固定 hit-rate workload 分别衡量 KV-pool capacity 节省与性能归因。

Engine/Scheduler 集成在 2026-07-17 的 RTX 5070 WSL 环境通过 focused `56 passed, 14 subtests passed in 5.29s` 与完整 `352 passed, 25 subtests passed in 9.37s` 回归。

fixed bounded capacity 验证 admission 单调性；独立 fixed-full-batch 容量测量 decode latency，避免把 admitted batch 差异混入性能解释。

初始 attribution 发现 75% hit 的 scheduler p50 在 FP16/BF16 都三轮稳定回退，因此把 submission 时已验证的 `shared_prefix_blocks` 存入 Engine request metadata；纯调度热路径不再重复读取 prefix registry。Scheduler 仍只接收派生整数，不接触 K/V 或 physical ids；DecodeEngine 继续在 active snapshot/invariant 中将缓存值与 Cache request state 交叉校验，外部 registry mutation 仍触发 version mismatch。

commit `fe72e27` 的 lookup-count targeted test 为 `1 passed`，focused 为 `61 passed, 8 subtests passed`，完整回归为 `361 passed, 25 subtests passed`。同 commit 的 8-trial/64-row confirmation 中，所有非零 hit-rate 的 scheduler p50 range 仍跨过 1；因此结构性结论是重复 registry lookup 已从热路径移除，性能结论为 near-neutral，而不是稳定 scheduler speedup。
