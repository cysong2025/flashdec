# FlashDec 深度路线图：从 v0.1 到可解释的 Decode Runtime

## 1. 路线图目标

FlashDec 后续不再通过增加零散算子或重复参数 sweep 扩充内容，而是围绕一个问题继续加深：

> 单 GPU decode runtime 如何在有限 KV 显存下选择请求、执行多层 decode、复用缓存，并用可复现实验证明 kernel 优化能否转化为系统收益？

目标交付不是完整 LLM serving framework，而是一个研究型、可解释、公开可复现的 runtime 原型。每个阶段都必须同时具备：

1. 明确的状态与内存语义。
2. PyTorch/reference 或纯 CPU 可验证路径。
3. RTX 5070 correctness 证据。
4. 可复现 benchmark 与负结果。
5. 能说明设计取舍的中文文档。

## 2. 当前基线与真实缺口

### 已完成

- PyTorch dense/paged decode reference。
- Triton dense/paged decode kernel，覆盖 FP16/BF16、MHA/GQA/MQA、head_dim 64/128 和变长 context。
- token-major、`block_size=32`、`num_warps=2`、`num_stages=None` 的冻结配置。
- PagedKVCache allocate/free/reuse、finish/cancel、capacity atomicity、fragmentation 与 invariant。
- PyTorch、独立 CUDA、fused CUDA 三条 RoPE/KV append 路径。
- DecodeEngine 单 layer dynamic batch、append -> paged decode 与显式 backpressure。
- 三类 synthetic workload 和首轮 complete-step benchmark。
- 首轮 full workload 12/12 invariant 通过；fused p50 几何平均为 1.0537x，但 p99 尚不稳定。

### 当前代码暴露出的缺口

| 缺口 | 当前证据 | 为什么值得做 |
| --- | --- | --- |
| 结果稳定性 | 已有 multi-trial 参数，但 3-trial CSV 尚未验证 | 单次 p99 和 short-churn 方向不稳定，不能冻结 release 结论 |
| 阶段耗时归因 | workload 只有 complete-step wall-clock | 无法直接回答 append、attention、scheduler、allocator 各占多少 |
| 调度策略 | `admit()` 不预留 block；workload 在 backpressure 时取消最老请求 | 当前是压力测试策略，不是 block-aware continuous batching |
| 多 layer 事务 | cache 存储有 layer 维度，但 runtime 强制 `num_layers=1` | 对多个 layer 逐次 append 会错误地重复推进 seq_len，需要真正事务语义 |
| 共享前缀 | 每个请求独占所有 physical blocks | 无法研究重复 prompt 下的显存节省、refcount 和 eviction |
| 发布证据 | 缺少 clean-install reproduction、CHANGELOG 和正式 tag | 代码可运行不等于第三方可以复现 |

## 3. 目标架构

```text
Synthetic / Trace-driven Workload
              |
              v
Block-aware Scheduler v2
waiting queue / admission / runnable subset / fairness
              |
              v
DecodeEngine
multi-layer token transaction / stable request-row mapping
              |
       +------+------+
       |             |
       v             v
Paged KV Runtime   Observability
free/reuse         stage timing
refcount/prefix    queue/memory metrics
       |
       v
Fused RoPE + KV Append -> Triton Paged Decode
```

Scheduler 只决定 request ids、顺序和资源预算，不生成 Q/K/V；DecodeEngine 只执行传入的单 token 数据路径；Paged KV Runtime 独立维护所有权与原子性。这个边界应一直保持。

## 4. 阶段 R0：闭合 v0.1 证据链

优先级：P0，必须先完成。预计 1-2 个阶段周。

### R0.1 Multi-trial 聚合

目标：把首轮结论升级为 release 级证据。

当前状态：聚合器、Markdown 输出和纯 Python 错误路径测试已实现，等待 RTX 5070 生成正式 36 行 CSV 后验证。

工作内容：

- 在 RTX 5070 完成 `--trials 3 --dtype both`。
- 使用 `benchmarks/summarize_decode_engine_trials.py`：
  - 校验 workload/dtype/trial 的 torch/fused 严格配对。
  - 校验两条 backend 的 state/allocator 轨迹一致。
  - 输出每个 trial 的 p50/p90/p99/TPS ratio。
  - 输出跨 trial median、min/max 和几何平均。
  - 自动拒绝缺行、重复行、seed/order 不一致和 invariant failure。
- 更新 Week 12 摘要，区分稳定结论与噪声。

验收门槛：

- 3 workloads x 2 dtypes x 2 backends x 3 trials，共 36 行完整记录。
- 36/36 invariant 通过，所有 pair 的 request/cache 指标一致。
- p50/p90 使用跨 trial 聚合；p99 同时报告范围，不用单个最大值做默认决策。
- 如果某 workload 的 speedup 跨过 1.0，明确记录为不稳定/无确定收益。

### R0.2 Complete-step 阶段归因

目标：解释 append-only 约 18.2% latency 降低为何只转化为 complete-step p50 约 5.1%。

当前状态：可选 Engine ranges、动态 workload profiler、阶段文本/Markdown/Chrome trace 输出和 CPU helper tests 已实现，等待 RTX 5070 验证 range 计数与 device time。

工作内容：

- 使用 `benchmarks/profile_decode_engine.py`。
- 为以下区域建立可选 profiler range，默认 fast path 不启用额外同步：
  - submit/admit。
  - allocator/preflight/metadata。
  - RoPE + KV append。
  - paged decode attention。
  - finish/cancel。
- 同时报告 CPU wall time、CUDA kernel time、launch 次数和 complete-step CUDA-event/wall-clock。
- 使用 short-churn、mixed-steady、long-pressure 三个固定 workload。
- 用 Amdahl-style 模型比较理论上限与实测收益，记录误差来源。

验收门槛：

- 能解释至少 80% 的 successful step 时间归属，无法归因的部分单列为 synchronization/runtime gap。
- profiler 模式与非 profiler 模式的 p50 差异单独记录，不把 instrumented latency 当正式性能结果。
- 至少得到一个“kernel 更快但系统收益有限”的量化解释。

### R0.3 v0.1.0 可复现发布

当前状态：reproducibility guide、Unreleased changelog、README quick start/support matrix、packaging extras、扩展环境检查和 release checker 已实现；clean WSL venv、正式 3-trial/profile 证据、版本升级和 tag 仍待完成。

交付物：

- `docs/reproducibility.md`。
- `CHANGELOG.md`。
- README quick start、支持矩阵、限制和架构图。
- CPU-only test 命令、RTX 5070 focused/full 命令、quick workload 命令。
- clean WSL venv 安装日志与核心验证记录。
- `v0.1.0` tag。

验收门槛：

- 新目录/新 venv 能完成 editable install。
- CPU suite 不依赖 CUDA Toolkit。
- GPU focused suite、full regression、quick workload 均可按文档复现。
- 所有公开性能数字都能追溯到 commit、命令、设备、seed/trial 和摘要。

## 5. 阶段 R1：Block-aware Scheduler v2

优先级：P1，是 v0.2 的第一条核心深度主线。预计 2 个阶段周。

当前状态：目标语义、deadlock 反例、lifetime commitment、状态所有权、测试与 benchmark 边界已在 `docs/design_scheduler.md` 冻结；R1-A 纯 Python planner 与 focused tests 已实现，Mac 无依赖 smoke 通过。Engine/Cache 集成和 RTX 5070 证据尚未开始，待 R0 release gate 闭合后实施。

### 要回答的问题

当前 workload 遇到容量压力时取消最老请求。更合理的问题是：

> 在不破坏 request/KV 状态的前提下，如何从 waiting/active requests 中选择本 step 可运行的子集，并避免无限饥饿？

### 设计边界

新增独立 `flashdec/scheduler.py`，Scheduler 不持有 K/V tensor，不调用 kernel，只消费 request metadata 和 cache capacity，输出 `SchedulerDecision`：

```text
admit_ids
runnable_ids
deferred_ids
needed_new_blocks
free_blocks_before_step
decision_reason
```

第一版只实现 deterministic FIFO + aging，不实现 priority API、swap 或生产级抢占。完整资源语义见 `docs/design_scheduler.md`。

仅按本 step 的 `free_blocks` 选择 runnable subset 不能作为默认策略：当全部 active requests 同时到达 block boundary 且没有 free block 时，任何 request 都无法推进到释放容量，系统会死锁。默认 policy 因此使用 admission-time lifetime commitment：

```text
request_total_blocks = ceil(
    (initial_context_tokens + max_new_tokens) / block_size
)

sum(active commitments) <= max_blocks - reserve_blocks
```

commitment 是 Scheduler 的逻辑容量账本，physical block 仍由 PagedKVCache 按 append 进度惰性分配。`greedy_step_only` 只作为可能 stall/deadlock 的实验对照。

### 工作内容

- `RequestSpec`：确定的 initial context、最大 decode token budget 和稳定提交顺序。
- `SchedulerConfig`：`max_active_requests`、`max_batch_requests`、block reserve、aging threshold 和 policy。
- waiting queue 与稳定提交顺序。
- lifetime block-aware admission：完整生命周期容量无法承诺时保持 waiting，而不是先 active 再失败。
- FIFO + aging/drain barrier：允许小请求利用空余容量，但有限 workload 不能永久绕过老请求。
- fair runnable subset：按 service wait 轮转并服从 `max_batch_requests`；默认策略的跨 boundary 容量由 commitment 保证。
- DecodeEngine 增加 scheduler-facing snapshot 与单调 `state_version`，拒绝 stale decision；allocator 所有权仍由 cache 管理。
- workload runner 对比 `cancel_on_backpressure`、`greedy_step_only` 与 `lifetime_fifo_aging`。
- 新增 adversarial boundary-deadlock workload，禁止 benchmark 无限等待。

### 指标

- queue depth、admission wait steps。
- runnable/deferred requests。
- successful tokens、stall/backpressure steps。
- forced cancellations。
- scheduler decision wall time。
- committed/physical block utilization、committed-but-unallocated blocks、fragmentation、reuse。
- per-request service steps 与最大等待时间。
- stale decision 与 resource-deadlock count。

### 核心测试

```text
oversubscribed waiting queue -> capacity-safe admission
all rows need a block -> lifetime commitment prevents boundary deadlock
deferred row -> later becomes runnable
finite workload -> no starvation
finish/cancel -> waiting request admitted and reuses blocks
scheduler decision -> no cache mutation
same decision seed/config -> deterministic order
stale decision -> rejected without partial mutation
request larger than schedulable capacity -> explicit rejection
```

### 验收门槛

- 默认 pressure workload 不依赖强制取消来恢复进度。
- 有限且单请求可容纳的请求集合，在停止到达后最终全部完成或由调用方显式取消。
- scheduler decision 不修改 cache；只有 Engine/cache API 可以改变所有权。
- active commitment 不超过 schedulable capacity，physical ownership 不超过 commitment。
- 相对 cancel-on-backpressure/greedy baseline，报告 completed tokens、p50/p99、等待时间和利用率的取舍，不要求每个指标都更好。

## 6. 阶段 R2：Multi-layer KV Token Transaction

优先级：P1，是 v0.2 的第二条核心深度主线。预计 2 个阶段周。

当前状态：状态机、committed/pending seq_len、shared location、abort rollback、sequential layer Engine API、测试和 benchmark 边界已在 `docs/design_multi_layer_kv_transaction.md` 冻结；代码与 RTX 5070 证据尚未开始。

### 要回答的问题

真实 Transformer 的一个 token 会为所有 layer 写入 K/V，但 request seq_len 只能增加一次。当前逐 layer 调用 `append()` 会错误地多次推进 seq_len，因此不能通过删除 `num_layers=1` 检查来解决。

### 事务模型

```text
begin_token(request_ids)
  -> preflight capacity
  -> reserve physical locations once
  -> write layer 0 ... layer N-1
commit_token()
  -> advance each request seq_len once

exception before commit
  -> rollback newly reserved blocks
  -> seq_len and ownership remain unchanged
```

open transaction 中 committed `seq_len` 保持不变，各 layer attention 使用 `effective_seq_len = committed_seq_len + 1`。已经写入 committed length 之外的 partial bytes 在 abort 后保持不可见，无需做昂贵清零。

### 工作内容

- `KVAppendTransaction` 或等价内部对象。
- block location 对所有 layer 一致，K/V storage 仍保留 layer 维度。
- 第一版 Cache 同时只允许一个 open batch transaction，所有 request 一起 commit/abort。
- PyTorch multi-layer append reference。
- CUDA 路径第一版允许每 layer 一个 fused launch，但 allocator/preflight/commit 只执行一次。
- DecodeEngine 增加 sequential `begin_step/step_layer/commit_step` API；保留单 layer compatibility wrapper。
- metrics 增加 reserved/active bytes，并区分 logical blocks 与跨 layer 实际字节。

### 核心测试

```text
2/4 layers -> seq_len only increments once
all layers -> physical block ids identical
failure before commit -> no partial seq_len/block mutation
finish/cancel -> release all layer storage ownership
multi-layer decode -> each layer matches reference
FP16/BF16 + GQA -> CPU/reference and RTX path aligned
```

### 验收门槛

- `num_layers=2/4` CPU/reference correctness 完整。
- 至少 `num_layers=2` 的 fused CUDA + Triton 路径在 RTX 5070 通过。
- 任意 layer 写入失败时，不留下 partial request state。
- benchmark 明确报告 layer 数、总 KV bytes、launch 数与 token latency，不把单 layer结果外推。

## 7. 阶段 R3：Shared Prefix Blocks（选择性进阶）

优先级：P2。只有 R0-R2 全部完成后开始。预计 2-3 个阶段周。

### 目标

研究重复 system prompt / prefix 下 physical KV blocks 的共享、引用计数和回收，而不是实现 tokenizer 或完整 prefill engine。

### 建议范围

- 调用方传入 opaque `prefix_id` 与已构建的 full blocks。
- 只共享 immutable full blocks；tail block 第一版保持 request-private。
- physical block refcount。
- request finish/cancel 只减少 refcount，最后一个 owner 才归还 free list。
- 有容量边界的 LRU，只淘汰无 active owner 的 cached prefix。
- hit/miss、shared blocks、saved blocks/bytes、eviction 指标。

### 核心测试

```text
two requests attach same prefix -> physical ids shared
one request finishes -> other request data remains valid
last owner finishes + eviction -> blocks become reusable
private tail append -> does not mutate shared prefix
capacity failure -> refcount and ownership unchanged
```

### Benchmark

固定请求数量与 context，构造 0%/25%/50%/75% prefix hit rate，对比：

- physical blocks/bytes。
- admission success。
- cache hit latency。
- decode step latency。
- eviction 次数。

验收重点是显存节省和所有权正确性，不要求 prefix lookup 本身带来 decode kernel 加速。

## 8. 阶段 R4：公开基线与项目表达

优先级：P2，可与 R3 二选一；如果时间有限，优先做公开基线而不是同时扩展更多功能。

### 有限公开对比

- 选择 FlashInfer 或 vLLM 的公开、可安装版本之一并固定版本/commit。
- 只比较共同支持的 paged decode shape、dtype、layout 语义。
- 分离 kernel-only 与 runtime workload，不把不同计时边界放在同一 speedup 表中。
- 同时记录安装成本、API/布局转换和不兼容项。

### 对外材料

- 一篇中文技术文章：从 PagedAttention kernel 到 block-aware decode runtime。
- 架构图、状态机、KV block ownership 图和性能归因图。
- 面试讲解按五层组织：算法、kernel、allocator、scheduler、实验方法。

验收门槛：任何对比数字都绑定版本、shape、计时边界和命令；无法公平对齐的项目明确标记为不可比。

## 9. 优先级与截止线

| 优先级 | 内容 | 是否影响“深度项目”完成 |
| --- | --- | --- |
| P0 | multi-trial、阶段归因、reproducibility、v0.1.0 | 必须 |
| P1 | block-aware scheduler | 必须，最重要的系统扩展 |
| P1 | multi-layer KV transaction | 必须，修复当前架构边界 |
| P2 | shared prefix blocks | 选择性进阶 |
| P2 | FlashInfer/vLLM 有限公开对比 | 选择性进阶，建议和 prefix 二选一 |
| 不做 | HTTP server、完整模型、sampling、TP/PP、多机、swap/offload | 不影响项目完成 |

如果时间不足，项目应在 R2 后停止增加功能，集中完成复现和文章。Scheduler + multi-layer transaction 比再增加三个小 kernel 更能证明 AI Infra 深度。

## 10. 每阶段统一 Definition of Done

每项功能只有同时满足以下证据才算完成：

1. 设计文档写清状态、所有权、失败原子性和不做什么。
2. CPU/reference tests 覆盖正常路径与错误路径。
3. GPU 路径与 reference 对齐，不用 pytest 总耗时代表性能。
4. benchmark 固定 warmup/repeat/trial/seed 和计时范围。
5. 结果包含 p50/p90/p99、吞吐、内存和 lifecycle 指标中适用的部分。
6. 至少记录一个负结果或未达到门槛的实验。
7. README/NEXT_STEPS/weekly status 与真实状态一致。
8. clean worktree、完整回归、结果绑定 commit。

## 11. 当前立即执行顺序

1. WSL 先验证 R1-A `tests/test_scheduler.py`，并完成现有 focused/full regression。
2. RTX 5070 运行 `--quick --trials 2` 与正式 `--trials 3`，使用聚合器冻结 Week 12 p50/p90/p99 结论。
3. 运行已实现的 complete-step profiler，完成 append/attention/runtime 阶段归因。
4. 完成 clean-install、`v0.1.0` reproducibility 与 release。
5. 实现 R1-B Engine/Cache snapshot、state version、decision apply 与 commitment lifecycle。
6. 实现 R1-C 三策略 workload 对照和 boundary-deadlock 实验。
7. Scheduler correctness/benchmark 冻结后，再按 R2 设计实现 multi-layer transaction；不并行启动 prefix。

这条顺序保证每次只引入一个新的系统变量，实验结果仍然可解释。
