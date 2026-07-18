# Multi-layer KV Token Transaction 设计说明

## 1. 目标

R2 要把 FlashDec 从“存储张量带 layer 维度，但执行语义固定单层”推进为真正的 multi-layer decode token transaction。

真实 Transformer 的一个 decode token 会依次经过所有 layer。每个 layer 都产生一组 K/V，并需要在本 layer attention 前把当前 token 写入 cache；但 request 的逻辑 `seq_len` 只能在整个 token 成功完成后增加一次。

R2 要保证：

```text
一个 token
  -> 预留一次 physical location
  -> layer 0 ... layer N-1 使用同一 block id / offset
  -> 每层 append 后可用 pending seq_len 做 attention
  -> 所有 layer 成功后 seq_len 只增长一次
```

任一 layer 失败时，不能留下 partial `seq_len`、泄漏 block 或可见的半个 token。

## 2. 当前实现为什么不能直接扩展

当前 storage 已经是：

```text
[num_layers, max_blocks, num_kv_heads, block_size, head_dim]
```

但 `PagedKVCache` 构造函数显式拒绝 `num_layers != 1`，而下面三条 append 路径都执行相同的生命周期：

```text
preflight
-> allocate append location
-> write one layer
-> _advance_request_lengths(ids)
```

- `append()`：PyTorch write 后增长 `seq_len`。
- `append_cuda()`：native K/V write 后增长 `seq_len`。
- `append_fused_cuda()`：RoPE + native K/V write 后增长 `seq_len`。

如果对 `num_layers=4` 简单循环调用四次，request 会从 `L` 错误增长到 `L+4`，四层还会写入不同 token offset。删除构造检查不能解决这个问题。

## 3. 核心语义：committed 与 pending token

### 3.1 Committed seq_len

`request.seq_len` 只表示已经完成所有 layer 的 token 数。普通 cache reader、scheduler snapshot 和 lifecycle metrics 默认只读取 committed state。

### 3.2 Pending token

open transaction 为每个 request 建立一个 pending token：

```text
position = committed_seq_len
effective_seq_len = committed_seq_len + 1
```

逐层 attention 必须使用 `effective_seq_len`，因为当前 layer 已经写入当前 token 的 K/V；但 transaction commit 前，`request.seq_len` 保持不变。

### 3.3 Shared location across layers

一个 physical block id 在 FlashDec 中代表包含所有 layer storage 的 block slot。事务只为每个 request 计算一次：

```text
logical_block = position // block_size
block_offset = position % block_size
physical_block = request.block_ids[logical_block]
```

所有 layer 写入：

```text
k_cache[layer_idx, physical_block, :, block_offset, :]
v_cache[layer_idx, physical_block, :, block_offset, :]
```

因此 layer 数会增加每个 physical block 的实际 bytes，但不会把 scheduler 的 logical block commitment 乘以 layer 数。

## 4. Transaction 状态机

```text
                begin_token
                    |
                    v
                  OPEN
                 /    \
       all layers      exception / explicit abort
          written       \
            |            v
            v          ABORTED
         COMMITTED
```

状态只能单向变化：

- `OPEN -> COMMITTED`
- `OPEN -> ABORTED`
- terminal transaction 不能再次 write、commit 或 abort。

第一版整个 Cache 同时只允许一个 open batch transaction。这与当前单线程 `DecodeEngine.step()` 边界一致，避免在 R2 同时引入 overlapping transaction、stream ownership 和并发锁问题。

## 5. 建议数据模型

```python
@dataclass(frozen=True)
class KVTokenTransactionView:
    transaction_id: int
    cache_version: int
    request_ids: tuple[Hashable, ...]
    positions: Tensor                 # [batch]
    physical_block_ids: Tensor        # [batch]
    block_offsets: Tensor             # [batch]
    block_tables: Tensor              # includes reserved boundary block
    effective_seq_lens: Tensor        # committed + 1
    next_layer_idx: int
    state: str                        # open/committed/aborted


@dataclass
class _KVTokenTransactionState:
    view metadata
    newly_allocated_by_request: dict[Hashable, int]
    original_block_counts: dict[Hashable, int]
    written_layers: set[int]
```

公开 view 不暴露 mutable request state 或 free list。positions/block tables 等 tensor 是供 attention 使用的快照；Cache write/commit/abort 必须根据 transaction id 回查内部 location，不能信任调用方可修改的 tensor 作为 ownership 依据。Cache 内部 state 记录 rollback 所需的最小信息。

## 6. Cache API

### 6.1 Begin

```python
tx = cache.begin_token(request_ids)
```

`begin_token()`：

1. 校验 request ids 非空、唯一且全部 active。
2. 拒绝已有 open transaction。
3. 对整个 batch 计算 boundary request 数并统一 capacity preflight。
4. capacity 足够后，才为 boundary rows 分配 block。
5. 记录每个 request 的原始 block count 与新 block。
6. 计算 positions、shared locations、block tables 和 effective seq lens。
7. 标记 request 处于 in-flight transaction，但不增长 committed `seq_len`。
8. 递增 cache/engine state version，使旧 scheduler decision 失效。

capacity failure 必须发生在任何 request/block mutation 之前。begin 内部即使在 preflight 后出现意外异常，也必须归还本次已取出的 block，并恢复原始 request block count。

### 6.2 Per-layer write

PyTorch reference：

```python
cache.write_token_layer(tx, layer_idx, k, v)
```

RoPE/Engine 路径：

```python
layer_result = engine.decode_transaction_layer(
    tx,
    layer_idx,
    q,
    k,
    v,
)
```

每次 layer write：

1. transaction 必须是当前 Cache 的 OPEN transaction。
2. 第一版强制 `layer_idx == tx.next_layer_idx`，按 0..N-1 顺序执行。
3. shape/dtype/device 在写入前完整校验。
4. 使用 transaction 已有 block ids/offsets，不再 preflight、allocate 或增长 seq_len。
5. 写入成功后才把 layer 标记为 written。
6. 当前 layer attention 使用 `tx.block_tables` 与 `tx.effective_seq_lens`。

顺序约束符合真实 Transformer 数据依赖，也能更早发现漏层、重复层和错误 layer routing。

### 6.3 Commit

```python
cache.commit_token(tx)
```

commit 只有在全部 `num_layers` 已按顺序写入后才能执行：

1. 对 batch 中每个 request 将 committed `seq_len += 1`。
2. 清除 request in-flight marker。
3. 将 reserved block 转为普通 committed ownership。
4. transaction 进入 COMMITTED。
5. 更新 transaction/token/allocator metrics 和 state version。

整个 batch 一起 commit，不支持部分 request commit。

### 6.4 Abort

```python
cache.abort_token(tx)
```

abort：

1. committed `seq_len` 保持 begin 前的值。
2. 从 request block list 移除本 transaction 新增的 boundary block。
3. 按 deterministic free-list 规则归还这些 block。
4. 清除所有 request 的 in-flight marker。
5. transaction 进入 ABORTED 并记录 rollback metrics/state version。

已经写入旧 tail block 的 K/V 字节不需要恢复原值。它们位于 committed `seq_len` 之外，对普通 reader 不可见；下一次 transaction 会覆盖同一 offset。新 block 中的 partial 数据也无需清零，归还后的新 owner 由自身 seq_len 屏蔽未写区域。

`allocation_count`、fresh/reuse counters 记录已经发生过的 allocator 事件，abort 时不倒扣；归还行为单独计入 `transaction_rollback_block_count`。现有 lifecycle `free_count` 继续表示 finish/cancel 释放，避免把 rollback 与 request termination 混为一个指标。

## 7. 异常与 lifecycle 规则

Engine 的 multi-layer step 必须使用：

```python
tx = cache.begin_token(ids)
try:
    for layer_idx in range(cache.num_layers):
        output = engine.decode_transaction_layer(...)
    cache.commit_token(tx)
except Exception:
    cache.abort_token(tx)
    raise
```

规则：

- open transaction 涉及的 request 不能 finish/cancel/再次 append。
- 第一版其他 request 也不能启动第二个 transaction。
- caller 必须丢弃 abort transaction 已产生的中间 layer output。
- commit 缺少任一 layer 时明确失败，transaction 保持 OPEN，调用方随后 abort。
- invalid shape、重复 layer、越序 layer、stale handle、double commit/abort 都必须显式报错。
- CUDA extension/toolchain 应尽可能在 `begin_token()` 前完成 lazy load；begin 后的任何异常都走 abort。

## 8. Engine API 边界

真实 layer 的 Q/K/V 是顺序产生的，不能要求调用方预先提供 `[num_layers, batch, ...]` 后一次性执行。因此 R2 的主 API 应是分层 transaction，而不是只提供一个 convenience tensor loop：

```python
tx = engine.begin_step(request_ids)

for layer_idx in range(num_layers):
    # q/k/v may depend on the previous layer output
    layer_result = engine.step_layer(tx, layer_idx, q, k, v)

result = engine.commit_step(tx)
```

`step_layer()` 执行：

```text
RoPE at tx.positions
-> write K/V at pre-reserved location for this layer
-> paged decode against cache[layer_idx]
   using tx.effective_seq_lens
```

现有单层 `DecodeEngine.step()` 保留兼容，并在内部包装：

```text
begin_step -> step_layer(layer 0) -> commit_step
```

这样单层 benchmark/API 不会出现两套 allocator 语义。

## 9. CUDA 路径重构

当前 native append API 同时做两件事：Python allocator mutation 和 CUDA write。R2 需要把它们拆开：

```text
Cache transaction:
  owns preflight / block reservation / commit / abort

Native write primitive:
  consumes layer cache + block_ids + offsets + Q/K/V + positions
  writes one layer only
  never changes request state or seq_len
```

第一版允许每 layer 一个 fused CUDA launch。R2 的核心收益不是减少 layer launch，而是保证 allocator/preflight 只执行一次、所有 layer 使用同一 location，并获得失败原子性。

PyTorch reference 与 fused CUDA 必须共享同一个 transaction allocator，不允许 native path 维护另一份 ownership 规则。

## 10. Scheduler 与内存计量关系

Scheduler commitment 仍按 request token block 数计算：

```text
commitment_blocks = ceil((initial_context + max_new_tokens) / block_size)
```

不乘 `num_layers`，因为一个 logical physical-block id 同时索引所有 layer storage。但实际显存必须报告：

```text
bytes_per_block = (
    num_layers
    * 2                       # K + V
    * num_kv_heads
    * block_size
    * head_dim
    * dtype_bytes
)

allocated_kv_bytes = used_blocks * bytes_per_block
reserved_kv_bytes = reserved_blocks * bytes_per_block
```

R1 scheduler snapshot 在 open token transaction 期间不能重新 plan；Engine 应显式拒绝构造可执行 snapshot，而不是把 reserved block 伪装成 committed physical ownership。`begin/commit/abort` 都改变 state version，避免 decision 与 reserved ownership 脱节。

## 11. Invariant

### 11.1 无 open transaction

保持现有规则：

```text
len(request.block_ids) == ceil(request.seq_len / block_size)
owned blocks + free blocks == all physical blocks
terminal request owns no block
```

### 11.2 有 open transaction

允许 boundary request 临时多一个 reserved block：

```text
request.seq_len == committed seq_len
len(block_ids) == ceil((seq_len + 1) / block_size)
effective_seq_len == seq_len + 1
transaction location matches last logical token position
each request belongs to at most one transaction
reserved block is owned, not free, and belongs to exactly one request
written_layers == {0, ..., next_layer_idx - 1}
```

无论 OPEN/COMMITTED/ABORTED，owned/free block 集合必须无重复、无遗漏。

## 12. Metrics

在现有 allocator/lifecycle metrics 上增加：

- `transaction_begin_count`
- `transaction_commit_count`
- `transaction_abort_count`
- `open_transaction_count`（0 或 1）
- `pending_request_count`
- `reserved_transaction_blocks`
- `transaction_layer_write_count`
- `transaction_rollback_block_count`
- `transaction_failure_count`
- `allocated_kv_bytes`
- `reserved_transaction_bytes`
- `bytes_per_block`

benchmark 还要报告每 token allocator/preflight 次数、每 layer launch 数和完整 token latency。

## 13. 测试计划

### 13.1 CPU/PyTorch transaction tests

```text
num_layers=2/4 -> all layers write, seq_len commits once
open transaction -> committed seq_len unchanged, effective seq_len +1
all layers -> same physical block id and offset
mixed boundary/tail batch -> only boundary rows reserve block
capacity failure -> no request/block/transaction mutation
layer write failure -> abort restores seq_len and block ownership
abort after old-tail write -> partial bytes remain invisible
commit before all layers -> rejected
duplicate/out-of-order layer -> rejected before write
finish/cancel/append during open tx -> rejected
stale/double commit/double abort -> rejected
single-layer compatibility wrapper -> existing reference unchanged
request churn with commit/abort -> no leaked blocks
scheduler snapshot during open transaction -> rejected
```

### 13.2 Engine correctness

```text
2/4 layers -> each step_layer output matches per-layer paged reference
layer output row order -> request id mapping remains stable
exception at layer k -> whole token state rolls back
finish after commit -> releases block once for all layer storage
```

### 13.3 RTX 5070

- `num_layers=2` 必做，`num_layers=4` 建议。
- FP16/BF16。
- head_dim 64/128。
- MHA/GQA，至少一个 MQA case。
- tail append 与 block-boundary append。
- fused CUDA writes 与 PyTorch transaction reference 对齐。
- 每个 layer 的 Triton paged decode output 对齐 reference。

## 14. Benchmark 设计

固定 token-major、`block_size=32`、`num_warps=2`、implicit stages，不重新 sweep 已冻结 kernel 参数。

建议矩阵：

| 维度 | 候选值 |
| --- | --- |
| layers | 1、2、4；显存允许时增加 8 |
| batch | 4、16 |
| context | 128、1024 |
| dtype | FP16、BF16 |
| append | torch transaction、fused CUDA transaction |

报告：

- complete multi-layer token p50/p90/p99。
- per-layer append/decode device time。
- tokens/s 与 layers/s。
- allocator/preflight/commit CPU time。
- CUDA launch count。
- used/reserved blocks 与实际 KV bytes。
- abort rollback latency（独立错误路径实验，不混入正常吞吐）。

不能拿单层 latency 线性外推多层结果，也不能把预先生成所有 Q/K/V 的 synthetic convenience 成本混入正式 Engine transaction 计时。

## 15. 实现阶段

### R2-A：Cache reference transaction

- 解除单层构造限制，但仅允许 transaction API 在 `num_layers > 1` 下写入。
- 实现 begin/write/commit/abort 与 open-transaction invariant。
- 保留单层 legacy append tests，新增 2/4-layer CPU tests。

### R2-B：Engine sequential layer API

- 实现 `begin_step()`、`step_layer()`、`commit_step()`、`abort_step()`。
- 保留 `step()` 单层 compatibility wrapper。
- 将 R1 state version 与 transaction lifecycle 对齐。

### R2-C：Fused CUDA location-only write

- native kernel 接收 transaction block ids/offsets。
- allocator、rollback 和 seq_len 只由 Python Cache transaction 管理。
- RTX 5070 完成 2-layer FP16/BF16 correctness。

### R2-D：Workload 与报告

- 已实现 12-case workload、torch/fused 配对和交替 trial 顺序。
- non-instrumented 路径分离完整 token wall/CUDA-event、per-layer device 与 begin/commit host time。
- 独立 profiler probe 记录 append/decode device time 和 CUDA event count；独立 rollback probe 不混入正常吞吐。
- 严格 summary 校验 matrix、shape、transaction、block、profile、rollback、seed/order。
- commit `fa0f89a` 的正式 144 行 RTX 5070 矩阵已通过严格校验。
- fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`；per-layer append device 为 `1.6103x`，decode device 为 `1.0024x`，CUDA event ratio 为 `1.9784x`。
- 24 个 dtype/case 组合中 20 个三轮 p50 稳定胜出，4 个跨过 1.0，没有稳定 torch-faster case。
- BF16 `l4_b4_c128` 的 instrumented append ratio 为 `0.4980x`，但 complete-token p50 三轮均胜出，因此只记录为 profiler attribution anomaly。
- 每轮只有 20 repeats，nearest-rank p99 接近该轮最大值；p99 必须与范围一起报告，不作为稳定生产长尾结论。

## 16. 验收门槛

- `num_layers=2/4` CPU/reference transaction correctness 完整。
- global `seq_len` 对每个成功 token 只增长一次。
- 任一 layer 失败后，不留下 partial visible token、block leak 或 in-flight marker。
- 至少 `num_layers=2` fused CUDA + Triton 路径在 RTX 5070 通过 FP16/BF16。
- 单层 `DecodeEngine.step()` compatibility regression 通过。
- benchmark 绑定 commit、设备、layer 数、KV bytes、launch 数和计时边界。
- 文档与代码明确：这是 multi-layer decode transaction，不是完整 Transformer/model serving。

## 17. 当前状态

本文冻结 R2 的状态机、所有权、回滚、Engine API 与 benchmark 边界。R2-A/R2-B/R2-C correctness 与 R2-D 正式性能证据均已完成；稳定结论是 fused append/launch 优化能够转化为 multi-layer complete-token 收益，限制是小样本 p99 波动和少数 profiler attribution anomaly。证据提交 `67bee15` 已在 RTX 5070 通过 `337 passed, 25 subtests passed in 5.82s` 的无跳过完整回归，R2 项目闭环完成；clean-install、版本号与 tag 属于后续 release gate。

## 18. R4-A：Trusted CUDA Transaction Fast Path

### 18.1 动机

R2 的 fused transaction 每个 layer 都调用公开 raw primitive。该入口必须防御调用方提供的非法 CUDA 索引，因此同步检查：

```text
block id lower/upper bound
block offset lower/upper bound
position non-negative
```

这些检查通过五次 device reduction + `.item()` 完成，会让 host 等待当前 CUDA stream。对公开 `flashdec.fused_rope_kv_append()`，该安全语义必须保留；但 Engine transaction 的 block id、offset 和 position 全部来自当前 `PagedKVCache` allocator：

```text
block_id in deterministic free pool
block_offset = committed_seq_len % block_size
position = committed_seq_len >= 0
```

因此同一组 device-value 检查在 Cache-owned 路径中是重复验证。

### 18.2 信任边界

R4-A 将 fused append 拆成 checked 与 trusted 两条入口：

| 调用路径 | 结构检查 | device-value 检查 | 可见性 |
| --- | --- | --- | --- |
| `flashdec.fused_rope_kv_append()` | 保留 | 保留 | public raw primitive |
| `PagedKVCache.write_token_layer_fused_cuda()` | 保留 | 跳过 | authoritative public transaction API |
| `DecodeEngine -> PagedKVCache.write_token_layer_fused_cuda()` | 保留 | 跳过 | 只依赖 Cache public API |

trusted raw 路径仍检查 shape、dtype、device、contiguity、RoPE 参数，并要求 transaction metadata 为 contiguous `int64`。`begin_token()` 在 host 上一次性验证 block id/offset/position 与 request block table 的关系；后续 Cache API 不信任调用方修改过的 detached tensor，而是根据 transaction id 回查当前 open internal state，再 materialize allocator 生成的位置。公开 `flashdec` namespace 不导出 trusted raw primitive。

### 18.3 状态与失败语义不变

本优化只删除重复的 device reduction + `.item()` 检查，不改变：

- begin/preflight 与 physical block reservation；
- layer 顺序和 request row mapping；
- commit 时单次 `seq_len` 增长；
- exception 自动 abort、boundary block rollback 和 state version；
- Triton decode、kernel 配置或 shared-prefix ownership。

第一 slice 不同时修改 transaction metadata tensor materialization、output buffer、CUDA Graph、Scheduler 或 kernel launch geometry。每层 transaction view 仍可能产生 H2D materialization/copy，因此不能称为完全 sync/copy-free；这样 checked/trusted A/B 的差异可以收窄归因到 host-blocking device-value validation。

### 18.4 验收边界

- checked public primitive 对越界 block id/offset 与负 position 继续报错。
- trusted primitive 不进入 `flashdec` public namespace。
- 修改 detached transaction view 的位置 tensor 不改变 Cache 实际写入位置或 rollback ownership。
- checked/trusted 在 FP16/BF16、GQA、tail/boundary case 的 Q output 与 K/V cache 完全对齐。
- layer failure 后仍无 visible token、block leak 或 open transaction。
- 同 commit benchmark 交替 checked/trusted 顺序；正式 wall 仅使用 `synchronize + perf_counter + synchronize`，不在计时区间 record CUDA event。
- profiler attribution 与正式 wall 分离，报告 append host/device time 和 CUDA event count。
- profiler range 证据来自成对的原始 user annotation：CPU range 提供 inclusive host time，以保留 checked 路径子 `.item()` 的同步等待；同名 CUDA range 独立提供 device time。CPU/CUDA range 数都按真实调用逐个验证，二者不相加，也不要求 Triton kernel 必须关联为 CPU annotation child。
- 只有 complete-token p50 总体至少 `1.05x`，且正式矩阵全部 16 个 `dtype x case` 分组的五轮 p50 `[min,max]` 都不穿过 1，才冻结为性能收益；否则记录负结果并停止扩展到 CUDA Graph。

当前实现、benchmark harness 与 dependency-free validator tests 已完成；commit `1169cb8` 的 RTX 5070 focused CUDA correctness 为 `40 passed in 2.34s`。第一次 quick 的 strict summary 正确拒绝了零 append CPU attribution，根因是 runner 用同名 key 字典压平 profiler 的 CPU/CUDA 分组。commit `4ee5fab` 改读 CPU raw range 后，第二次 quick 又在 `flashdec::paged_decode` 观察到有效 CPU `116.108 us`、CPU event 自身 device total 为 0；这说明 Triton device event 没有挂成该 CPU annotation 的 child，并不表示 decode 没有执行。runner 因此改为 CPU host 与同名 CUDA device 分离取数。

commit `4e18f5d` 的第三次 RTX FP16 quick 已通过严格 2-row summary。`l2_b4_c32` trusted 相对 checked 的 complete-token p50/TPS 为 `1.7856x/1.8755x`；append CPU/device 为 `2.3751x/2.4540x`，CUDA events 从 `166` 降到 `106`，`aten::item` 与 local-scalar operator 计数分别从 `20` 降到 `0`。decode device 仅为 `1.0062x`，begin/commit 也接近中性，因此结果方向与“移除每层 device-value synchronization”一致。Profiler 是独立的 instrumented run，CPU/device attribution 不能相加或替代 complete-token wall；单 trial、单 dtype、缩小 context 也不能外推为稳定收益。当前 commit 的完整回归和 5-trial/160-row 正式矩阵仍未完成，R4-A 尚未冻结。transaction metadata 每 token 只 materialize 一次是后续独立 slice，必须在正式因果结论之后再评估。
