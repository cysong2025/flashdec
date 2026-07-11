# Paged KV Cache 设计说明

本文记录 Paged KV Cache 的数据结构、paged PyTorch reference，以及 runtime v2 的 request 生命周期和 physical block allocator。它既是 paged decode Triton kernel 的正确性基准，也是后续 DecodeEngine 的内存管理基础。

## 1. 设计目标

decode 阶段每个 request 的上下文长度不同。如果继续使用 dense KV cache，短序列会浪费 padding 空间，动态 batch 也很难管理。Paged KV Cache 把每个 request 的逻辑 token 序列切成固定大小 block，再通过 block table 映射到物理 block。

Week 5 的目标不是优化性能，而是把语义定义清楚：

- 每个 request 维护自己的 logical block list。
- 每个 logical block 指向一个 physical block。
- append 一个 token 时写入当前尾部 block；尾部 block 满后分配新 physical block。
- `block_tables` 把 request 的 logical block index 映射到 physical block id。
- paged reference 输出必须和 dense reference 对齐。

## 2. Cache Layout

`PagedKVCache` 的物理存储 layout：

```text
k_cache / v_cache:
[num_layers, max_blocks, num_kv_heads, block_size, head_dim]
```

storage 保留 `num_layers` 维度作为未来扩展点，但 runtime v2 为避免错误共享 seq_len/block ownership，当前显式要求 `num_layers=1`。

paged reference 使用单层 cache：

```text
k_cache / v_cache:
[num_blocks, num_kv_heads, block_size, head_dim]
```

对应调用方式：

```python
paged_decode_attention_ref(
    q,
    cache.k_cache[layer_idx],
    cache.v_cache[layer_idx],
    block_tables,
    seq_lens,
)
```

## 3. Logical Token 到 Physical Address

给定：

```text
block_size = 16
logical_token_idx = t
```

映射关系：

```text
logical_block = t // block_size
block_offset = t % block_size
physical_block = block_tables[seq_idx, logical_block]

K/V 物理位置:
[physical_block, kv_head, block_offset, head_dim]
```

一个例子：

```text
request A seq_len = 33
block_size = 16
request A logical blocks = [0, 1, 2]
block_tables[A] = [7, 2, 9]
```

则第 32 个 token：

```text
logical_block = 32 // 16 = 2
block_offset = 32 % 16 = 0
physical_block = block_tables[A, 2] = 9
```

所以读取：

```text
k_cache[9, kv_head, 0, :]
v_cache[9, kv_head, 0, :]
```

## 4. 当前 API

创建 cache：

```python
cache = PagedKVCache(
    num_layers=1,
    num_kv_heads=8,
    head_dim=128,
    block_size=16,
    max_blocks=4096,
    dtype=torch.float16,
    device="cuda",
)
```

append 一个 decode token：

```python
block_tables = cache.append(
    layer_idx=0,
    request_ids=[101, 202],
    k=k,  # [num_requests, num_kv_heads, head_dim]
    v=v,  # [num_requests, num_kv_heads, head_dim]
)
```

生成 kernel/reference 需要的元数据：

```python
block_tables = cache.block_tables([101, 202])
seq_lens = cache.seq_lens_tensor([101, 202])
```

为了和 dense reference 对齐，可以 materialize 一份 dense KV：

```python
dense_k, dense_v, dense_seq_lens = cache.to_dense(layer_idx=0, request_ids=[101, 202])
```

decode 完成后结束或取消 request，并查询 allocator：

```python
released = cache.finish_request(request_id=101)
released = cache.cancel_request(request_id=202)
state = cache.request_state(101)
metrics = cache.metrics()
cache.validate_invariants()
```

## 5. Paged Reference 语义

`paged_decode_attention_ref` 的输入：

```text
q:            [num_seqs, num_q_heads, head_dim]
k_cache:      [num_blocks, num_kv_heads, block_size, head_dim]
v_cache:      [num_blocks, num_kv_heads, block_size, head_dim]
block_tables: [num_seqs, max_blocks_per_seq]
seq_lens:     [num_seqs]
out:          [num_seqs, num_q_heads, head_dim]
```

对每个 `(sequence, q_head)`：

1. 根据 `seq_lens[seq_idx]` 得到有效 token 数。
2. 根据 `block_tables` 找到对应 physical blocks。
3. 按 logical token 顺序还原 K/V。
4. 使用与 dense reference 相同的 GQA/MQA 映射：

```python
kv_head = q_head // (num_q_heads // num_kv_heads)
```

5. 使用 FP32 计算 score、softmax 和 value accumulation。
6. `seq_len == 0` 时输出 zero。

## 6. Correctness Anchor

Week 5 的核心验证方式：

```text
PagedKVCache append tokens
        |
        |-- cache.to_dense(...)
        |       -> dense_decode_attention_ref(...)
        |
        |-- cache.block_tables(...) + physical K/V
                -> paged_decode_attention_ref(...)

比较两个输出是否一致
```

这能验证三件事：

- append 写入位置正确。
- block table 映射正确。
- paged reference 与 dense reference 的 attention 语义一致。

## 7. 当前限制

- runtime v2 当前只支持 `num_layers=1`；构造多 layer cache 会显式报错，避免错误共享 seq_len/block ownership。
- finished/cancelled request id 不能重新激活；终态记录当前保留用于状态查询和 lifecycle 计数。
- 释放 physical block 时不会清零 K/V。正确性依赖 block ownership、block table 和 seq_len；新 request 的有效 token 不会读取旧尾部数据。
- 当前 allocator 和 request scheduler 位于 Python 层，目标是先定义清楚语义，不代表最终高吞吐 serving 实现。
- paged Triton kernel 已在 RTX 5070 验证 `head_dim=64/128`、FP16/BF16 和 `block_size=8/16/32`；当前通用配置为 block32。
- 当前 reference 会显式 gather logical K/V，适合作 correctness，不适合当性能实现。

## 8. Runtime v2 状态机与 allocator

PagedKVCache v1 只验证 append 和 block table；v2 增加以下状态机：

```text
add/implicit append admission -> active -> finished
                                active -> cancelled
```

终态转换是单向的。finish/cancel 会释放全部 physical blocks、清空 request 的 block ownership，但保留历史 `seq_len` 和终态供 `request_state()` 查询。终态 request 不允许 append、生成 block table、materialize dense cache 或重新激活。

free list 的规则：

- 初始包含 `[0, max_blocks)` 的全部 physical block。
- 首次分配按 block id 顺序取出。
- finish/cancel 释放的 block 放到 free list 前端，后续 request 优先复用。
- 物理 K/V 不清零；新 request 只通过自己的 seq_len 和 block table 暴露有效 token。

批量 append 的容量原子性：

1. 完成 request id、shape、device、dtype 和终态检查。
2. 对整个 batch 计算 `needed_new_blocks`。
3. 若容量不足，直接报错；不创建新 request，不分配 block，也不增长已有 seq_len。
4. 容量足够后才按 batch row 写入 K/V 并提交状态。

`metrics()` 当前报告：

- used/free/max blocks 与 block utilization。
- active/reserved tokens 与 internal fragmentation。
- allocation、fresh allocation、free、reuse、capacity failure 计数。
- active、finished、cancelled request 计数。

`validate_invariants()` 用于测试和 debug，验证：

- owned blocks 和 free blocks 无交集、无重复。
- 两者并集恰好覆盖整个 physical block pool。
- active request 的 block 数和 seq_len 一致。
- finished/cancelled request 不再持有 block。

v2 correctness 不只比较 attention 输出，还要覆盖请求状态机：

```text
add -> append -> finish -> block reuse
add -> append -> cancel -> block reuse
capacity exhausted -> append fails without partial mutation
mixed active/finished requests -> block table and seq_lens remain correct
```

上述代码已实现，当前等待 RTX 5070 focused/full correctness 验证。

## 9. Week 6 接口衔接

Week 6 paged decode kernel 应接收：

```text
q
k_cache / v_cache physical storage
block_tables
seq_lens
```

Week 6 v1 当前限制：

- `block_size = 8/16/32`，当前通用 benchmark 默认值为 32
- `head_dim = 64/128`
- FP16/BF16
- 保留 GQA/MQA head 映射。

正确性仍然对齐：

```python
paged_decode_attention_ref(...)
```
