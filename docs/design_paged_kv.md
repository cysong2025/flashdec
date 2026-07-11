# Paged KV Cache 设计说明

本文记录 Week 5 的 Paged KV Cache 数据结构和 paged PyTorch reference。它是 Week 6 paged decode Triton kernel 的正确性基准。

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

Week 5 当前验证单 layer 路径，保留 `num_layers` 维度是为了后续扩展到多层 KV cache。

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

- 当前只实现 append，不实现 free/reuse request。
- 当前 Week 5 tests 验证单 layer 路径。
- Week 7 已将 paged Triton kernel 扩展到 `head_dim=64/128`、FP16/BF16；后续工程迭代已补充 `block_size=8/16/32` 代码与测试路径，其中 8/32 待 RTX 5070 上板确认。
- 当前 reference 会显式 gather logical K/V，适合作 correctness，不适合当性能实现。

## 8. Week 6 接口衔接

Week 6 paged decode kernel 应接收：

```text
q
k_cache / v_cache physical storage
block_tables
seq_lens
```

Week 6 v1 当前限制：

- `block_size = 8/16/32`，当前实测默认值为 16
- `head_dim = 64/128`
- FP16/BF16
- 保留 GQA/MQA head 映射。

正确性仍然对齐：

```python
paged_decode_attention_ref(...)
```
