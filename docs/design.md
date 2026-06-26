# FlashDec 设计文档

本文是 FlashDec 的设计文档初稿，当前重点定义 dense decode attention reference 的语义。后续 dense Triton kernel、Paged KV Cache、paged decode attention 都要对齐这里的输入输出约定。

## 1. 项目范围

FlashDec 聚焦 LLM decode 阶段的高性能 attention 与 Paged KV Cache。项目不实现完整 serving engine，而是围绕以下核心能力展开：

- PyTorch reference：定义正确性标准。
- Triton kernel：实现 dense decode attention 和 paged decode attention。
- Paged KV Cache：用 block table 管理变长请求的 K/V cache。
- correctness、benchmark、profiling：形成可复现的工程闭环。

## 2. Prefill 与 Decode 的区别

prefill 阶段处理 prompt 中的一段 token，通常有较大的矩阵乘法和较高并行度。

decode 阶段每次只处理一个新 token，但要读取历史所有 token 的 K/V cache。此时：

```text
q: 当前 token 的 query
k/v cache: 历史 token 的 key/value
out: 当前 token 聚合历史上下文后的 attention 输出
```

decode attention 常见瓶颈不是算力不够，而是 K/V cache 读取带宽、访存布局和动态 batch 管理。

## 3. Dense Decode Attention 语义

Week 3 的 dense reference 使用 dense K/V cache，不涉及分页。

输入 shape：

```text
q:        [num_seqs, num_q_heads, head_dim]
k_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
v_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
seq_lens: [num_seqs]
```

输出 shape：

```text
out: [num_seqs, num_q_heads, head_dim]
```

对每个 sequence 和 q head：

1. 根据 GQA/MQA 映射找到对应 kv head。
2. 只读取 `[0, seq_lens[seq_idx])` 范围内的 K/V。
3. 计算 `score = dot(q, k) * sm_scale`。
4. 使用数值稳定 softmax。
5. 用 softmax 权重加权求和 V。

## 4. GQA/MQA Head 映射

FlashDec 使用如下映射：

```python
kv_head = q_head // (num_q_heads // num_kv_heads)
```

约束：

- `num_q_heads` 必须能被 `num_kv_heads` 整除。
- MHA：`num_q_heads == num_kv_heads`。
- GQA：`num_q_heads > num_kv_heads`，多个 q head 共享一个 kv head。
- MQA：`num_kv_heads == 1`，所有 q head 共享同一个 kv head。

## 5. seq_lens 的作用

`seq_lens` 表示每个 sequence 的有效历史长度。dense cache 通常 padding 到同一个 `max_seq_len`，但 padding token 不能参与 attention。

例如：

```text
max_seq_len = 8
seq_lens[0] = 5
```

第 0 个 sequence 只能读取 K/V 的前 5 个 token，后 3 个 padding 位置必须忽略。

这件事对后续 paged attention 也很关键，因为每个 request 的逻辑长度不同。

## 6. 数值稳定 Softmax

reference 使用 safe softmax：

```text
scores = scores - max(scores)
probs = exp(scores) / sum(exp(scores))
```

这样可以避免 score 较大时 `exp` 溢出。

Week 4 的 Triton dense decode kernel 会进一步使用 online softmax，避免在 kernel 内 materialize 整个 attention score。

## 7. Correctness Anchor

`dense_decode_attention_ref` 是后续实现的 correctness anchor：

- Week 4 dense Triton decode kernel 要与它对齐。
- Week 5 paged reference 要与它对齐。
- Week 6/7 paged Triton decode kernel 也要与它对齐。

这也是 Week 3 先写 reference、测试和 benchmark，而不是直接写 Triton attention kernel 的原因。

Paged KV Cache 的 block table、physical block layout 和 paged reference 语义见 `docs/design_paged_kv.md`。

## 8. Week 4 Triton Kernel 计划

Week 4 的 dense decode Triton kernel 计划：

- 每个 Triton program 处理一个 `(sequence, q_head)`。
- 沿 context 维度分 block 读取 K/V。
- 使用 FP32 accumulation。
- 实现 running max、running exp sum、running output accumulator。
- 先支持 `head_dim=64`，再支持 `head_dim=128`。
- correctness 对齐 `dense_decode_attention_ref`。
