# Online Softmax 笔记

本笔记对应 Week 4：dense decode attention Triton kernel。

目标是理解为什么 attention kernel 不能总是把完整 score 矩阵 materialize 出来，以及如何在遍历 K/V cache 的过程中一边做 softmax、一边累加输出。

## 1. 普通 Safe Softmax

对一组 score：

```text
s_i = q dot k_i
```

普通 softmax 是：

```text
p_i = exp(s_i) / sum_j exp(s_j)
```

如果 `s_i` 很大，`exp(s_i)` 可能溢出。所以实际实现通常先减最大值：

```text
m = max_i s_i
p_i = exp(s_i - m) / sum_j exp(s_j - m)
```

这就是 safe softmax。

## 2. 为什么 Decode Attention 不想存完整 Scores

decode attention 每次只处理当前 token：

```text
q: 当前 token 的 query
k/v cache: 历史 token 的 K/V
```

朴素写法会先算出完整 scores：

```text
scores = q @ k_cache
probs = softmax(scores)
out = probs @ v_cache
```

这在 PyTorch reference 中很清楚，但在 kernel 里会带来问题：

- 需要把 `scores` 中间结果写到显存或占用大量片上资源。
- context 越长，中间 score 越大。
- decode attention 常常受 K/V cache 读取带宽限制，额外中间张量会放大访存压力。

Week 4 的 dense Triton kernel 采用 online softmax：沿 context 分 block 读取 K/V，不保存完整 scores。

## 3. Online Softmax 的三个状态

kernel 遍历 context block 时维护三个状态：

```text
m: 当前已经看过 scores 的 running max
l: 当前 softmax denominator 的 running exp sum
acc: 当前输出向量的 running weighted value sum
```

其中：

```text
acc = sum_i exp(s_i - m) * v_i
l   = sum_i exp(s_i - m)
out = acc / l
```

## 4. 合并一个新 Block

假设旧状态是：

```text
m_old
l_old
acc_old
```

新 block 的 score 是 `s_block`，先算：

```text
m_block = max(s_block)
m_new = max(m_old, m_block)
```

因为 max 变了，旧的 `l_old` 和 `acc_old` 需要按新的 max 重新缩放：

```text
alpha = exp(m_old - m_new)
p_block = exp(s_block - m_new)
```

更新：

```text
l_new = l_old * alpha + sum(p_block)
acc_new = acc_old * alpha + sum(p_block * v_block)
```

遍历完所有 K/V block 后：

```text
out = acc / l
```

这就是 online softmax 的核心。

## 5. Week 4 Kernel 中的对应变量

`flashdec/kernels/dense_decode.py` 中：

```text
m_i: running max
l_i: running exp sum
acc: running output accumulator
```

每个 Triton program 处理一个：

```text
(sequence, q_head)
```

每个 program 会沿 context 维度按 `BLOCK_SEQ` 遍历：

```text
K block -> scores -> online softmax update -> V block -> acc update
```

最后写出：

```text
out[sequence, q_head, :]
```

## 6. 和 Week 3 Reference 的关系

Week 3 的 `dense_decode_attention_ref` 是 correctness anchor。它可以慢，但语义必须清楚。

Week 4 的 `dense_decode_attention` 要和 reference 对齐：

- 同样的 `q/k_cache/v_cache/seq_lens` shape。
- 同样的 MHA/GQA/MQA head 映射。
- 同样只读取 `[0, seq_lens[i])` 的有效 context。
- 同样使用 FP32 score、softmax 和 value accumulation。

区别是：

- reference 用 PyTorch 操作表达语义。
- Triton kernel 用 online softmax 合并 QK、softmax、V accumulation。

## 7. 本周需要观察的性能现象

运行 Week 4 benchmark 后重点看：

- Triton dense decode 是否快于 Week 3 naive PyTorch reference。
- context 变长时 latency 是否接近线性增长。
- GQA/MQA 下 `num_kv_heads` 更少是否影响 K/V 读取量。
- p50 和 p90 是否稳定。
- `BLOCK_SEQ` 从 16、32、64、128 改变时，性能是否变化明显。

如果某些 shape 上 Triton 没有明显更快，先不要急着优化。优先确认：

- correctness 是否稳定。
- kernel 是否真的合并了 reference 中的多步 PyTorch 操作。
- benchmark 是否排除了编译时间。
- 是否存在过小 shape 下 launch overhead 占主导的问题。
