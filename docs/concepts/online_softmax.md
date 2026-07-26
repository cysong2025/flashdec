# Online Softmax 与 Decode Attention

Online softmax 允许 attention kernel 按 K/V block 扫描 context，同时维持数值稳定的 softmax 和输出累积，不需要物化完整 score matrix。它是 FlashDec dense/paged decode kernels 的数值基础。

## 1. Safe softmax

对当前 query 与历史 keys 的 score：

```text
s_i = dot(q, k_i) * sm_scale
```

直接计算 `exp(s_i)` 可能溢出。Safe softmax 先减去全局最大值：

```text
m   = max_i(s_i)
p_i = exp(s_i - m) / sum_j(exp(s_j - m))
out = sum_i(p_i * v_i)
```

PyTorch reference 可以清楚地表达这个定义，但在 kernel 中先保存全部 scores 会增加 context-sized 中间状态和额外访存。

## 2. 流式状态

按 block 遍历 K/V 时维护三个 FP32 状态：

```text
m   running maximum score
l   running normalized exponential sum
acc running weighted-value sum
```

在已经处理的 token 集合 `A` 上：

```text
l_A   = sum(i in A) exp(s_i - m_A)
acc_A = sum(i in A) exp(s_i - m_A) * v_i
```

最终输出为 `acc / l`。

## 3. 合并一个新 block

设旧状态为 `(m_old, l_old, acc_old)`，新 block 的 scores 为 `s_block`：

```text
m_block = max(s_block)
m_new   = max(m_old, m_block)
alpha   = exp(m_old - m_new)
p_block = exp(s_block - m_new)
```

将旧累积重新缩放到新的最大值基准：

```text
l_new   = l_old * alpha + sum(p_block)
acc_new = acc_old * alpha + sum(p_block * v_block)
```

由于旧项和新项都相对同一个 `m_new` 归一化，重复该合并直到 context 结束后，`acc / l` 与一次性 safe softmax 的数学结果一致。

## 4. Mask、尾页与空 context

Paged decode 的最后一页通常未填满。无效 token 必须同时满足：

- 不参与 block maximum；
- 不贡献 denominator；
- 不读取未拥有的 physical K/V slot；
- 不贡献 weighted-value accumulation。

实现上通常把无效 score 视为 `-inf`，并对 load/store 使用显式 mask。FlashDec 还定义 `seq_len == 0` 的输出为全零，避免 `0/0` 或全 `-inf` reduction 产生 NaN。该语义同时由 dense reference、paged reference 和 Triton kernels 实现。

## 5. MHA、GQA 与 MQA

Online softmax 本身不改变 head mapping。每个 query head 先映射到对应 KV head：

```text
queries_per_kv_head = num_q_heads / num_kv_heads
kv_head = q_head // queries_per_kv_head
```

随后对该 KV head 的有效历史 tokens 执行相同的流式归约。MHA 中映射为一对一；GQA/MQA 中多个 query heads 读取同一 KV head，但各自维护独立 `(m, l, acc)`。

## 6. Dense 与 Paged 寻址

Dense decode 直接按 sequence/token stride 读取连续 K/V。Paged decode 先执行：

```text
token index
  -> logical page + page offset
  -> block_table lookup
  -> physical page + page offset
```

地址来源不同，但 score、mask、running-state update 和最终归一化必须与 dense 数学一致。可读定义见 [`dense_decode_attention_ref`](../../flashdec/reference.py) 与 [`paged_decode_attention_ref`](../../flashdec/paged_reference.py)；kernel 实现在 [`dense_decode.py`](../../flashdec/kernels/dense_decode.py) 与 [`paged_decode.py`](../../flashdec/kernels/paged_decode.py)。

## 7. 数值与验证约束

- Q/K score、`m`、`l` 和 `acc` 使用 FP32 计算或累积，再转换回输出 dtype。
- reference 与 kernel 使用相同 `sm_scale`、有效 token 范围和 head mapping。
- correctness 同时覆盖变长 context、尾页、空 context、MHA/GQA/MQA、FP16/BF16 和自定义 scale。
- 性能测量在 reference parity 通过后开始，并排除 JIT 与输入构造。

Online softmax 减少中间 score materialization，但不自动证明整个 decode runtime 更快。长 context 仍需要读取大部分 K/V，实际性能还受 page layout、block table 间接寻址、launch geometry 和完整 Engine 路径影响。

## 8. 延伸阅读

- [FlashDec 总体设计](../design.md)
- [Paged KV Cache 设计](../design_paged_kv.md)
- [研究问题](../research_questions.md)
- [Primary references](../references.md)
