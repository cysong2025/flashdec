# Week 3 状态记录

## 本周主题

attention reference 与 dense decode baseline。

## 本周目标

- 先把 decode attention 的张量语义定义清楚。
- 写出一个朴素、可信、容易解释的 PyTorch dense decode attention reference。
- 让后续 Week 4 的 Triton dense decode kernel 能对齐这个 reference。
- 支持 GQA/MQA 场景中的 q head 到 kv head 映射。
- 用随机 shape tests 和手写小例子验证 reference 语义。

## 为什么 Week 3 很关键

Week 1 和 Week 2 练的是 Triton 基础、访存、benchmark 和 profiling。Week 3 开始进入 FlashDec 的真正主线：LLM decode attention。

本周先不写 Triton attention kernel。重点是把正确性标准先立住：

```text
q: 当前 token query
k/v: dense KV cache
seq_lens: 每个 sequence 的有效 context 长度
out: 当前 token 对历史 context attention 后的输出
```

后续所有 Triton kernel、paged reference、paged decode kernel 都要和这个 dense PyTorch reference 对齐。

## 输入输出约定

计划实现的 dense decode reference 使用以下 shape：

```text
q:        [num_seqs, num_q_heads, head_dim]
k_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
v_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
seq_lens: [num_seqs]
out:      [num_seqs, num_q_heads, head_dim]
```

GQA/MQA 映射：

```python
kv_head = q_head // (num_q_heads // num_kv_heads)
```

约束：

- `num_q_heads` 必须能被 `num_kv_heads` 整除。
- `seq_lens[i]` 表示第 `i` 个 sequence 可见的历史 token 数。
- attention 只在 `[0, seq_lens[i])` 范围内计算。
- softmax 需要做数值稳定处理。

## 本周编码任务

1. 在 `flashdec/reference.py` 中新增 dense decode reference：

```text
dense_decode_attention_ref
```

建议函数参数：

```python
def dense_decode_attention_ref(q, k_cache, v_cache, seq_lens, sm_scale=None):
    ...
```

2. 写手写小例子测试：

```text
tests/test_decode_reference.py
```

小例子重点验证：

- 单 sequence。
- 单 q head / 单 kv head。
- `seq_lens` 小于 `max_seq_len` 时不会读 padding token。
- softmax 权重可以人工算出或通过简单 PyTorch 公式对齐。

3. 写随机 shape tests。

建议覆盖：

```text
num_seqs: 1, 2, 4
num_q_heads: 1, 4, 8
num_kv_heads: 1, 2, 4
head_dim: 16, 64, 128
max_seq_len: 1, 17, 128
dtype: float16, float32
```

重点 shape：

- MHA：`num_q_heads == num_kv_heads`
- GQA：`num_q_heads > num_kv_heads`
- MQA：`num_kv_heads == 1`
- 变长 batch：不同 sequence 的 `seq_lens` 不同。

4. 写 reference benchmark：

```text
benchmarks/run_decode_reference.py
```

这个 benchmark 不追求高性能，目的是给 Week 4 的 Triton dense decode kernel 提供 baseline。

建议记录：

- `num_seqs`
- `num_q_heads`
- `num_kv_heads`
- `head_dim`
- `max_seq_len`
- dtype
- `mean_ms`
- `p50_ms`
- `p90_ms`

5. 写 `docs/design.md` 初稿。

设计文档至少包含：

- FlashDec 当前范围。
- dense decode attention 的输入输出。
- GQA/MQA head 映射。
- 为什么需要 `seq_lens`。
- 为什么 dense reference 是后续 paged reference 的 correctness anchor。
- Week 4 Triton kernel 的计划。

## 当前已完成

- 新增 dense decode attention PyTorch reference：
  - `flashdec.reference.dense_decode_attention_ref`
- 新增 reference correctness tests：
  - `tests/test_decode_reference.py`
- 新增 reference benchmark：
  - `benchmarks/run_decode_reference.py`
- 新增设计文档初稿：
  - `docs/design.md`

实现范围：

- 支持 MHA/GQA/MQA。
- 支持 variable `seq_lens`。
- 支持 `sm_scale=None` 时默认使用 `head_dim ** -0.5`。
- reference 内部使用 FP32 计算 score、softmax 和 value accumulation，再转回输入 dtype。
- `seq_len == 0` 时输出 zero，便于后续处理空上下文边界。

## 本周学习任务

重点理解：

- Transformer attention 的 Q/K/V 含义。
- prefill attention 与 decode attention 的区别。
- MHA、GQA、MQA 的区别。
- safe softmax：
  - 先减最大值。
  - 再 `exp`。
  - 再除以 sum。
- online softmax 的基本动机，为 Week 4 做准备。
- vLLM Paged Attention 中 QK、softmax、value accumulation 的语义。

## 推荐执行顺序

### Day 1：语义设计

- 读 Week 3 计划。
- 写 `docs/design.md` 初稿结构。
- 明确函数签名和 shape 约束。

### Day 2：实现 dense reference

- 在 `flashdec/reference.py` 中实现 `dense_decode_attention_ref`。
- 先用 Python for-loop 写清楚语义。
- 不要过早优化。

### Day 3：手写小例子测试

- 写 `tests/test_decode_reference.py`。
- 用小 shape 验证 `seq_lens` 和 softmax。
- 覆盖 MHA 和 MQA 的最小例子。

### Day 4：随机 shape tests

- 增加 GQA/MQA 随机测试。
- 覆盖 FP16/FP32。
- 检查 dtype 和输出 shape。

### Day 5：benchmark baseline

- 写 `benchmarks/run_decode_reference.py`。
- 跑一组小 shape 和中等 shape。
- 记录结果但不做性能夸张结论。

### Day 6：整理设计文档

- 补 `docs/design.md`。
- 把 reference 语义、shape、GQA 映射写清楚。

### Day 7：复盘和提交

- 更新本文件的完成状态。
- 提交 GitHub。

## RTX 5070 验证记录（2026-06-26）

### Correctness

用户已说明 Week 3 代码已经在 RTX 5070 上完成测试。原始 pytest 输出待补充到本节，建议补充以下命令的完整输出：

```bash
pytest tests/test_decode_reference.py
```

### Benchmark

运行命令：

```bash
python benchmarks/run_decode_reference.py --output benchmarks/results/week3_decode_reference.csv
```

结果文件：

```bash
benchmarks/results/week3_decode_reference.csv
```

硬件和软件环境：

- GPU：NVIDIA GeForce RTX 5070
- PyTorch：2.11.0+cu128
- CUDA：12.8
- dtype：float16
- repeats：30

| num_seqs | num_q_heads | num_kv_heads | head_dim | max_seq_len | actual_seq_len | mean_ms | p50_ms | p90_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8 | 64 | 128 | 70-70 | 1.480278 | 1.029536 | 1.448640 | 0.911712 | 12.898208 |
| 4 | 8 | 8 | 64 | 512 | 325-451 | 4.097298 | 3.687136 | 4.644384 | 3.301856 | 9.444896 |
| 8 | 16 | 4 | 64 | 512 | 307-502 | 15.455861 | 14.764896 | 16.754496 | 13.650176 | 27.230560 |
| 16 | 16 | 4 | 128 | 1024 | 566-1024 | 31.033368 | 29.498720 | 35.093216 | 27.113216 | 47.425087 |

### 观察

- dense decode reference benchmark 已能稳定生成 CSV，说明 Week 3 的 baseline 路径跑通。
- latency 随 `num_seqs`、head 数量、`head_dim`、`max_seq_len` 和实际 `seq_lens` 增大而明显上升。
- 第 4 组 shape 的 p50 已达到约 29.5 ms，说明朴素 PyTorch dense reference 只适合作为 correctness anchor，不适合真实 decode serving。
- 第 3、4 组覆盖 GQA 场景：`num_q_heads=16`，`num_kv_heads=4`，后续 Triton kernel 必须保持相同 head 映射语义。
- `max_ms` 明显高于 p50/p90，说明 benchmark 中存在尾部抖动；后续对比 Triton kernel 时应优先看 p50/p90，同时保留 mean/max 用于观察稳定性。
- Week 4 的目标不是和 cuBLAS 类 matmul 对比，而是用 Triton 把 decode attention 中 QK、softmax、V accumulation 合成一个 kernel，减少 PyTorch reference 的多次调度和中间张量开销。

## Week 3 完成判定

- `dense_decode_attention_ref` 实现完成。
- 手写小例子和随机 shape tests 已在 RTX 5070 上运行，pytest 原始输出待补充。
- 支持 MHA/GQA/MQA。
- 支持变长 `seq_lens`。
- `benchmarks/run_decode_reference.py` 已生成 baseline CSV：`benchmarks/results/week3_decode_reference.csv`。
- `docs/design.md` 初稿完成。
- 你能口头解释：
  - decode attention 与 prefill attention 的区别。
  - `q/k/v` 的 shape 含义。
  - `seq_lens` 为什么必须参与 attention。
  - GQA/MQA 的 head 映射。
  - dense reference 为什么是后续 Triton/paged kernel 的 correctness anchor。
