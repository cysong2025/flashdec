# Week 4 状态记录

## 本周主题

dense decode attention Triton kernel。

## 本周目标

- 写出第一个真正的 decode attention Triton kernel。
- 使用 online softmax 在 kernel 内部完成 QK、softmax 和 V accumulation。
- 对齐 Week 3 的 `dense_decode_attention_ref`。
- 先支持 dense KV cache，为 Week 5-7 的 paged KV cache 和 paged decode kernel 打基础。

## 当前已完成

- 新增 Triton dense decode kernel：
  - `flashdec.kernels.dense_decode.dense_decode_attention`
- 新增 correctness tests：
  - `tests/test_dense_decode.py`
- 新增 benchmark 脚本：
  - `benchmarks/run_dense_decode.py`
- 新增 online softmax 笔记：
  - `docs/notes/online_softmax.md`

## Kernel 实现范围

输入输出 shape 与 Week 3 reference 保持一致：

```text
q:        [num_seqs, num_q_heads, head_dim]
k_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
v_cache:  [num_seqs, max_seq_len, num_kv_heads, head_dim]
seq_lens: [num_seqs]
out:      [num_seqs, num_q_heads, head_dim]
```

当前实现：

- 每个 Triton program 处理一个 `(sequence, q_head)`。
- 使用 GQA/MQA 映射：

```python
kv_head = q_head // (num_q_heads // num_kv_heads)
```

- 沿 context 维度按 `BLOCK_SEQ` 遍历 K/V。
- 使用 FP32 计算 score、softmax 和 value accumulation。
- 使用 online softmax 维护：
  - `m_i`: running max。
  - `l_i`: running exp sum。
  - `acc`: running output accumulator。
- 支持 `head_dim=64` 和 `head_dim=128`。
- 支持 `float16` 和 `float32` 输入。
- 支持 `seq_len == 0` 时输出 zero。

## 当前环境限制

当前 Codex 环境没有 RTX 5070 / CUDA GPU，无法本地运行 Triton kernel correctness 和 benchmark。

本次提交只做了可在当前环境完成的静态验证：

- Python 语法编译。
- 文件结构和 import 路径检查。
- 文档记录。

## RTX 5070 验证记录（2026-06-26）

### Correctness

运行命令：

```bash
pytest tests/test_dense_decode.py
```

结果：

```text
collected 14 items
tests/test_dense_decode.py .............. [100%]
14 passed in 5.62s
```

这说明 Week 4 dense decode Triton kernel 的 correctness tests 已经在 RTX 5070 环境通过。覆盖内容包括 padding mask、`seq_len == 0`、MHA/GQA/MQA、`head_dim=64`、`head_dim=128`、variable `seq_lens`、自定义 `sm_scale`，以及不同 `BLOCK_SEQ` 配置。

### Benchmark

待在 RTX 5070 上运行。

默认 shape sweep：

```bash
python benchmarks/run_dense_decode.py --output benchmarks/results/week4_dense_decode.csv
```

不同 `BLOCK_SEQ` 对比：

```bash
python benchmarks/run_dense_decode.py --block-seq 16 --output benchmarks/results/week4_dense_decode_block16.csv
python benchmarks/run_dense_decode.py --block-seq 32 --output benchmarks/results/week4_dense_decode_block32.csv
python benchmarks/run_dense_decode.py --block-seq 64 --output benchmarks/results/week4_dense_decode_block64.csv
python benchmarks/run_dense_decode.py --block-seq 128 --output benchmarks/results/week4_dense_decode_block128.csv
```

## 上板后要记录

- benchmark CSV 输出路径。
- Triton dense decode 相比 Week 3 PyTorch reference 的 `speedup_vs_ref`。
- `BLOCK_SEQ` 对 p50/p90 的影响。
- 哪些 shape 仍然不够快，后续需要 profiling。

## Week 4 完成判定

- `dense_decode_attention` Triton kernel 实现完成。
- correctness tests 在 RTX 5070 上通过：`14 passed in 5.62s`。
- benchmark 待生成：`benchmarks/results/week4_dense_decode.csv`。
- 能解释 online softmax 的三个状态变量：
  - running max。
  - running exp sum。
  - running output accumulator。
- 能说明 dense decode kernel 为什么是 Week 5 paged attention 的前置步骤。
