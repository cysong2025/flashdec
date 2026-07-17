# Week 4 状态记录

## 本周主题

dense decode attention Triton kernel。

## 阶段目标

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

运行命令：

```bash
python benchmarks/run_dense_decode.py --output benchmarks/results/week4_dense_decode.csv
```

结果文件：

```bash
benchmarks/results/week4_dense_decode.csv
```

硬件和软件环境：

- GPU：NVIDIA GeForce RTX 5070
- PyTorch：2.11.0+cu128
- CUDA：12.8
- dtype：float16
- repeats：30
- Triton 配置：`block_seq=64`，`num_warps=4`

| shape `(num_seqs,q_heads,kv_heads,head_dim,max_seq_len)` | actual_seq_len | ref mean_ms | ref p50_ms | triton mean_ms | triton p50_ms | triton p90_ms | speedup_vs_ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1,8,8,64,128)` | 69-69 | 1.057916 | 1.021696 | 0.029214 | 0.032192 | 0.040384 | 36.2128 |
| `(4,8,8,64,512)` | 302-489 | 3.687276 | 3.687296 | 0.030478 | 0.019904 | 0.047008 | 120.9821 |
| `(8,16,4,64,512)` | 259-482 | 15.873489 | 15.055104 | 0.027958 | 0.021184 | 0.051520 | 567.7538 |
| `(16,16,4,128,1024)` | 512-994 | 31.804113 | 29.516577 | 0.093687 | 0.084320 | 0.126048 | 339.4703 |

观察：

- Triton dense decode kernel 在默认 shape sweep 中明显快于 Week 3 的朴素 PyTorch reference。
- 最大 shape 上，PyTorch reference mean 为 31.804113 ms，Triton mean 为 0.093687 ms，说明把 QK、softmax 和 V accumulation 合成一个 kernel 后，Python 循环、多次 kernel launch 和中间张量开销被大幅减少。
- 第 3、4 组覆盖 GQA：`num_q_heads=16`，`num_kv_heads=4`，benchmark 说明当前 GQA head 映射路径可以正常参与性能测试。
- 这里的 `speedup_vs_ref` 是相对朴素 PyTorch reference 的加速，不代表已经达到 FlashInfer/vLLM 等成熟实现水平。后续 Week 8-9 仍需要 profiling 和优化。

待补充不同 `BLOCK_SEQ` 对比：

```bash
python benchmarks/run_dense_decode.py --block-seq 16 --output benchmarks/results/week4_dense_decode_block16.csv
python benchmarks/run_dense_decode.py --block-seq 32 --output benchmarks/results/week4_dense_decode_block32.csv
python benchmarks/run_dense_decode.py --block-seq 64 --output benchmarks/results/week4_dense_decode_block64.csv
python benchmarks/run_dense_decode.py --block-seq 128 --output benchmarks/results/week4_dense_decode_block128.csv
```

## 上板后要记录

- `BLOCK_SEQ` 对 p50/p90 的影响。
- 哪些 shape 仍然不够快，后续需要 profiling。

## Week 4 完成判定

- `dense_decode_attention` Triton kernel 实现完成。
- correctness tests 在 RTX 5070 上通过：`14 passed in 5.62s`。
- benchmark 已生成：`benchmarks/results/week4_dense_decode.csv`。
- 默认 `block_seq=64` 下，Triton dense decode 相比朴素 PyTorch reference 的 mean speedup 为 36.2128x 到 567.7538x。
- 实现与笔记覆盖 online softmax 的三个状态变量：
  - running max。
  - running exp sum。
  - running output accumulator。
- 设计记录说明 dense decode kernel 与后续 paged attention 的语义衔接。
