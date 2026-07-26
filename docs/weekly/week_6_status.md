# Week 6 状态记录

## 阶段主题

paged decode kernel v1。

## 阶段目标

- 理解 paged decode attention 中 logical token 到 physical block 的间接索引。
- 把 Week 4 dense decode kernel 的 online softmax 改造成 paged KV cache 版本。
- 掌握最后一个 block 的 mask 处理，避免 padding token 参与 attention。
- 建立第一版 paged decode correctness 和 benchmark 路径。
- 性能记录覆盖 paged v1 的潜在瓶颈：block table 间接索引、K/V 访存、launch overhead、occupancy。

## 实施记录

### Day 1：复盘 Week 5 语义

- 复习 `docs/design_paged_kv.md`。
- 手画 logical token index 到 `(physical_block, block_offset)` 的映射。
- 明确 paged reference 是 Week 6 kernel 的 correctness anchor。

### Day 2：阅读 dense decode kernel

- 对照 `flashdec/kernels/dense_decode.py` 理解：
  - 每个 program 处理一个 `(sequence, q_head)`。
  - online softmax 的 `m_i`、`l_i`、`acc`。
  - `seq_lens` 如何控制有效 token 范围。

### Day 3-4：实现 paged decode v1

- 把 dense kernel 中的 dense token address 改成 block table address：

```text
logical_block = token_idx // block_size
block_offset = token_idx % block_size
physical_block = block_tables[seq_idx, logical_block]
```

- 先限定：
  - `block_size = 16`
  - `head_dim = 64`
  - FP16
- 保留 MHA/GQA/MQA head 映射：

```python
kv_head = q_head // (num_q_heads // num_kv_heads)
```

### Day 5：correctness tests

- 对齐 `paged_decode_attention_ref`。
- 覆盖：
  - 变长 sequence。
  - 非连续 physical blocks。
  - `seq_len == 0`。
  - GQA head mapping。
  - 自定义 `sm_scale`。
  - v1 不支持 shape 的报错。

### Day 6：benchmark 脚本

- 新增 paged decode benchmark。
- 记录 shape、dtype、block size、used blocks、p50/p90/mean latency。
- 上板后生成第一版 CSV。

### Day 7：复盘

- 记录 correctness 和 benchmark 结果。
- 判断 v1 的主要瓶颈来自访存、索引、launch overhead 还是 occupancy。
- 给 Week 7 的 head_dim 128、BF16 和更完整 shape matrix 做准备。

## 当前已完成

- 新增 Triton paged decode kernel v1：
  - `flashdec.kernels.paged_decode.paged_decode_attention`
- 新增 CUDA-only correctness tests：
  - `tests/test_paged_decode.py`
- 新增 benchmark 脚本：
  - `benchmarks/run_paged_decode.py`
- 更新 kernel lazy export：
  - `flashdec.kernels.paged_decode_attention`
- 更新 benchmark 文档：
  - `benchmarks/README.md`

## Kernel 实现范围

当前 v1 支持：

- physical K/V cache layout：

```text
[num_blocks, num_kv_heads, block_size, head_dim]
```

- block table layout：

```text
[num_seqs, max_blocks_per_seq]
```

- 每个 Triton program 处理一个 `(sequence, q_head)`。
- 沿 logical block 遍历 K/V。
- 每个 logical block 通过 `block_tables` 查 physical block。
- 对最后一个 block 使用 `seq_lens` mask。
- 使用 FP32 accumulation 和 online softmax。
- `seq_len == 0` 时输出 zero。
- 支持 MHA/GQA/MQA 的 q head 到 kv head 映射。

v1 限制：

- 仅支持 FP16。
- 仅支持 `head_dim = 64`。
- 仅支持 `block_size = 16`。
- 尚未做 kernel 参数 sweep 和 profiling。

## 当前环境限制

当前 Codex macOS 环境没有 PyTorch / pytest / CUDA / Triton，因此不能在本机直接运行 Triton correctness 和 benchmark。

可在本机完成：

```bash
python3 -m compileall flashdec tests benchmarks
```

结果：编译通过。

## RTX 5070 验证记录（2026-06-28）

运行环境：

- OS：Linux / WSL2。
- Python：3.12.3。
- pytest：9.1.1。
- 测试路径：`<repo>`。

运行命令：

```bash
pytest -vv tests/test_paged_decode.py
```

结果：

```text
collected 6 items

tests/test_paged_decode.py::test_paged_decode_attention_matches_reference_variable_lengths PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_gqa_mapping PASSED
tests/test_paged_decode.py::test_paged_decode_attention_zero_seq_len_outputs_zero PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_custom_scale PASSED
tests/test_paged_decode.py::test_paged_decode_attention_rejects_unsupported_head_dim PASSED
tests/test_paged_decode.py::test_paged_decode_attention_rejects_unsupported_block_size PASSED

6 passed in 3.79s
```

覆盖结论：

- paged decode Triton kernel v1 与 `paged_decode_attention_ref` 对齐。
- 变长 sequence、非连续 physical block、`seq_len == 0`、GQA head mapping 和自定义 `sm_scale` 均通过。
- v1 不支持的 `head_dim=128` 和 `block_size=8` 会按预期报错。

## 需要在 RTX 5070 开发板完成

联合回归 Week 5 + Week 6：

```bash
pytest -vv tests/test_paged_cache.py tests/test_paged_decode.py
```

## RTX 5070 Benchmark 记录（2026-06-28）

运行环境：

- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。
- dtype：float16。
- block size：16。
- num warps：4。
- repeats：30。

运行命令：

```bash
python benchmarks/run_paged_decode.py --output benchmarks/results/week6_paged_decode.csv
```

输出文件：

```text
benchmarks/results/week6_paged_decode.csv
```

结果摘要：

| shape `(num_seqs,q_heads,kv_heads,head_dim,max_seq_len)` | actual_seq_len | used_blocks | ref mean_ms | ref p50_ms | triton mean_ms | triton p50_ms | triton p90_ms | speedup_vs_ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1,8,8,64,128)` | 95-95 | 6 | 1.146363 | 1.114496 | 0.029984 | 0.017952 | 0.049056 | 38.2325 |
| `(4,8,8,64,512)` | 301-486 | 96 | 4.598040 | 4.519072 | 0.037918 | 0.034816 | 0.036736 | 121.2631 |
| `(8,16,4,64,512)` | 256-476 | 191 | 18.400075 | 15.830560 | 0.054101 | 0.044960 | 0.084640 | 340.1039 |
| `(16,16,4,64,1024)` | 562-1018 | 762 | 33.337782 | 31.871328 | 0.178548 | 0.177024 | 0.178560 | 186.7158 |

观察：

- paged decode Triton v1 在默认 shape 上均明显快于 paged PyTorch reference，mean speedup 为 38.2325x 到 340.1039x。
- 最大 shape 上 Triton p50 为 0.177024 ms，p90 为 0.178560 ms，默认 benchmark 下尾部比较稳定。
- 第 3 组 reference 出现较大 max latency：73.098045 ms，因此后续比较时应继续优先看 p50/p90，并保留 mean/max 观察抖动。
- 当前 speedup 只表示相对自写 paged PyTorch reference 的加速，不代表已经达到 FlashInfer/vLLM 等成熟实现水平。
- Week 7 需要补 head_dim 128、BF16、更多 batch/context sweep，并开始判断主要瓶颈来自 K/V 访存、block table 间接索引还是 occupancy。

## Week 6 完成判定

- paged decode Triton kernel v1 代码已完成。
- correctness tests 已在 RTX 5070 上通过：`6 passed in 3.79s`。
- benchmark 已在 RTX 5070 上完成，结果文件为 `benchmarks/results/week6_paged_decode.csv`。
- 本地静态编译通过。
