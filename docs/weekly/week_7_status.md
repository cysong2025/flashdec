# Week 7 状态记录

## 本周主题

真实 decode shape 补全。

## 本周学习目标

- 理解真实 LLM decode 中常见的 `head_dim=128`。
- 理解 FP16 和 BF16 的数值范围差异，以及 BF16 为什么常用于大模型推理。
- 理解 GQA/MQA 如何减少 KV cache 体积。
- 学会设计 batch/context shape matrix，而不是只测 toy shape。
- 能根据 batch、context、head_dim、dtype 解释 latency 变化。

## 本周计划

### Day 1：扩展 kernel 支持范围

- 将 paged decode kernel 从 `head_dim=64` 扩展到 `head_dim=64/128`。
- 将 dtype 从 FP16 扩展到 FP16/BF16。
- 保持 `block_size=16` 不变，避免同时扩大过多变量。

### Day 2：补 correctness tests

- 参数化覆盖：
  - `head_dim=64/128`。
  - FP16/BF16。
  - MHA/GQA/MQA。
  - variable seq lens。
  - non-contiguous physical block。
  - `seq_len == 0`。

### Day 3-4：做 shape sweep benchmark

- batch sweep：
  - `1, 2, 4, 8, 16, 32, 64, 128`
- context sweep：
  - `128, 256, 512, 1024, 2048, 4096, 8192`
- 默认使用：
  - `num_q_heads=32`
  - `num_kv_heads=8`
  - `head_dim=128`
  - `block_size=16`

### Day 5：整理兼容性文档

- 更新 `docs/compatibility.md`。
- 明确支持和暂不支持的 shape、dtype、layout。

### Day 6-7：复盘

- 记录 Week 7 correctness 和 benchmark。
- 初步判断不同 shape 的瓶颈。
- 为 Week 8 参数、layout、访存优化做准备。

## 当前已完成

- 更新 paged decode Triton kernel：
  - 支持 `head_dim=64/128`。
  - 支持 FP16/BF16。
- 更新 correctness tests：
  - `tests/test_paged_decode.py`
- 新增 Week 7 benchmark 脚本：
  - `benchmarks/run_week7_paged_decode.py`
- 更新 benchmark 文档：
  - `benchmarks/README.md`
- 新增兼容性文档：
  - `docs/compatibility.md`

## 当前实现范围

`paged_decode_attention` 当前支持：

- `block_size=16`
- `head_dim=64/128`
- FP16/BF16
- MHA/GQA/MQA
- variable seq lens
- non-contiguous physical blocks
- `seq_len == 0`

暂不支持：

- `block_size=8/32`
- `head_dim` 之外的 64/128
- FP32 paged Triton kernel
- kernel autotune
- 多种 KV cache physical layout

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
collected 14 items

tests/test_paged_decode.py::test_paged_decode_attention_matches_reference_variable_lengths[64-dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_matches_reference_variable_lengths[64-dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_matches_reference_variable_lengths[128-dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_matches_reference_variable_lengths[128-dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_gqa_mapping[dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_gqa_mapping[dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_mqa_mapping[dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_mqa_mapping[dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_zero_seq_len_outputs_zero[dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_zero_seq_len_outputs_zero[dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_custom_scale[dtype0] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_supports_custom_scale[dtype1] PASSED
tests/test_paged_decode.py::test_paged_decode_attention_rejects_unsupported_head_dim PASSED
tests/test_paged_decode.py::test_paged_decode_attention_rejects_unsupported_block_size PASSED

14 passed in 4.48s
```

覆盖结论：

- `head_dim=64/128` 均已通过 correctness。
- FP16/BF16 均已通过 correctness。
- GQA 和 MQA 映射均已通过 correctness。
- `seq_len == 0`、自定义 `sm_scale` 和不支持 shape 的报错路径均正常。

## 需要在 RTX 5070 开发板完成

Week 7 shape sweep：

```bash
python benchmarks/run_week7_paged_decode.py --output benchmarks/results/week7_paged_decode.csv
```

快速冒烟：

```bash
python benchmarks/run_week7_paged_decode.py --quick --mode triton --output benchmarks/results/week7_paged_decode_quick.csv
```

## RTX 5070 Quick Benchmark 记录（2026-06-28）

运行环境：

- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。
- dtype：float16 / bfloat16。
- head_dim：128。
- `num_q_heads=32`，`num_kv_heads=8`。
- block size：16。
- num warps：4。
- repeats：30。

运行命令：

```bash
python benchmarks/run_week7_paged_decode.py --quick --mode triton --output benchmarks/results/week7_paged_decode_quick.csv
```

输出文件：

```text
benchmarks/results/week7_paged_decode_quick.csv
```

batch quick sweep，固定 `max_seq_len=1024`：

| dtype | batch | actual_seq_len | used_blocks | mean_ms | p50_ms | p90_ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| float16 | 1 | 1000-1000 | 63 | 0.101050 | 0.078272 | 0.125792 |
| float16 | 8 | 520-1019 | 355 | 0.267021 | 0.264704 | 0.279488 |
| float16 | 32 | 516-1007 | 1517 | 0.774960 | 0.773792 | 0.781792 |
| bfloat16 | 1 | 572-572 | 36 | 0.057907 | 0.048672 | 0.092320 |
| bfloat16 | 8 | 543-950 | 346 | 0.304409 | 0.256480 | 0.374112 |
| bfloat16 | 32 | 551-1005 | 1579 | 0.776469 | 0.774048 | 0.775488 |

context quick sweep，固定 `batch=16`：

| dtype | max_seq_len | actual_seq_len | used_blocks | mean_ms | p50_ms | p90_ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| float16 | 128 | 67-123 | 103 | 0.071132 | 0.068352 | 0.074944 |
| float16 | 1024 | 538-995 | 789 | 0.451404 | 0.450976 | 0.460416 |
| float16 | 4096 | 2061-3824 | 2843 | 1.577713 | 1.561408 | 1.620192 |
| bfloat16 | 128 | 66-128 | 102 | 0.065961 | 0.057632 | 0.099456 |
| bfloat16 | 1024 | 548-957 | 734 | 0.405505 | 0.405120 | 0.406592 |
| bfloat16 | 4096 | 2049-3930 | 2887 | 1.659111 | 1.600544 | 1.773216 |

观察：

- quick benchmark 覆盖了 FP16/BF16、`head_dim=128`、GQA 配置 `32 q heads / 8 kv heads`。
- batch 从 1 增加到 32 时，p50 从约 0.078 ms / 0.049 ms 增加到约 0.774 ms，说明总 work 和 program 数增加后 latency 明显上升。
- context 从 128 增加到 4096 时，p50 从约 0.068 ms / 0.058 ms 增加到约 1.56-1.60 ms，符合 decode attention 随 K/V 读取量增长而变慢的预期。
- BF16 与 FP16 的 quick 结果整体同量级，部分小 shape 存在随机 seq_len 差异和尾部波动，完整 sweep 需要统一看更多 shape。

## RTX 5070 Full Benchmark 记录（2026-06-28）

运行环境：

- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128。
- CUDA：12.8。
- dtype：float16 / bfloat16。
- head_dim：128。
- `num_q_heads=32`，`num_kv_heads=8`。
- block size：16。
- num warps：4。
- repeats：30。
- mode：all，包含 paged PyTorch reference 与 Triton kernel。

运行命令：

```bash
python benchmarks/run_week7_paged_decode.py --output benchmarks/results/week7_paged_decode.csv
```

输出文件：

```text
benchmarks/results/week7_paged_decode.csv
```

batch sweep，固定 `max_seq_len=1024`：

| dtype | batch | actual_seq_len | used_blocks | triton p50_ms | triton p90_ms | triton mean_ms | speedup_vs_ref |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 1 | 581-581 | 37 | 0.080000 | 0.130464 | 0.087395 | 65.2440 |
| float16 | 2 | 608-750 | 85 | 0.073760 | 0.115904 | 0.083419 | 96.8117 |
| float16 | 4 | 863-926 | 224 | 0.159392 | 0.166080 | 0.159271 | 125.8175 |
| float16 | 8 | 539-910 | 352 | 0.241344 | 0.243392 | 0.240126 | 143.1503 |
| float16 | 16 | 513-967 | 763 | 0.443040 | 0.453920 | 0.441274 | 142.3299 |
| float16 | 32 | 561-1010 | 1644 | 0.786048 | 0.794880 | 0.788991 | 164.6468 |
| float16 | 64 | 519-1024 | 3189 | 1.474528 | 1.488160 | 1.479383 | 173.0416 |
| float16 | 128 | 513-1022 | 6129 | 2.877760 | 2.924800 | 3.158624 | 169.0484 |
| bfloat16 | 1 | 960-960 | 60 | 0.074144 | 0.115808 | 0.084798 | 43.5779 |
| bfloat16 | 2 | 937-952 | 119 | 0.091904 | 0.135776 | 0.106111 | 70.5114 |
| bfloat16 | 4 | 555-795 | 167 | 0.130720 | 0.149280 | 0.129535 | 134.6661 |
| bfloat16 | 8 | 564-916 | 351 | 0.245056 | 0.248352 | 0.244814 | 129.6497 |
| bfloat16 | 16 | 518-989 | 763 | 0.418176 | 0.422496 | 0.421465 | 156.6867 |
| bfloat16 | 32 | 514-1006 | 1545 | 0.775744 | 0.780896 | 0.775779 | 163.4091 |
| bfloat16 | 64 | 515-1020 | 3115 | 1.475584 | 1.541248 | 1.488649 | 177.8657 |
| bfloat16 | 128 | 514-1013 | 6114 | 2.884512 | 2.893696 | 2.883469 | 177.8489 |

context sweep，固定 `batch=16`：

| dtype | max_seq_len | actual_seq_len | used_blocks | triton p50_ms | triton p90_ms | triton mean_ms | speedup_vs_ref |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 128 | 67-119 | 102 | 0.057632 | 0.087424 | 0.064421 | 989.2193 |
| float16 | 256 | 130-254 | 203 | 0.145632 | 0.163776 | 0.139689 | 446.6203 |
| float16 | 512 | 357-500 | 423 | 0.241760 | 0.243296 | 0.238071 | 275.9466 |
| float16 | 1024 | 513-967 | 763 | 0.443040 | 0.453920 | 0.441274 | 142.3299 |
| float16 | 2048 | 1269-1916 | 1543 | 0.869440 | 0.884320 | 0.851031 | 78.2837 |
| float16 | 4096 | 2123-3746 | 3029 | 1.547840 | 1.563168 | 1.549796 | 41.5249 |
| float16 | 8192 | 4889-8180 | 6536 | 3.304384 | 3.316736 | 3.303728 | 20.1878 |
| bfloat16 | 128 | 65-124 | 101 | 0.059200 | 0.100544 | 0.072750 | 889.7430 |
| bfloat16 | 256 | 148-249 | 211 | 0.134208 | 0.148704 | 0.135898 | 471.9845 |
| bfloat16 | 512 | 279-505 | 395 | 0.241440 | 0.307840 | 0.256333 | 262.7931 |
| bfloat16 | 1024 | 518-989 | 763 | 0.418176 | 0.422496 | 0.421465 | 156.6867 |
| bfloat16 | 2048 | 1035-1933 | 1476 | 0.821664 | 0.828832 | 0.822369 | 77.1214 |
| bfloat16 | 4096 | 2069-3912 | 2894 | 1.604704 | 1.629536 | 1.609332 | 38.6654 |
| bfloat16 | 8192 | 4465-7856 | 5906 | 3.166912 | 3.186240 | 3.169148 | 20.9854 |

观察：

- 完整 sweep 覆盖了 FP16/BF16、batch 1-128、context 128-8192、`head_dim=128`、GQA 配置 `32 q heads / 8 kv heads`。
- batch sweep 中，Triton p50 大致随 batch 增大而上升：FP16 从 0.080000 ms 增至 2.877760 ms，BF16 从 0.074144 ms 增至 2.884512 ms。
- context sweep 中，Triton p50 随上下文长度增长明显上升：FP16 从 0.057632 ms 增至 3.304384 ms，BF16 从 0.059200 ms 增至 3.166912 ms。
- speedup_vs_ref 在短 context 上非常高，最长 context 上降到约 20x，说明随着 K/V 读取量增大，kernel 更接近 memory-bound。
- reference 结果存在明显尾部抖动，因此性能判断优先看 Triton p50/p90 与趋势。
- Week 8 优化应优先围绕长 context 下的 K/V 访存、block table 间接索引开销、`num_warps` 和 cache layout 做实验。

## 上板后要记录

- Week 8 优化优先级。

## Week 7 完成判定

- 代码已支持 `head_dim=64/128` 与 FP16/BF16。
- correctness tests 已在 RTX 5070 上通过：`14 passed in 4.48s`。
- quick Triton benchmark 与完整 shape sweep benchmark 均已在 RTX 5070 上完成。
- 兼容性文档已新增。
