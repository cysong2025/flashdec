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
/Users/songchuangye/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall flashdec tests benchmarks
```

结果：编译通过。

## RTX 5070 验证记录（2026-06-28）

运行环境：

- OS：Linux / WSL2。
- Python：3.12.3。
- pytest：9.1.1。
- 测试路径：`/home/user/work/flashdec`。

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

## 上板后要记录

- `benchmarks/results/week7_paged_decode.csv` 的结果摘要。
- batch sweep 中 latency 随 batch 的变化。
- context sweep 中 latency 随 context 的变化。
- Week 8 优化优先级。

## Week 7 完成判定

- 代码已支持 `head_dim=64/128` 与 FP16/BF16。
- correctness tests 已在 RTX 5070 上通过：`14 passed in 4.48s`。
- quick Triton benchmark 已在 RTX 5070 上完成，完整 shape sweep 待生成 CSV。
- 兼容性文档已新增。
