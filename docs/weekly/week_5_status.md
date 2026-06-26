# Week 5 状态记录

## 本周主题

Paged KV Cache 数据结构。

## 本周目标

- 把 block table 和 cache 语义定义清楚。
- 实现一个简单可解释的 `PagedKVCache`。
- 实现基于 block table 的 PyTorch paged decode reference。
- 验证 paged KV 输出与 dense KV reference 对齐。

## 当前已完成

- 新增 Paged KV Cache runtime：
  - `flashdec.cache.PagedKVCache`
- 新增 paged decode PyTorch reference：
  - `flashdec.paged_reference.paged_decode_attention_ref`
- 新增 correctness tests：
  - `tests/test_paged_cache.py`
- 新增 paged KV 设计文档：
  - `docs/design_paged_kv.md`
- 更新 package 顶层 lazy exports：
  - `flashdec.PagedKVCache`
  - `flashdec.paged_decode_attention_ref`
  - `flashdec.dense_decode_attention_ref`

## 实现范围

`PagedKVCache` 当前支持：

- 固定大小 physical block。
- 每个 request 维护 logical block list。
- append 一个 token。
- 自动分配新的 physical block。
- 生成 padded block table tensor。
- 维护 `seq_lens`。
- 将 paged KV materialize 成 dense KV，方便和 dense reference 对齐。

物理 cache layout：

```text
[num_layers, max_blocks, num_kv_heads, block_size, head_dim]
```

paged reference 输入 layout：

```text
q:            [num_seqs, num_q_heads, head_dim]
k_cache:      [num_blocks, num_kv_heads, block_size, head_dim]
v_cache:      [num_blocks, num_kv_heads, block_size, head_dim]
block_tables: [num_seqs, max_blocks_per_seq]
seq_lens:     [num_seqs]
```

## Correctness 覆盖

`tests/test_paged_cache.py` 覆盖：

- interleaved request append 后，单个 request 的 physical blocks 可以非连续。
- `block_tables` 与 `seq_lens` 正确生成。
- paged reference 与 dense reference 输出一致。
- `seq_len == 0` 时输出 zero。
- 非空 sequence 的 block table 缺失 physical block 时会报错。
- physical block 容量不足时会报错。

## 当前环境限制

当前 Codex macOS 环境没有 PyTorch / pytest / CUDA / Triton，因此不能在本机直接运行 pytest correctness。

已完成静态验证：

```bash
/Users/songchuangye/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall flashdec tests
```

结果：编译通过。

本机 pytest 阻塞原因：

```text
No module named pytest
No module named torch
```

## 需要在 RTX 5070 开发板完成

在 WSL2 Ubuntu + RTX 5070 环境运行：

```bash
pytest tests/test_paged_cache.py
pytest tests/test_decode_reference.py tests/test_paged_cache.py
```

如果要一起回归前几周：

```bash
pytest tests/test_triton_basics.py tests/test_matmul.py tests/test_decode_reference.py tests/test_dense_decode.py tests/test_paged_cache.py
```

## 上板后要记录

- `tests/test_paged_cache.py` 的通过数量和耗时。
- FP16 CUDA 路径是否与 dense reference 对齐。
- interleaved append 是否生成预期的非连续 block table。
- 若失败，记录具体 shape、dtype、block_size 和误差范围。

## Week 5 完成判定

- `PagedKVCache` 实现完成。
- `paged_decode_attention_ref` 实现完成。
- dense KV 与 paged KV 对齐测试已写好。
- paged KV 设计文档完成。
- 本地语法编译通过。
- RTX 5070 correctness 仍待上板验证，不能写成已通过。
