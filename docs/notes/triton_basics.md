# Triton 基础笔记

本笔记对应 Week 1 的三个小算子：

- `vector_add`
- `row_softmax`
- `rmsnorm`

目标不是“学完 Triton”，而是先掌握以后写 attention kernel 必须用到的最小概念集合。

## 1. Triton 是什么

Triton 是一种用 Python 写 GPU kernel 的 DSL。它不像 PyTorch 那样直接调用已有算子，而是让我们自己描述一个 GPU program 如何读取数据、计算、写回结果。

在 FlashDec 项目中，Triton 的作用是：

- 快速实现自定义 attention kernel。
- 控制 block size、num warps、访存 layout。
- 用较少代码做 kernel 参数实验和 autotune。

## 2. program id

`tl.program_id(axis=0)` 可以理解成当前 Triton program 在 grid 里的编号。

在 `vector_add` 中：

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

含义是：

- 每个 program 处理一段连续元素。
- `pid = 0` 处理 `[0, BLOCK_SIZE)`。
- `pid = 1` 处理 `[BLOCK_SIZE, 2 * BLOCK_SIZE)`。
- 依此类推。

后面写 decode attention 时，可以让一个 program 处理一个 `(sequence, q_head)`，这就是同一个思想。

## 3. block size

`BLOCK_SIZE` 表示一个 Triton program 一次处理多少元素。

它不是 CUDA 里的 thread block 数量，而是 Triton 层面的 tile 大小。Triton 编译器会把这个 tile 映射到底层线程、warp、寄存器和指令。

在 Week 1 中：

- `vector_add` 的 `BLOCK_SIZE` 默认是 1024。
- `row_softmax` 的 `BLOCK_SIZE` 是 `next_power_of_2(n_cols)`。
- `rmsnorm` 的 `BLOCK_SIZE` 也是 `next_power_of_2(n_cols)`。

后面做 PagedAttention 时，block size 会变得更关键，因为它直接影响：

- 每次读取多少 K/V token。
- mask 开销。
- 寄存器压力。
- 访存合并效果。

## 4. mask load/store

真实 shape 往往不是 block size 的整数倍，所以最后一个 block 可能越界。

在 `vector_add` 中：

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

mask 的作用：

- `tl.load` 只读取合法位置。
- `tl.store` 只写回合法位置。
- 避免越界访问。

在 PagedAttention 中，mask 会出现在：

- 最后一个 KV block token 数不满时。
- context length 小于 block 范围时。
- 某些 batch 内序列长度不同的时候。

## 5. stride

stride 表示张量某个维度前进一步，在底层内存地址上要跳多少个元素。

在 `row_softmax` 中：

```python
row_idx = tl.program_id(axis=0)
tl.load(x_ptr + row_idx * stride_x_row + offsets)
```

含义是：

- 每个 program 处理一行。
- `row_idx * stride_x_row` 找到这一行的起始地址。
- `offsets` 在这一行内连续读取。

在 attention 中，stride 会更复杂，因为 K/V cache 可能是多维 layout：

```text
[num_blocks, num_kv_heads, block_size, head_dim]
```

或者：

```text
[num_blocks, num_kv_heads, head_dim, block_size]
```

不同 layout 会改变 K/V load 是否连续，直接影响性能。

## 6. coalesced memory access

合并访存指相邻线程读取相邻内存，从而让 GPU 更高效地发起内存事务。

Week 1 的三个 kernel 都尽量按连续地址读取：

- vector add：连续读取 `x[offsets]` 和 `y[offsets]`。
- softmax：一行内连续读取 `x[row, offsets]`。
- RMSNorm：一行内连续读取 `x[row, offsets]` 和 `weight[offsets]`。

后面 PagedAttention 中，最容易出性能问题的地方是：

- block table 间接索引导致 K/V 物理地址不连续。
- GQA/MQA 的 head 映射导致不同 q head 复用 kv head。
- K 和 V 的 layout 对 load pattern 不友好。

## 7. reduction

reduction 是把一组元素归约成一个值，例如 sum/max。

`row_softmax` 使用：

```python
tl.max(row, axis=0)
tl.sum(numerator, axis=0)
```

`rmsnorm` 使用：

```python
variance = tl.sum(x * x, axis=0) / n_cols
```

后面 decode attention 中会频繁用到 reduction：

- 对 QK 分数求 max。
- 对 `exp(score - max)` 求 sum。
- 对 weighted value 求和。

## 8. Week 1 三个算子的意义

### vector add

练习：

- program id。
- offsets。
- mask。
- 最简单的 load/store。

### row-wise softmax

练习：

- 一行一个 program。
- max reduction。
- sum reduction。
- 数值稳定 softmax。

### RMSNorm

练习：

- 一行一个 program。
- 平方和 reduction。
- 读取 weight。
- 输出保持原 dtype。

RMSNorm 也和 LLM 推理很相关，因为很多 Llama 系模型使用 RMSNorm。

## 9. 需要在 RTX 5070 上验证

当前 Codex 工作区没有 CUDA/Triton，所以还不能完成真实 correctness 和 benchmark。

在 5070 开发板上需要运行：

```bash
pytest tests/test_triton_basics.py
python benchmarks/run_microbench.py --op all --dtype float16
```

期望结果：

- 所有 correctness tests 通过。
- 生成 `benchmarks/results/week1_microbench.csv`。
- CSV 中包含 vector add、row_softmax、rmsnorm 三个结果。

如果失败，优先排查：

1. PyTorch / Triton / CUDA 版本是否兼容。
2. RTX 5070 对应的 CUDA compute capability 是否被当前 Triton 支持。
3. `float16` 是否正常；如果 BF16 失败，先不阻塞 Week 1。
4. shape 太大导致编译或资源问题，先降低 `--rows` / `--cols`。

## 10. 本周工程总结

1. Triton vector add、row-wise softmax 和 RMSNorm 覆盖了自定义 GPU kernel 的基本编程模型。
2. 每个 kernel 都配有 PyTorch reference 和 correctness tests，保证结果可验证。
3. CUDA event benchmark helper 为后续 attention kernel 提供统一的计时和 CSV 输出方式。
