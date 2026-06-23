# GPU 访存基础笔记

本笔记对应 Week 2：matmul、访存、autotune、profiling 入门。

目标不是完整学习 CUDA 架构，而是建立后续写 dense decode attention 和 paged decode attention 必须用到的性能直觉。

## 1. 为什么 Week 2 要写 matmul

矩阵乘法是最经典的 compute-bound GPU kernel。它能集中练习：

- tile/block 划分。
- global memory 读取。
- register accumulator。
- `tl.dot`。
- M/N/K shape sweep。
- kernel 参数对性能的影响。

本项目最终目标是 decode attention。attention 里也有 QK 点积、softmax、weighted value 求和。虽然 decode attention 的访存模式和 GEMM 不一样，但 matmul 是理解 GPU kernel 性能的好入口。

## 2. Global Memory

global memory 是 GPU 显存。它容量大，但访问延迟高。

在 Week 2 matmul 中：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

每个 Triton program 会从 global memory 读取一个 A tile 和一个 B tile，计算出 C 的一个 tile。

global memory 性能关键点：

- 尽量让相邻 program 或相邻 lane 访问连续地址。
- 减少重复读取。
- 用合适 tile 大小提高数据复用。

## 3. Shared Memory

shared memory 是 SM 内部的高速片上存储。CUDA C++ matmul 常显式把 A/B tile 搬到 shared memory，再让多个线程复用。

在 Triton 中，很多 shared/register 层面的细节由编译器管理。我们仍然要通过 tile 大小、访问模式和 `tl.dot` 写法，让编译器更容易生成高效代码。

本周先不手写 shared memory。重点是理解：

- global memory 慢。
- 片上存储快但容量小。
- tile 的目的就是让读进来的数据尽可能多次参与计算。

## 4. Register

register 是每个线程/程序执行时最快的存储资源，但数量有限。

在 matmul 中，accumulator 通常放在 register 里：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

如果 `BLOCK_M * BLOCK_N` 太大，accumulator 会占用大量 register，可能降低 occupancy 或导致 spilling。tile 越大不一定越快，这是 autotune 有意义的原因之一。

## 5. Coalesced Memory Access

合并访存指多个相邻执行单元访问相邻地址，让 GPU 用更少的 memory transaction 完成读取。

在 Week 1 中：

- vector add 连续读取一维数组。
- softmax 连续读取一行。
- RMSNorm 连续读取一行和 weight。

在 Week 2 matmul 中：

- A tile 读取 `[BLOCK_M, BLOCK_K]`。
- B tile 读取 `[BLOCK_K, BLOCK_N]`。
- 如果 A/B 是 contiguous，K/N 方向的读取更容易形成连续访存。

后续 paged attention 会更难，因为 block table 间接索引会让 K/V cache 的物理地址不再天然连续。

## 6. Compute-Bound 与 Memory-Bound

compute-bound 表示性能主要受计算吞吐限制。典型例子是大矩阵乘法，因为同一批 A/B 数据可以被复用很多次，计算量很大。

memory-bound 表示性能主要受显存带宽和访存延迟限制。典型例子是简单 elementwise kernel 或 decode attention 中读取长 context 的 K/V cache。

判断方式：

- 算术强度高：更可能 compute-bound。
- 每读一个数据只做少量计算：更可能 memory-bound。
- 增大 shape 后延迟基本随读取字节数线性增长：常见 memory-bound 信号。

## 7. 为什么 cuBLAS 很难被手写 matmul 打败

cuBLAS 的 GEMM 已经高度优化：

- 针对不同 GPU 架构有专门 kernel。
- 使用 Tensor Core。
- 对 tile、warp、pipeline、prefetch 做了大量调优。
- 按 shape、dtype、layout 选择不同实现。

Week 2 手写 matmul 的目标不是超过 cuBLAS，而是理解：

- 一个 C tile 如何由 A/B tile 累加得到。
- tile 参数为什么影响性能。
- profiler 中 kernel launch 和 GPU execution 如何区分。
- 为什么成熟库的性能优势来自长期工程调优。

## 8. 为什么 Decode Attention 更偏 Memory-Bound

prefill 阶段通常一次处理较长 prompt，有较大的矩阵乘法和并行度，更容易利用 Tensor Core。

decode 阶段每次只生成一个 token：

```text
q: 当前 token
k/v cache: 历史所有 token
```

它需要不断读取 K/V cache，但每个历史 token 参与的计算量有限。尤其 batch 较小或 context 很长时，性能常常受 K/V 读取带宽、cache layout、block table 间接索引影响。

这也是 FlashDec 后续关注 PagedAttention 和 Paged KV Cache 的原因。

## 9. Week 2 要观察什么

运行 `benchmarks/run_matmul_bench.py` 后，重点观察：

- fixed Triton matmul 和 `torch.matmul` 的差距。
- autotuned Triton 是否比 fixed 配置更快。
- M/N/K 变大时 latency 如何变化。
- p50 与 p90 差距是否明显。
- profiler 里 CPU launch 时间和 CUDA kernel 时间分别是多少。

如果手写 Triton matmul 明显慢于 `torch.matmul`，这是预期结果。更重要的是能解释为什么。
