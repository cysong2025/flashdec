# CUDA KV Append 设计说明

## 目标与边界

本实现是 FlashDec 的第一条原生 CUDA 数据路径。它只完成一个可独立验证的动作：把当前 decode step 中每个 request 的一条 K 和 V，写入 allocator 已经决定好的 physical KV block。

它**不**负责 request 调度、block 分配、RoPE 或 attention。`PagedKVCache` 的 Python runtime 仍负责：

1. 检查 request 是否 active，以及本轮是否会耗尽 block pool。
2. 为跨 block 的 token 分配 physical block。
3. 生成每一行的 `physical_block` 与 `block_offset`。
4. 在 CUDA kernel 成功发射后增长 `seq_len` 并生成 metadata。

这样做先固定 allocator 和 native copy 的接口；下一步融合 RoPE 时，不会把状态机错误、RoPE 公式错误和 CUDA 写入错误混在一起排查。

## 接口

低层公开函数：

```python
flashdec.cuda_kv_append(
    k_cache,        # [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache,        # 同上
    block_ids,      # [batch]，CUDA int32/int64
    block_offsets,  # [batch]，CUDA int32/int64
    k,              # [batch, num_kv_heads, head_dim]
    v,              # 同上
)
```

高层 runtime 接口：

```python
block_tables = cache.append_cuda(
    layer_idx=0,
    request_ids=request_ids,
    k=k,
    v=v,
)
```

`append_cuda()` 与已有 `append()` 使用相同的 block allocator、capacity preflight、request lifecycle、`block_tables` 和 `seq_lens` 语义。它只替换 K/V 写入这一段；当前支持 contiguous CUDA FP16、BF16、FP32 tensor。

## 映射与访存

cache 固定为 token-major layout：

```text
[physical_block, kv_head, block_offset, head_dim]
```

一个 CUDA thread 处理一个 `(request, kv_head, head_dim)` 元素，同时写一对 K/V：

```text
request = linear_idx / (num_kv_heads * head_dim)
head    = (linear_idx / head_dim) % num_kv_heads
dim     = linear_idx % head_dim

cache[block_ids[request], head, block_offsets[request], dim] = value[request, head, dim]
```

同一 request/head 的相邻 thread 访问连续的 `head_dim`，因此源 K/V 和目标 cache slot 在该维度上连续。block id/offset 是逐 request 的 metadata；它们由 Python allocator 传入，kernel 不读取或修改 Python request 状态。

## 构建策略

扩展采用 `torch.utils.cpp_extension.load()` JIT 构建：导入 `flashdec` 不会触发编译，第一次调用 `load_cuda_kv_append_extension()` 或 `cuda_kv_append()` 才编译。源码位于：

- `flashdec/csrc/kv_append.cpp`：PyBind 与基本 tensor contract。
- `flashdec/csrc/kv_append_kernel.cu`：CUDA kernel。
- `flashdec/_cuda_kv_append.py`：内部 lazy loader、Python shape/dtype/device/bounds checks；下划线避免与公开函数 `flashdec.cuda_kv_append()` 发生模块名冲突。

构建前必须在**同一个 WSL shell 和虚拟环境**中设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
```

`MAX_JOBS=1` 只是在首次 JIT 编译时降低 WSL 的并行编译内存压力；它不是 kernel 的 GPU launch 参数。

## 正确性与状态语义

测试分成三层：

1. raw op：指定非连续 physical block 和 offset，逐 slot 对照输入 K/V。
2. runtime integration：相同 token 序列分别走 `append()` 与 `append_cuda()`，比较 cache 内容、block table、request state、metrics 和 invariants。
3. error path：CPU input、non-contiguous input 与 capacity failure；capacity failure 必须不创建 request、不分配 block。

extension 会在 allocator mutation 前完成 JIT build；因此工具链或编译失败不会改变 cache state。capacity 在分配前统一检查。kernel 本身只接收 Python 已验证、由 allocator 生成的合法地址；本阶段没有为 runtime CUDA launch failure 设计 state rollback。

## 当前状态

RTX 5070 已成功完成 JIT build 和完整 correctness。初次运行的 2 项公开 API namespace 失败已通过将内部模块改为 `_cuda_kv_append.py` 修复；修复后 focused 结果为 `34 passed in 3.59s`，完整回归为 `198 passed in 5.13s`。验证覆盖 raw slot 写入（FP16/BF16/FP32）、地址边界、runtime allocator 对齐、capacity atomicity 与既有 paged cache/decode 回归。当前没有 native KV append 的性能结论；下一步才测量 PyTorch assignment、independent CUDA append 与后续 fused RoPE+append 的 launch 数和 latency。
