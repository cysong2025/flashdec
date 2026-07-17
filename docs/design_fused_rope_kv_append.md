# Fused RoPE + Paged KV Append 设计说明

## 目标

`fused_cuda` 将一个 decode step 的三项工作放到一次 CUDA launch 中：

1. 计算 rotated Q 并返回。
2. 计算 rotated K，并写入 allocator 指定的 physical KV block。
3. 将不旋转的 V 写入同一 physical slot。

它建立在已验证的 `PagedKVCache` allocator 上：Python 仍负责 request lifecycle、capacity preflight、block id/offset 和 `seq_len`；kernel 不管理 request 状态。

## 三条路径

```text
torch       : PyTorch RoPE(Q/K) + Python cache.append
cuda        : PyTorch RoPE(Q/K) + independent CUDA K/V append
fused_cuda  : one CUDA kernel -> rotated Q + rotated K cache write + raw V cache write
```

`rope_paged_kv_append_ref()` 永远固定第一条路径，是语义基线。正式接口是：

```python
result = flashdec.rope_paged_kv_append(
    cache, layer_idx, request_ids, q, k, v,
    append_backend="fused_cuda",
)
```

## Kernel 映射

输入 Q 为 `[batch, num_q_heads, head_dim]`；K/V 为 `[batch, num_kv_heads, head_dim]`；cache 为 token-major：

```text
[physical_block, kv_head, block_offset, head_dim]
```

kernel 的线性线程 id 同时覆盖两个独立范围：

```text
q range : batch * num_q_heads  * head_dim
kv range: batch * num_kv_heads * head_dim
```

落在 q range 的 thread 写一个 rotated Q 元素。落在 kv range 的 thread 读取 `block_ids[request]` 和 `block_offsets[request]`，写一个 rotated K 与一个 raw V。对于 GQA/MQA，Q/KV head 数不同也能独立映射；不要求一个 thread 的 Q 和 KV 元素属于同一 head。

RoPE 采用既有 split-half convention。kernel 在 float 中计算 `sin/cos` 和旋转，再转换回 FP16/BF16/FP32 输出；`rotary_dim < head_dim` 时 tail 直接复制。

## 原子性边界

`PagedKVCache.append_fused_cuda()` 先检查 q/k/v/position/rotary 参数、加载 extension、执行 capacity preflight，之后才分配 block。因此参数、toolchain 或 capacity error 不会创建 request 或消费 block。和其他 CUDA path 一样，不为异步 CUDA runtime fault 实现 allocator rollback。

## 当前验证状态

RTX 5070 已完成首次 JIT build 和 correctness：focused `66 passed in 44.35s`，完整回归 `214 passed in 4.52s`。首次 focused 时间包含 fused extension 编译，不能用作性能数字。测试覆盖：

- raw fused op 的 GQA、非连续 physical block、partial rotary 和 FP16/BF16/FP32。
- `fused_cuda` 与 PyTorch reference 的 rotated Q、cache、block table、seq_lens、request state 和 metrics 对齐。
- fused capacity failure atomicity 与 CPU cache error path。

三路径 CUDA-event 实验已完成，并将 JIT build 和 cache prefill 排除在计时外。`fused_cuda` 在 6/6 个 p50 case 胜出，p50 几何平均为 `1.2226x` vs torch；独立 CUDA append 为 `0.9840x`。该结果支持 GPU Engine 显式选择 fused 路径，公开 reference API 继续默认使用 torch。
