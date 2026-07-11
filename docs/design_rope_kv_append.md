# RoPE + Paged KV Append 设计说明

## 目标

这一数据路径负责把一个 decode step 的新 Q/K/V 接入 PagedKVCache：先根据每个 request 的当前位置旋转 Q/K，再把 rotated K 和原始 V 写入 physical block，最后返回 paged attention 所需元数据。

当前实现是可读的 PyTorch reference，用来固定 CUDA extension 必须遵循的语义，不作为性能实现。

## 接口

```python
result = flashdec.rope_paged_kv_append_ref(
    cache=cache,
    layer_idx=0,
    request_ids=request_ids,
    q=q,  # [num_requests, num_q_heads, head_dim]
    k=k,  # [num_requests, num_kv_heads, head_dim]
    v=v,  # [num_requests, num_kv_heads, head_dim]
    rotary_dim=head_dim,
    base=10_000.0,
)

out = flashdec.decode(
    result.q,
    cache.k_cache[0],
    cache.v_cache[0],
    result.block_tables,
    result.seq_lens,
)
```

`RopeAppendResult` 包含：

- `q`：rotated Q。
- `positions`：append 前 position，用于 debug 和 CUDA 对齐。
- `block_tables`：本次 request rows 的 logical-to-physical block table。
- `seq_lens`：append 后长度，可直接用于 attention。

## Position 语义

decode token 的 position 必须使用 append 前的长度：

```text
new request: position = 0
active request: position = current seq_len
append completes: new seq_len = position + 1
```

`PagedKVCache.next_positions()` 只读取状态，不创建 request、不分配 block，也不增长 seq_len。finished/cancelled request 会被拒绝。

## RoPE 约定

当前采用 split-half convention。对 rotary prefix：

```text
x = [x_first, x_second]
rotated_first  = x_first * cos - x_second * sin
rotated_second = x_second * cos + x_first * sin
```

频率为：

```text
inv_freq[i] = base ^ (-2i / rotary_dim)
angle = position * inv_freq
```

cos/sin 和旋转使用 FP32 计算，结果转换回输入 dtype。`rotary_dim < head_dim` 时，剩余 tail 不修改。

## Append 顺序与原子性

```text
read pre-append positions
        |
        +-> apply RoPE to Q
        +-> apply RoPE to K
        |
PagedKVCache.append(rotated K, raw V)
        |
return rotated Q + block_tables + post-append seq_lens
```

capacity failure 由 PagedKVCache v2 在写入前统一检查，因此失败时不会创建新 request、分配 block 或增长已有 seq_len。RoPE 可能已经生成临时 tensor，但 cache 状态保持不变。

## 当前支持与限制

支持 FP16/BF16/FP32、partial rotary dim、多 Q/KV heads、变长 request position 和跨 block append。

当前不支持：

- CUDA KV append 或 fused RoPE kernel。
- interleaved adjacent-pair RoPE。
- RoPE scaling、YaRN 或 NTK-aware scaling。
- 多 layer execution。

## 验证

RTX 5070 focused correctness：

```bash
python -m pytest -vv \
  tests/test_rope_append.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py
```

focused 通过后执行：

```bash
python -m pytest -vv
```

当前代码尚未在 RTX 5070 执行，因此不能写入通过数量或 CUDA 性能结论。
