# FlashDec 兼容性记录

本文记录当前 FlashDec 已支持和暂不支持的 shape / dtype / kernel 范围。结论只基于公开项目代码和个人 RTX 5070 验证结果。

## Paged KV Cache

`flashdec.cache.PagedKVCache` 当前支持：

- 固定大小 physical block。
- 每个 request 维护 logical block list。
- append 单个 decode token。
- 生成 padded `block_tables`。
- 维护 `seq_lens`。
- materialize dense KV cache 以对齐 reference。
- active/finished/cancelled request lifecycle。
- `finish_request()` / `cancel_request()` block release。
- reuse-priority physical block free list。
- block utilization、internal fragmentation 和 allocation/free/reuse metrics。
- capacity preflight 和 allocator invariant validation。
- R2-A multi-layer token transaction：shared location、sequential layer write、commit/abort rollback。
- R3-A Cache ownership core：opaque prefix id、immutable multi-layer full blocks、active refcount、request-private tail 与 inactive LRU。
- R3-B scheduler integration：`RequestSpec.prefix_id`、Cache-derived shared block metadata、global residency + private lifetime commitment、admission attach 与 stale external mutation rejection。
- R3-D hot-path metadata cache：shared block 数只在 submission 时从 resident registry 派生，后续 snapshot/commitment 使用 immutable Engine metadata，并继续与 Cache authoritative request state/version 对齐。

当前限制：

- legacy `append()`、RoPE helper 和 `DecodeEngine.step()` compatibility wrapper 仍限制 `num_layers=1`；`num_layers>1` 使用 `begin_step()` / `step_layer()` / `commit_step()` sequential transaction API。
- finished/cancelled request id 当前不能重新激活。
- runtime v2 已通过 RTX 5070 focused/full correctness 验证。
- R3-B prefix 必须覆盖完整 initial context，并在 request submission 前 resident；尚不包含模型 prefill、content hashing、admission-time prefix eviction、swap/offload 或生产级多线程 serving。

## Paged Decode Triton Kernel

`flashdec.kernels.paged_decode.paged_decode_attention` 当前支持：

| 能力 | 当前状态 |
| --- | --- |
| KV cache layout | token-major 为默认且已验证；dim-major 已通过 RTX 5070 correctness 与 quick benchmark |
| block table layout | `[num_seqs, max_blocks_per_seq]` |
| block size | `8`, `16`, `32`（均已通过 RTX 5070 correctness） |
| head dim | `64`, `128` |
| dtype | `float16`, `bfloat16` |
| MHA | 支持 |
| GQA | 支持 |
| MQA | 支持 |
| benchmark 默认 block size | `32` |
| 默认 `num_warps` | `2` |
| 默认 `num_stages` | `None`，使用 Triton implicit default |
| variable seq lens | 支持 |
| non-contiguous physical blocks | 支持 |
| `seq_len == 0` | 输出 zero |

当前限制：

- `block_size=8/32` 已通过 RTX 5070 correctness；full sweep 后 32 是通用 benchmark 默认值。FP16 的极小 batch/短 context 可单独测试 16。
- 暂不支持 `head_dim` 之外的 64/128。
- 暂不支持 FP32 Triton paged decode kernel。
- 已完成 `num_warps=2/4/8` 手动 sweep，但暂未做自动 autotune。
- block size quick/full sweep 已完成，暂未做 block size autotune。
- dim-major layout `[num_blocks, num_kv_heads, head_dim, block_size]` 已通过 RTX 5070 correctness、quick 和 full benchmark；full p50 几何平均约慢 31%，不是默认 runtime layout，也不做自动 layout dispatch。
- `num_stages=default/1/2/3/4` full sweep 已完成，最佳候选仅约快 0.39%，因此不修改默认 staging。
- 暂未和 FlashInfer / vLLM / TensorRT-LLM 做成熟库性能对比。

## RoPE + Paged KV Append Reference

当前代码支持：

- split-half RoPE。
- `rotary_dim` 为偶数前缀，允许小于 head_dim。
- position 使用 append 前的 request seq_len。
- FP16/BF16/FP32 输入，FP32 计算 cos/sin 和旋转。
- Q 和 K 旋转，V 保持不变。
- rotated K 写入 token-major paged cache，并返回 block tables/seq_lens。
- block 边界分配、capacity failure 原子性和 terminal request 检查。
- `rope_paged_kv_append(..., append_backend="torch" | "cuda")`：默认 `torch` 与 reference 一致；`cuda` 使用 PyTorch RoPE 加 `PagedKVCache.append_cuda()`。

当前限制：

- 独立 CUDA KV append extension 和 `PagedKVCache.append_cuda()` 已在 RTX 5070 通过 JIT build 与 correctness；focused 为 `34 passed in 3.59s`，完整回归为 `198 passed in 5.13s`。
- 统一 RoPE helper 只走 legacy single-layer append；multi-layer Cache 由 DecodeEngine transaction 调用 location-only fused write，不通过该 helper 推进 seq_len。
- 不包含 RoPE scaling、YaRN、NTK-aware scaling 或 interleaved-pair convention。
- RTX 5070 focused 为 `38 passed in 3.60s`，完整回归为 `186 passed in 4.96s`。
- native extension 当前要求 CUDA-resident、contiguous FP16/BF16/FP32 token-major cache 与 K/V；Toolkit 前置检查已通过 `nvcc 12.8.93`、`CUDA_HOME=/usr/local/cuda-12.8` 和 Ninja 1.13.0。
- RoPE 的 `append_backend="cuda"` 集成已通过 RTX 5070 correctness（focused `56 passed in 3.85s`，full `204 passed in 4.47s`）；它不是 fused RoPE kernel，也没有性能结论。
- `append_backend="fused_cuda"` 和低层 `flashdec.fused_rope_kv_append()` 已在 RTX 5070 通过 JIT/correctness（focused `66 passed in 44.35s`，full `214 passed in 4.52s`）；当前支持 token-major contiguous FP16/BF16/FP32。Week 11 append-only full benchmark 的 p50 几何平均为 1.2226x vs torch。

## DecodeEngine v1

`flashdec.DecodeEngine` 当前支持：

- waiting/active/finished/cancelled request lifecycle。
- admission、deterministic active request row order、single-token `step()`。
- `torch`/`cuda`/`fused_cuda` append backend，以及 reference/Triton paged decode backend。
- capacity 不足时返回 `DecodeStepResult(status="backpressure")`，不改变 KV ownership 或 seq_len。
- short-churn、mixed-steady、long-pressure synthetic workload 与完整 step p50/p90/p99/TPS/memory metrics。
- 可选 `profile_ranges=True` 的 preflight/append/decode 归因；默认关闭。

当前限制：DecodeEngine 的 multi-layer reference 与 fused CUDA sequential transaction API、rollback 路径和正式 workload 已在 RTX 5070 验证。它仍不是完整模型执行器：Q/K/V 由调用方提供，不包含 multi-layer prompt prefill、model forward、sampling、prefix 内容构建或网络服务。R1 scheduler 36 行矩阵、R2 144 行矩阵与最终 `337 passed, 25 subtests passed` 回归均已完成；R3-B focused `56 passed, 14 subtests passed` 与完整 `352 passed, 25 subtests passed` 回归已完成。

### Week 11 Native CUDA KV Append

focused 验证：

```bash
python -m pytest -vv \
  tests/test_cuda_kv_append.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py
```

结果：`34 passed in 3.59s`。

完整回归结果：`198 passed in 5.13s`。

覆盖 FP16/BF16/FP32 raw physical slot 写入、block id/offset 边界检查、`append_cuda()` 与 Python allocator/cache 对齐、capacity failure atomicity、non-contiguous 输入拒绝与公开 API lazy import。尚未测量 native append 的性能。

## 已验证 Correctness

### Week 5

`tests/test_paged_cache.py` 在 RTX 5070 上通过：

```text
6 passed in 1.68s
```

覆盖：

- Paged KV Cache append。
- 非连续 physical block。
- paged reference 与 dense reference 对齐。
- FP16 CUDA 路径。
- `seq_len == 0` 和错误输入。

### Week 6

`tests/test_paged_decode.py` 在 RTX 5070 上通过：

```text
6 passed in 3.79s
```

覆盖：

- FP16。
- `head_dim=64`。
- MHA/GQA。
- 非连续 physical block。
- `seq_len == 0`。
- 自定义 `sm_scale`。

### Week 7

`tests/test_paged_decode.py` 在 RTX 5070 上通过：

```text
14 passed in 4.48s
```

覆盖：

- `head_dim=128`。
- FP16/BF16。
- `num_q_heads=32, num_kv_heads=2` 的 GQA。
- `num_q_heads=16, num_kv_heads=1` 的 MQA。
- `seq_len == 0`、自定义 `sm_scale` 和不支持 shape 的报错路径。

### Block Size 扩展

`tests/test_paged_decode.py tests/test_public_api.py` 在 RTX 5070 上通过：

```text
36 passed in 6.17s
```

覆盖 block size 8/16/32、head dim 64/128、FP16/BF16、MHA/GQA/MQA、错误输入和公共 decode API。

### KV Layout 扩展

`tests/test_paged_cache.py tests/test_paged_decode.py tests/test_public_api.py` 在 RTX 5070 上通过：

```text
73 passed in 9.40s
```

覆盖 token-major/dim-major KV cache、两种 layout 的 block-size 推断、variable sequence lengths、non-contiguous physical blocks、MHA/GQA/MQA 与 FP16/BF16。quick benchmark 的 20 条记录和 full benchmark 的 56 条记录均全部 `validated=True`；full 中 token-major 在 25/28 个 p50、25/28 个 p90 比较中更快，因此作为默认 layout。

### Paged KV Runtime v2

focused 验证：

```bash
python -m pytest -vv \
  tests/test_paged_cache.py \
  tests/test_paged_decode.py \
  tests/test_public_api.py
```

结果：

```text
90 passed in 4.47s
```

完整回归：

```bash
python -m pytest -vv
```

结果：

```text
170 passed in 4.66s
```

覆盖 finish/cancel、block release/reuse、终态错误路径、容量失败无 partial request mutation、fragmentation/utilization metrics、active-only metadata、request churn 无 block 泄漏，以及既有 paged decode/kernel/public API 回归。

## Benchmark 路径

Week 6 默认 benchmark：

```bash
python benchmarks/run_paged_decode.py --output benchmarks/results/week6_paged_decode.csv
```

Week 7 shape sweep：

```bash
python benchmarks/run_week7_paged_decode.py --output benchmarks/results/week7_paged_decode.csv
```

快速冒烟 benchmark：

```bash
python benchmarks/run_week7_paged_decode.py --quick --mode triton --output benchmarks/results/week7_paged_decode_quick.csv
```

Week 8 `num_warps` sweep 与有效带宽估算：

```bash
python benchmarks/run_week8_paged_decode.py --quick --output benchmarks/results/week8_paged_decode_warps_quick.csv
python benchmarks/run_week8_paged_decode.py --output benchmarks/results/week8_paged_decode_warps.csv
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
python benchmarks/run_block_size_sweep.py --output benchmarks/results/week8_paged_decode_block_size.csv
python benchmarks/run_layout_sweep.py --quick --output benchmarks/results/week8_paged_decode_layout_quick.csv
python benchmarks/run_layout_sweep.py --output benchmarks/results/week8_paged_decode_layout.csv
```

## 当前工程状态

- kernel 配置已冻结为 token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- PagedKVCache runtime v2、RoPE/KV append、DecodeEngine、R1 Scheduler 与 R2 multi-layer transaction 均已完成 RTX 验证。
- R3 Shared Prefix Blocks 已完成 R3-A Cache core、R3-B scheduler integration 与 R3-C RTX 24 行 FP16/BF16 三轮正式矩阵。当前稳定结论是 physical KV 节省与 bounded-capacity admission 提升；decode latency 尚无稳定收益结论。
- clean-install、版本升级和 release tag 保留在 R3 闭合后的最终发布门禁；公开基线继续作为非阻塞选择性扩展。
