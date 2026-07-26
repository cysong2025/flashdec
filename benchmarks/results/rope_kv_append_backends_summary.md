# RoPE + Paged KV Append Backend Comparison

## 目标

比较已通过 correctness 的三条 GPU 数据路径：

```text
torch       : PyTorch RoPE + Python cache.append
cuda        : PyTorch RoPE + independent CUDA KV append
fused_cuda  : one CUDA kernel -> rotated Q + rotated K write + raw V write
```

计时使用 CUDA event，只包含预填充 context 后的一次 append GPU work；不包含 extension JIT build、cache prefill 或 Python allocator 的 CPU wall-clock 开销。每个 case 在计时前完成 `torch`/`cuda`/`fused_cuda` reference 对齐检查。

## 环境与命令

- GPU：NVIDIA GeForce RTX 5070
- PyTorch：2.11.0+cu128
- CUDA：12.8
- 配置：token-major、`block_size=32`、warmup 20、repeat 100
- commit：`bf01042`（benchmark 脚本）

```bash
python benchmarks/run_rope_kv_append_bench.py \
  --dtype both \
  --output benchmarks/results/week11_rope_kv_append.csv
```

## Full p50 结果

| dtype | case | torch ms | cuda ms | fused ms | cuda speedup | fused speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | default | 1.544416 | 1.761696 | 1.436736 | 0.8767x | 1.0749x |
| FP16 | large-batch | 4.529056 | 4.007392 | 3.419392 | 1.1302x | 1.3245x |
| FP16 | long-context | 1.596608 | 1.882432 | 1.465536 | 0.8482x | 1.0894x |
| BF16 | default | 1.706944 | 1.840032 | 1.447424 | 0.9277x | 1.1793x |
| BF16 | large-batch | 3.538240 | 2.852576 | 2.448480 | 1.2404x | 1.4451x |
| BF16 | long-context | 1.359744 | 1.448000 | 1.076192 | 0.9390x | 1.2635x |

六个 p50 的几何平均：

```text
cuda        : 0.9840x vs torch
fused_cuda  : 1.2226x vs torch
```

`fused_cuda` 等价于约 18.2% 的 p50 latency 降低，并在 6/6 个 full case 胜出。独立 `cuda` append 只在 large-batch 获益，几何平均略慢于 torch。

## 观察与决策

1. 单独把 K/V 写入改为 CUDA 不足以稳定获益：它保留 PyTorch RoPE，又新增写入 kernel 与 metadata 路径。
2. fused 路径消除了独立 Q/K RoPE 和 K 中间 tensor，因此在所有 p50 case 获益；large-batch 的收益最明显。
3. 部分 mean/max/p90 存在离群值，尤其 torch BF16 default。它们不能由这组 append-only 数据直接解释；后续 DecodeEngine workload 要报告 complete-step p90/p99。
4. 融合收益超过项目 5% 的默认决策门槛。因此 GPU DecodeEngine 将显式选择 `append_backend="fused_cuda"`；公开 `rope_paged_kv_append()` 仍默认 `torch`，以保留 CPU/reference 可用性。

## 下一步

停止对独立 CUDA append 的微调，进入 DecodeEngine v1：动态 request admission、active batch、append -> paged decode、finish/cancel、block reuse 与 backpressure。之后以 synthetic workload 测量完整 step latency、tokens/s、fragmentation 和 reuse。
