# Paged Decode Kernel Experiments

本文记录 FlashDec paged-decode kernel 的受控实验：每个实验只改变一个主要因素，并把假设、计时范围、结果和默认配置决策绑定到正式摘要。完整系统结果见[性能报告](performance_report.md)。

## 测量约定

- CUDA event 只覆盖被比较的 kernel 路径；输入生成、JIT、reference validation 和 layout conversion 不进入计时。
- 同一组对照共享 shape、dtype、seed 和逻辑输入。
- 主要指标是绝对 p50/p90 latency；speedup 只在计时边界相同时使用。
- `effective_*_gbps` 按逻辑 Q/K/V/output 字节估算，不是硬件 DRAM counter。
- 小于 0.1 ms 的 case 对 host 调度和系统噪声敏感，不能单独决定默认配置。
- 负结果与异常值保留，不从矩阵中删除。

## 实验 1：`num_warps`

问题：更多 warps 会提高并行度，还是会因同步、调度与 register pressure 让当前 kernel 变慢？

```bash
python benchmarks/run_paged_decode_warp_sweep.py \
  --dtype both \
  --block-size 16 \
  --kv-layout token_major \
  --num-warps 2 4 8 \
  --warmup 5 \
  --repeat 30 \
  --seed 87 \
  --output benchmarks/results/paged_decode_warp_sweep.csv
```

矩阵覆盖 FP16/BF16、batch `1–128`、context `128–8192`、`num_warps=2/4/8`，固定 `head_dim=128`、32 query heads、8 KV heads。28 个 dtype/case 组合中，2 warps 的 p50 全部最优；相对 4 warps 的平均 p50 speedup 为 `2.10x`，相对 8 warps 为 `3.75x`。

结论：通用默认值采用 `num_warps=2`。该结果只适用于当前 program mapping，不能解释为 Triton kernel 的普遍规律。逐 case 数据、环境和历史 provenance 见[warp-selection summary](../benchmarks/results/paged_decode_warp_selection_summary.md)。

## 实验 2：Paged block size

问题：更大的 block 会减少 block-table 间接索引，但也会扩大尾块无效位置和编译期张量形状，哪一项在当前 shape 上占主导？

```bash
python benchmarks/run_block_size_sweep.py \
  --output benchmarks/results/paged_decode_block_size.csv
```

实验比较 block size `8/16/32`，固定 2 warps。84 行 full sweep 全部通过 reference validation：block 32 赢得 24/28 个 p50、25/28 个 p90 和 26/28 个 mean case；相对 block 16 的 p50 几何平均为 `1.31x`，相对 block 8 为 `1.95x`。少数 FP16 小 shape 更适合 block 16，但不足以支持额外的默认 dispatch。

结论：通用默认值采用 block size 32；调用方仍可为明确的小 shape 显式选择其他 block size。完整表格见[block-size summary](../benchmarks/results/paged_decode_block_size_summary.md)。

## 实验 3：KV physical layout

问题：让同一 token 的 head dimension 连续（token-major），还是让同一 dimension 的 token 位置连续（dim-major），更适合当前 paged-decode 访问模式？

```bash
python benchmarks/run_layout_sweep.py \
  --output benchmarks/results/paged_decode_kv_layout.csv
```

两种 layout 使用相同逻辑 K/V、block table 和 attention 语义，layout conversion 在计时前完成。56 行 full matrix 中，token-major 赢得 25/28 个 p50 和 25/28 个 p90 case；`dim-major/token-major` 的 p50 几何平均为 `1.314x`。context sweep 的 12 个 p50 case 全部由 token-major 获胜。

结论：只保留 token-major `[page, kv_head, token, dim]` 作为 runtime 默认 layout。完整表格见[layout summary](../benchmarks/results/paged_decode_kv_layout_summary.md)。

## 实验 4：Triton staging

问题：在 layout、block size 和 warps 固定后，显式软件流水能否带来足够稳定的收益？

```bash
python benchmarks/run_num_stages_sweep.py \
  --output benchmarks/results/paged_decode_staging.csv
```

比较 implicit default 与 `num_stages=1/2/3/4`。预先采用 5% p50 几何平均门槛，并要求主要 shape 无超过 5% 回退、FP16/BF16 方向一致。最佳候选 stage 2 只有 `1.0039x`，且不同 dtype 的方向不完全一致。

结论：保留 `num_stages=None`。这是正式负结果；没有继续引入 shape/dtype staging dispatch。完整表格见[staging summary](../benchmarks/results/paged_decode_staging_summary.md)。

## 实验 5：访存与 profiling 解释

随着 context 增长，K/V 逻辑读取量近似线性增加。长 context case 的估算有效带宽稳定在约 `1.1–1.3 TB/s`，而小 batch/短 context 更受 launch 与固定开销影响。PyTorch profiler、Chrome trace 和 CUDA-event 结果共同支持“当前 kernel 越来越接近 K/V 访存主导”的解释，但没有 Nsight 硬件 counter，因此不把估算带宽称为实测 DRAM bandwidth。

最终默认配置的绝对 latency 与 profiler 输出见[default profile](../benchmarks/results/paged_decode_default_profile_summary.md)。

## 配置结论

| 维度 | 默认值 | 证据边界 |
| --- | --- | --- |
| KV layout | token-major | 25/28 p50 case 胜出；只比较两种已实现 layout |
| block size | 32 | 24/28 p50 case 胜出；少数 FP16 小 shape 例外 |
| `num_warps` | 2 | 28/28 p50 case 胜出；绑定当前 kernel mapping |
| `num_stages` | implicit (`None`) | 所有显式候选均未达到预设 5% 门槛 |

这些配置不是独立 speedup 的乘积，也不能跨硬件直接继承。修改默认值需要重新运行 correctness、完整 sweep 和相应正式摘要校验。
