# Week 12 DecodeEngine Dynamic Workload Summary

## 实验基线

- 日期：2026-07-12。
- GPU：NVIDIA GeForce RTX 5070。
- PyTorch：2.11.0+cu128；CUDA：12.8。
- frozen decode 配置：Triton、token-major、`block_size=32`、`num_warps=2`。
- append 对比：`torch` 与 `fused_cuda`。
- dtype：FP16、BF16。
- workload：`short_churn`、`mixed_steady`、`long_pressure`。
- seed：431；warmup：5 steps。
- 计时范围：request submit/admit、allocator、RoPE/KV append、paged decode、finish/cancel 与 CUDA synchronization；排除 Q/K/V 生成、prompt prefill、warmup 和 JIT build。

原始 quick/full CSV 保留在本地 `benchmarks/results/`，默认不提交。该摘要记录 full CSV 的 12 行结果。

## 正确性与状态一致性

- quick 与 full 均为 12 个唯一配置：3 workloads x 2 dtypes x 2 append backends。
- 12/12 full rows 均为 `validated_invariants=True`。
- 每行均满足 `final_used_blocks + final_free_blocks == max_blocks`。
- 同一 workload/dtype 下，torch 与 fused 的 successful steps、completed tokens、request lifecycle、backpressure 和 allocator 指标完全一致，因此性能比较没有混入不同的调度轨迹。

## 完整 workload 性能

所有 speedup 均为 `torch / fused_cuda`；大于 1 表示 fused 更好。TPS ratio 为 `fused / torch`。

| dtype | workload | torch p50 ms | fused p50 ms | p50 speedup | p90 speedup | p99 speedup | TPS ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | short_churn | 1.170799 | 1.180663 | 0.9916x | 0.9562x | 4.2817x | 1.0994x |
| FP16 | mixed_steady | 1.658845 | 1.463159 | 1.1337x | 1.2511x | 1.3054x | 1.1736x |
| FP16 | long_pressure | 1.533528 | 1.477008 | 1.0383x | 1.0444x | 1.6917x | 1.2317x |
| BF16 | short_churn | 1.150724 | 1.174137 | 0.9801x | 0.9698x | 0.4477x | 0.9204x |
| BF16 | mixed_steady | 1.616732 | 1.479356 | 1.0929x | 1.0443x | 0.4760x | 1.0007x |
| BF16 | long_pressure | 1.583778 | 1.446398 | 1.0950x | 1.1138x | 0.3985x | 1.0105x |

六组 full pair 的几何平均：

| metric | fused vs torch |
| --- | ---: |
| p50 speedup | 1.0537x |
| p90 speedup | 1.0588x |
| p99 speedup | 0.9641x |
| mean latency speedup | 1.0674x |
| tokens/s ratio | 1.0674x |

FP16 与 BF16 的 p50 几何平均分别为 1.0529x 和 1.0546x，方向一致。完整 Engine 的 p50 latency 平均降低约 5.1%，明显小于 Week 11 append-only 的约 18.2% latency 降低。这表明 append fusion 的收益只有一部分能穿过 paged attention、Python lifecycle/allocator 和同步开销传递到端到端 step；按 latency reduction 估算，传递比例约为 28%。

## Runtime 与 Paged KV 行为

状态指标取 FP16 torch 行；同一 workload 的 dtype/backend 行一致。

| workload | successful / measured steps | completed tokens | mean active | finished | cancelled | backpressure | reused / allocations | final used/free | final fragmentation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short_churn | 120 / 120 | 959 | 7.992 | 221 | 24 | 0 | 250 / 258 (96.9%) | 4 / 12 | 89 / 128 tokens (69.5%) |
| mixed_steady | 160 / 160 | 2554 | 15.963 | 76 | 0 | 0 | 201 / 246 (81.7%) | 31 / 65 | 220 / 992 tokens (22.2%) |
| long_pressure | 80 / 112 | 1280 | 16.000 | 0 | 32 | 32 | 32 / 48 (66.7%) | 16 / 0 | 176 / 512 tokens (34.4%) |

结论：

- `short_churn` 完成 221 个请求并取消 24 个请求，96.9% 的 allocation 使用已释放 block，证明高 churn 下没有 block leak。
- `mixed_steady` 保持接近 16 的 active batch，同时完成 76 个不同 context 请求；最终 block utilization 为 32.3%，fragmentation 为 22.2%。
- `long_pressure` 在所有请求跨 block boundary 时产生 32 个明确 backpressure step，取消 32 个请求后复用 32 个 block；最终 pool 为 16 used / 0 free，invariant 仍成立。

## 需要保留的负结果

`short_churn` 的 full p50 在 FP16/BF16 都没有从 fused 获益（0.9916x/0.9801x）。这说明短请求中 Python 调度、allocator、paged decode 和同步的固定成本足以覆盖 append fusion 的收益，不能仅凭 append-only microbenchmark 决定完整 runtime 性能。

尾延迟目前也不能冻结结论：六组 p99 的几何平均为 0.9641x，BF16 的三个 fused p99 均出现明显离群值；同时 FP16 torch 也有 7-10 ms 级离群值。quick 与 full 的 FP16 short-churn p50 speedup 从 1.1448x 变化到 0.9916x，进一步说明短 workload 对单次运行噪声敏感。

## 下一步实验方法

benchmark 已增加 `--trials`，相邻 trial 会反转 `torch`/`fused_cuda` 的执行顺序，并对 seed 递增。下一轮使用 3 trials，按 workload/dtype/trial 成对计算 speedup，再报告跨 trial 的中位数或几何平均：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials3.csv
```

只有当 p50/p90 跨 trial 方向稳定时，才把 `fused_cuda` 固定为端到端默认结论；p99 需要单独报告范围和离群值，不能由一次 full run 推断。
