# Benchmarks

这里放 benchmark 脚本和结果。

计划目录：

- `results/`：CSV 和 Markdown benchmark 输出。
- `profiles/`：Nsight / PyTorch profiler 输出。体积太大的 profiler 文件不要直接提交到 Git，可以只保留截图或摘要。

benchmark 记录至少包含：

- 测试日期。
- GPU 型号。
- PyTorch / Triton / CUDA 版本。
- shape。
- warmup 次数。
- 计时迭代次数。
- p50 / p90 / mean latency。
- 简单结论。

当前脚本：

- `run_microbench.py`：Week 1 小算子 benchmark。
- `run_matmul_bench.py`：Week 2 matmul shape sweep，对比 `torch.matmul`、fixed Triton 和 autotuned Triton。
- `profile_matmul.py`：Week 2 PyTorch profiler 文本摘要。
- `run_decode_reference.py`：Week 3 dense decode PyTorch reference baseline。
- `run_dense_decode.py`：Week 4 dense decode Triton kernel benchmark。
- `run_paged_decode.py`：Week 6 paged decode Triton kernel benchmark。
- `run_week7_paged_decode.py`：Week 7 paged decode batch/context/dtype shape sweep。
- `run_week8_paged_decode.py`：Week 8 paged decode `num_warps` sweep，并输出 tokens/s、估算字节数和有效 GB/s。
- `run_block_size_sweep.py`：固定当前 `num_warps` 默认值，对比 paged decode 的 `block_size=8/16/32`。
- `run_layout_sweep.py`：固定 `block_size=32, num_warps=2`，对比 token-major 与 dim-major KV cache layout。
- `profile_paged_decode.py`：Week 9 paged decode PyTorch profiler；支持 FP16/BF16 联合运行、token-major/dim-major 元数据、四类代表场景和可选 Chrome trace。
- `run_num_stages_sweep.py`：Week 10 有边界的 `default/1/2/3/4` staging sweep；固定 layout、block size 和 warps，只覆盖默认决策所需的代表场景。
- `run_rope_kv_append_bench.py`：Week 11 对比 `torch`、独立 `cuda` 与 `fused_cuda` 的 RoPE + paged KV append CUDA-event latency；计时前完成 cache prefill、extension preload 与 correctness 对齐。
- `run_decode_engine_workload.py`：Week 12 动态 DecodeEngine workload；对比完整 step 的 `torch` 与 `fused_cuda` append path，输出 wall-clock p50/p90/p99、tokens/s、active batch、block allocator/reuse 和 backpressure 指标。
- `summarize_decode_engine_trials.py`：严格验证 3-trial CSV 的 36 行矩阵、torch/fused 状态轨迹、block accounting、seed 和交替 backend 顺序，输出 per-trial ratio、跨 trial median/range/geometric mean 与稳定性方向。

当前通用 benchmark/profile 默认配置为 `block_size=32, num_warps=2`。FP16 的少数小 shape 可显式使用 `block_size=16` 对照。

最终默认配置 profiling：

```bash
python benchmarks/profile_paged_decode.py \
  --case all \
  --dtype both \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --repeat 10 \
  --output-dir benchmarks/profiles/week9_final_default \
  --summary-output benchmarks/results/week9_final_default_summary.md
```

`--case all` 覆盖 small、medium、large、large-batch；summary 每行包含完整 shape、dtype、layout、block size、num warps、GPU、PyTorch 和 CUDA 版本。

Week 10 `num_stages` 快速验证：

```bash
python benchmarks/run_num_stages_sweep.py \
  --cases medium \
  --dtype both \
  --num-stages default 1 2 3 4 \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --warmup 3 \
  --repeat 10 \
  --output benchmarks/results/week10_num_stages_quick.csv
```

完整 sweep 使用 medium、large、large-batch，`warmup=5, repeat=30`。默认值只在候选相对 implicit default 的 p50 几何平均稳定提升超过 5%、主要 shape 无超过 5% 回退、FP16/BF16 方向一致时修改；否则保留 `num_stages=None`。

已提交的精简结果摘要：

- `results/week8_block_size_summary.md`：RTX 5070 block-size correctness、quick sweep 和 block-size/warp 交叉实验结论。
- `results/week9_summary.md`：paged decode profiler 与 CUDA event 摘要。
- `results/week9_final_default_summary.md`：token-major、block32、2 warps 的 FP16/BF16 四场景最终 profiling 摘要。
- `results/week10_num_stages_summary.md`：`default/1/2/3/4` staging full sweep、几何平均和最终冻结决策。
- `results/week12_decode_engine_workload_summary.md`：动态 runtime 首轮 full workload 的 complete-step latency、吞吐、allocator/backpressure 指标和多 trial 后续方法。

Week 12 dynamic runtime 快速验证：

```bash
python benchmarks/run_decode_engine_workload.py \
  --quick \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_quick.csv
```

完整运行使用 `--workload all`（默认）、`warmup_steps=5`，并默认比较 `torch` 与 `fused_cuda` append。计时是完整 runtime 的 wall-clock，包含 submit/admit、`Engine.step` 和 finish/cancel；它不能与 Week 11 的 append-only CUDA-event 数字直接比较。

为减少尾延迟噪声和固定执行顺序偏差，正式结论使用：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials3.csv
```

相邻 trial 会反转 append backend 顺序，并使用 `seed + trial_index`；CSV 的 `trial`、`trial_count`、`backend_order` 和 `seed` 可用于严格配对。

三轮 CSV 同步回来后执行：

```bash
python benchmarks/summarize_decode_engine_trials.py \
  --input benchmarks/results/week12_decode_engine_workload_trials3.csv \
  --output benchmarks/results/week12_decode_engine_workload_trials3_summary.md
```

聚合器默认要求 3 workloads x 2 dtypes x 2 backends x 3 trials。缺行、重复行、invariant failure、torch/fused lifecycle/allocator 轨迹不同、seed 不连续或 backend 顺序未交替都会直接失败，不生成可能误导的摘要。
