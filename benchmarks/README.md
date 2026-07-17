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
- `profile_decode_engine.py`：在独立 instrumented 模式下标记 request submit/admit、Engine step、preflight、RoPE/KV append、paged decode 和 finish/cancel，输出 CPU/device time、CUDA event count、PyTorch profiler table、可选 Chrome trace 与阶段摘要。
- `run_scheduler_workload.py`：在完全相同的有限 request trace 上比较 cancel-on-backpressure、greedy-step-only 和 lifetime FIFO + aging；输出完成率、强制取消、deadlock、有效 TPS、等待/公平性、commitment/physical block 和完整 step latency。
- `summarize_scheduler_workload.py`：严格验证 36 行 scheduler case/dtype/policy/trial 矩阵、commit/device、seed、轮转执行顺序和策略特有的 progress invariant，再输出跨 trial 中位数摘要。
- `run_multi_layer_engine.py`：R2-D multi-layer transaction workload；覆盖 1/2/4 layers、batch 4/16、context 128/1024、FP16/BF16 和 torch/fused CUDA，分离 complete-token wall/CUDA-event latency、begin/commit host time、独立 profiler attribution、KV bytes 与 rollback probe。
- `summarize_multi_layer_trials.py`：严格验证 multi-layer matrix、case/shape identity、transaction/block accounting、profiler range、rollback evidence、seed 和交替 backend 顺序，再输出跨 trial 稳定性摘要。
- `run_shared_prefix_workload.py`：R3-C 0%/25%/50%/75% shared-prefix workload；分离 bounded-capacity admission 与 fixed-full-batch decode，输出 physical/saved blocks/bytes、attach/registration/eviction latency、complete-step latency 和 TPS。
- `summarize_shared_prefix_trials.py`：严格验证 hit-rate/dtype/trial matrix、case-order 轮转、seed、capacity monotonicity、block/byte accounting、prefix lifecycle、context correctness 与最终 cleanup，再输出跨 trial 中位数摘要。

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

Complete-step profiler 快速验证：

```bash
python benchmarks/profile_decode_engine.py \
  --workload mixed_steady \
  --dtype float16 \
  --append-backends torch fused_cuda \
  --quick \
  --export-trace \
  --output-dir benchmarks/profiles/week12_decode_engine_quick \
  --summary-output benchmarks/results/week12_decode_engine_profile_quick_summary.md
```

profiler 会先使用 disposable Engine 完成 JIT/warmup，再对新的同配置 Engine 记录 ranges。summary 中的 `engine_step` 是包含 preflight/append/decode 的 inclusive range，不能与子 range 相加；instrumented wall-clock 也不能替代 non-instrumented multi-trial CSV。

R1 Scheduler 三策略快速验证：

```bash
python benchmarks/run_scheduler_workload.py \
  --case boundary_deadlock \
  --dtype float16 \
  --trials 1 \
  --output benchmarks/results/r1_scheduler_workload_quick.csv
```

正式矩阵使用：

```bash
python benchmarks/run_scheduler_workload.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r1_scheduler_workload_trials3.csv

python benchmarks/summarize_scheduler_workload.py \
  --input benchmarks/results/r1_scheduler_workload_trials3.csv \
  --output benchmarks/results/r1_scheduler_workload_trials3_summary.md
```

相邻 trial 会轮转策略执行顺序。聚合器严格要求 36 行完整矩阵和统一 commit/device，并验证 boundary case 的策略语义。`completed_tokens` 包含随后被取消请求已经执行的无效工作；`useful_tokens` 只统计最终完成请求，因此策略吞吐结论优先使用 `useful_tokens_per_second`。任何 `resource_deadlocks > 0` 的行都必须与完成率、取消数一起解释，不能只比较 p50。

R2-D multi-layer 快速验证：

```bash
python benchmarks/run_multi_layer_engine.py \
  --case l2_b4_c128 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/r2_multi_layer_engine_quick.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/r2_multi_layer_engine_quick.csv \
  --output benchmarks/results/r2_multi_layer_engine_quick_summary.md \
  --expected-trials 1 \
  --expected-cases l2_b4_c32 \
  --expected-dtypes float16
```

正式矩阵：

```bash
python benchmarks/run_multi_layer_engine.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r2_multi_layer_engine_trials3.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/r2_multi_layer_engine_trials3.csv \
  --output benchmarks/results/r2_multi_layer_engine_trials3_summary.md
```

正式矩阵共 `12 cases x 2 dtypes x 2 backends x 3 trials = 144 rows`。输入生成、context seed、JIT build、profiler probe 和 rollback probe 均排除在正式 complete-token latency 外；profiler 字段只做 append/decode/launch 归因，rollback latency 不混入正常吞吐。Summary 同时报告 ratio 与 torch/fused 绝对 attribution median，任何低于 1 的 ratio 都必须结合绝对时间解释。

commit `fa0f89a` 的 RTX 5070 正式结果已通过 144 行严格校验。fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`；24 个 dtype/case 组合中 20 个三轮 p50 稳定胜出、4 个跨过 1.0。每轮仅 20 repeats，nearest-rank p99 接近单轮最大值，因此必须连同范围报告。

R3-C shared-prefix quick 验证：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --quick \
  --hit-rate all \
  --dtype float16 \
  --trials 1 \
  --output benchmarks/results/r3_shared_prefix_workload_quick.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/r3_shared_prefix_workload_quick.csv \
  --output benchmarks/results/r3_shared_prefix_workload_quick_summary.md \
  --expected-trials 1 \
  --expected-dtypes float16
```

正式矩阵：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r3_shared_prefix_workload_trials3.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/r3_shared_prefix_workload_trials3.csv \
  --output benchmarks/results/r3_shared_prefix_workload_trials3_summary.md
```

正式矩阵共 `4 hit rates x 2 dtypes x 3 trials = 24 rows`。`capacity_admitted_requests` 来自固定 60% bounded pool 的第一次调度；decode latency 使用独立的无共享基线容量，使四档 hit rate 始终运行相同 batch。context 构建、materialization correctness、extension/Triton warmup、registration 和 attach probe 均排除在 complete-step latency 外。共享 prefix 不改变 attention 算法，因此 latency 小幅变化不能单独解释为 kernel 加速。

commit `fd36ed0` 的 RTX 5070 FP16 quick 已通过 4 行严格校验。0%/25%/50%/75% 的 context physical/logical blocks 分别为 `4/4`、`4/4`、`3/4`、`2/4`，bounded-pool admission 分别为 `2/4`、`2/4`、`3/4`、`3/4`。quick 每档只有 3 次正式 step 采样，latency/TPS 非单调，不形成正式性能结论。
