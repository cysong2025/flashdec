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
- `run_fused_transaction_fast_path.py`：R4-A 同 commit checked/trusted transaction A/B；覆盖 2/4 layers、batch 4/16、context 128/1024 与 FP16/BF16，正式 wall 不创建 CUDA event，parity、rollback 与 profiler 独立执行。
- `summarize_fused_transaction_fast_path.py`：严格验证 checked/trusted 完整矩阵、交替顺序、seed、transaction/block/byte trajectory、parity、rollback 和 profiler item/local-scalar 证据，再输出跨 trial ratio 与绝对 attribution。
- `run_persistent_transaction_metadata.py`：R4-B 同 commit `materialized/persistent` controlled ablation；两侧固定使用 R4-A trusted raw math，只切换 Cache transaction metadata 生命周期，输出 parity、rollback、trajectory 与 build/materialization/reuse/release/resident counters。
- `summarize_persistent_transaction_metadata.py`：严格验证 160-row 配对矩阵、全 p50/p90/p99 ranges、CPU-only attribution 和 metadata counter 公式，并报告预注册 keep gate；尚未产生的正式 summary 不属于 release evidence。

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

commit `1d5d8d0` 的 RTX 5070 FP16/BF16 三轮正式矩阵共 24 行并通过严格校验。0%/25%/50%/75% 的 context physical blocks 为 `64/52/36/20`，bounded-pool admission 为 `9/12/15/16`，75% hit 避免占用 `68.8%` context KV capacity 和 `5.5 MiB`。summary 将每个 hit-rate 与同 dtype、同 trial 的 0% baseline 配对，报告 p50/p90/p99/TPS median `[min,max]`，并单独报告 scheduler/Engine p50 attribution。只有全三轮同向才标记 `shared_faster` 或 `shared_slower`。

R3-D hot-path metadata cache 完成 correctness 后，使用 8 trials 做最终 confirmation：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 8 \
  --output benchmarks/results/r3_shared_prefix_workload_trials8.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/r3_shared_prefix_workload_trials8.csv \
  --output benchmarks/results/r3_shared_prefix_workload_trials8_summary.md \
  --expected-trials 8 \
  --expected-dtypes float16 bfloat16
```

commit `fe72e27` 的 RTX confirmation 共 64 行，seed `613-620`，四种 hit-rate 顺序各轮转两次。容量轨迹与 R3-C 一致；所有非零 complete、scheduler 与 Engine p50 paired range 都跨过 1，因此最终性能结论是 near-neutral/no stable direction。旧 3-trial summary 作为优化前负结果基线保留，不能与新 run 直接相除声称 metadata cache 的因果 speedup。

R4-A trusted transaction quick gate：

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case l2_b4_c128 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/r4_fused_transaction_fast_path_quick.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/r4_fused_transaction_fast_path_quick.csv \
  --output benchmarks/results/r4_fused_transaction_fast_path_quick_summary.md \
  --expected-trials 1 \
  --expected-cases l2_b4_c32 \
  --expected-dtypes float16
```

commit `4e18f5d` 的 RTX 5070 quick 保留 provisional FP16 `l2_b4_c32` complete-token p50/TPS `1.7856x/1.8755x`、append CPU `2.3751x` 与 item/local-scalar `20/20 -> 0/0`。随后 formal 在 CPU range `8/8`、同名 CUDA user annotation `7/8` 时严格停止；commit `5d2f9c0` 加入 warmup/retry 后，l4 stress 的三次 probe 又稳定得到首个 decode CPU range 的正 host time与零 correlated device time。两次负结果共同证明 stage-device 关联不是稳定契约；旧 device/event 数字撤回，strict schema 改为 CPU-only。失败的 stress 没有 CSV，不能复用。

CPU-only profiler l4 stress quick：

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case l4_b4_c128 \
  --dtype float16 \
  --trials 3 \
  --quick \
  --output benchmarks/results/r4_fused_transaction_fast_path_l4_stress.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/r4_fused_transaction_fast_path_l4_stress.csv \
  --output benchmarks/results/r4_fused_transaction_fast_path_l4_stress_summary.md \
  --expected-trials 3 \
  --expected-cases l4_b4_c32 \
  --expected-dtypes float16
```

正式矩阵使用五轮以降低 host/stream 抖动对微优化判断的影响：

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case all \
  --dtype both \
  --trials 5 \
  --output benchmarks/results/r4_fused_transaction_fast_path_trials5.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/r4_fused_transaction_fast_path_trials5.csv \
  --output benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md \
  --expected-trials 5
```

正式矩阵共 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`，strict summary 报告 80 个 paired trials。`checked` 与 `trusted` 复用相同 Cache transaction API；runner 只在 benchmark context 中把 Cache 内部 raw launch 切换为 checked 或 trusted，因此状态机和 Engine 路由不变。每个 paired trial 先按轮换顺序完成两条 path 的 non-instrumented synchronized wall，再统一执行 profiler/rollback，避免 attribution 重采集夹在两侧 wall 之间。CPU-only profiler 先在 WARMUP cycle 执行并 abort 一个同形 token，再在唯一 active cycle采集；每个 layer 必须有精确一个 CPU user annotation，其 inclusive CPU total保留 checked 路径 `.item()` 的 host/stream 等待。checked 每个 profiled layer 必须有 5 次 `aten::item`/`aten::_local_scalar_dense`，trusted 为 0。少记 range/scalar 或缺少有效 CPU time可用相同 seed、全新 engine/profiler 重采集，固定最多三次并写入 `profile_attempt_count`；多出 range/scalar 则视为 active-work/fast-path 契约回归并立即终止，不能用重试掩盖。append/decode device time与 CUDA activity不再属于该 strict schema；若未来需要分段 GPU attribution，使用独立 CUDA Event/Nsight probe。summary validator 负责证据完整性，不替代人工性能 gate：overall p50 至少 `1.05x`，且全部 16 个 `dtype x case` 分组的五轮 p50 `[min,max]` 都不穿过 1。该证据只归因于 device-value validation，仍存在的 transaction-view H2D materialization/copy 留给独立 R4-B。

commit `4018449` 的 RTX 5070/CUDA 12.8 正式矩阵通过严格校验：160 rows、80 paired trials，全部 16 个 `dtype x case` 分组均为 `trusted_faster` 且五轮 p50 最小值都大于 1。overall complete-token p50/p90/p99、TPS 和 append CPU/layer ratio 分别为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`；focused `73 passed, 23 subtests passed`，完整回归 `410 passed, 48 subtests passed`。7/16 分组的 p99 range 穿过 1，因此不声明稳定尾延迟收益。R4-A 已冻结，canonical release evidence 见[R4-A 五轮正式摘要](results/r4_fused_transaction_fast_path_trials5_summary.md)；当前进入 R4-B persistent transaction metadata。

R4-B 先执行 RTX focused correctness：

```bash
python -m pytest -q -ra \
  tests/test_paged_cache.py \
  tests/test_multi_layer_transaction.py \
  tests/test_multi_layer_engine.py \
  tests/test_persistent_transaction_metadata_benchmark.py \
  tests/test_persistent_transaction_metadata_summary.py
```

focused 全部通过后执行三轮 FP16 quick：

```bash
python benchmarks/run_persistent_transaction_metadata.py \
  --case l4_b4_c128 \
  --dtype float16 \
  --metadata-paths materialized persistent \
  --trials 3 \
  --quick \
  --seed 811 \
  --output benchmarks/results/r4_persistent_transaction_metadata_quick.csv

python benchmarks/summarize_persistent_transaction_metadata.py \
  --input benchmarks/results/r4_persistent_transaction_metadata_quick.csv \
  --output benchmarks/results/r4_persistent_transaction_metadata_quick_summary.md \
  --expected-trials 3 \
  --expected-cases l4_b4_c32 \
  --expected-dtypes float16
```

只有 quick CSV 与 strict summary 都通过后，才执行五轮正式矩阵：

```bash
python benchmarks/run_persistent_transaction_metadata.py \
  --case all \
  --dtype both \
  --metadata-paths materialized persistent \
  --trials 5 \
  --warmup 3 \
  --repeat 20 \
  --profile-steps 2 \
  --parity-steps 2 \
  --rollback-repeats 2 \
  --seed 811 \
  --output benchmarks/results/r4_persistent_transaction_metadata_trials5.csv

python benchmarks/summarize_persistent_transaction_metadata.py \
  --input benchmarks/results/r4_persistent_transaction_metadata_trials5.csv \
  --output benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md \
  --expected-trials 5 \
  --expected-cases \
    l2_b4_c128 l2_b4_c1024 l2_b16_c128 l2_b16_c1024 \
    l4_b4_c128 l4_b4_c1024 l4_b16_c128 l4_b16_c1024 \
  --expected-dtypes float16 bfloat16
```

这不是把当前实现与旧 commit `4018449` 做逐指令或跨 commit 比较。runner 在同一 commit 内用 benchmark-only private hooks 恢复 legacy materialization boundary；两侧始终执行相同的 R4-A trusted fused CUDA/Triton math。计数口径只覆盖 Cache transaction-view：`materialized` 为每 token `2L+2` 次 materialization、0 次 reuse；`persistent` 为 1 次 materialization、每层 1 次 reuse（共 `L` 次），两侧均 build/release 各 1 次且 terminal resident 为 0。它不等于所有 Engine result tensor clone 的总数。

keep gate 预注册为 overall complete-token p50 `>=1.05x`，且 16/16 个 `dtype x case` 分组的 paired p50 五轮最小值严格大于 1。RTX 数据尚未执行，当前不能声明 R4-B 完成、加速或尾延迟改善。
