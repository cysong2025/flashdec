# Benchmarks

这里记录 benchmark 脚本、计时边界和执行命令。Git 跟踪的正式摘要、本地 CSV/log/quick/profile 产物边界与历史结果入口见[结果索引](results/README.md)。

输出目录：

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

脚本：

- `run_microbench.py`：vector add、softmax 和 RMSNorm 小算子 benchmark。
- `run_matmul_bench.py`：matmul shape sweep，对比 `torch.matmul`、fixed Triton 和 autotuned Triton。
- `profile_matmul.py`：matmul PyTorch profiler 文本摘要。
- `run_decode_reference.py`：dense decode PyTorch reference baseline。
- `run_dense_decode.py`：dense decode Triton kernel benchmark。
- `run_paged_decode.py`：paged decode Triton kernel benchmark。
- `run_paged_decode_shape_sweep.py`：paged decode batch/context/dtype shape sweep。
- `run_paged_decode_warp_sweep.py`：paged decode `num_warps` sweep，并输出 tokens/s、估算字节数和有效 GB/s。
- `run_block_size_sweep.py`：固定当前 `num_warps` 默认值，对比 paged decode 的 `block_size=8/16/32`。
- `run_layout_sweep.py`：固定 `block_size=32, num_warps=2`，对比 token-major 与 dim-major KV cache layout。
- `profile_paged_decode.py`：paged decode PyTorch profiler；支持 FP16/BF16 联合运行、token-major/dim-major 元数据、四类代表场景和可选 Chrome trace。
- `run_num_stages_sweep.py`：有边界的 `default/1/2/3/4` staging sweep；固定 layout、block size 和 warps，只覆盖默认决策所需的代表场景。
- `run_rope_kv_append_bench.py`：对比 `torch`、独立 `cuda` 与 `fused_cuda` 的 RoPE + paged KV append CUDA-event latency；计时前完成 cache prefill、extension preload 与 correctness 对齐。
- `run_decode_engine_workload.py`：动态 DecodeEngine workload；对比完整 step 的 `torch` 与 `fused_cuda` append path，输出 wall-clock p50/p90/p99、tokens/s、active batch、block allocator/reuse 和 backpressure 指标。
- `summarize_decode_engine_trials.py`：严格验证 3-trial CSV 的 36 行矩阵、torch/fused 状态轨迹、block accounting、seed 和交替 backend 顺序，输出 per-trial ratio、跨 trial median/range/geometric mean 与稳定性方向。
- `profile_decode_engine.py`：在独立 instrumented 模式下标记 request submit/admit、Engine step、preflight、RoPE/KV append、paged decode 和 finish/cancel，输出 CPU/device time、CUDA event count、PyTorch profiler table、可选 Chrome trace 与阶段摘要。
- `run_scheduler_workload.py`：在完全相同的有限 request trace 上比较 cancel-on-backpressure、greedy-step-only 和 lifetime FIFO + aging；输出完成率、强制取消、deadlock、有效 TPS、等待/公平性、commitment/physical block 和完整 step latency。
- `summarize_scheduler_workload.py`：严格验证 36 行 scheduler case/dtype/policy/trial 矩阵、commit/device、seed、轮转执行顺序和策略特有的 progress invariant，再输出跨 trial 中位数摘要。
- `run_multi_layer_engine.py`：multi-layer transaction workload；覆盖 1/2/4 layers、batch 4/16、context 128/1024、FP16/BF16 和 torch/fused CUDA，分离 complete-token wall/CUDA-event latency、begin/commit host time、独立 profiler attribution、KV bytes 与 rollback probe。
- `summarize_multi_layer_trials.py`：严格验证 multi-layer matrix、case/shape identity、transaction/block accounting、profiler range、rollback evidence、seed 和交替 backend 顺序，再输出跨 trial 稳定性摘要。
- `run_shared_prefix_workload.py`：0%/25%/50%/75% shared-prefix workload；分离 bounded-capacity admission 与 fixed-full-batch decode，输出 physical/saved blocks/bytes、attach/registration/eviction latency、complete-step latency 和 TPS。
- `summarize_shared_prefix_trials.py`：严格验证 hit-rate/dtype/trial matrix、case-order 轮转、seed、capacity monotonicity、block/byte accounting、prefix lifecycle、context correctness 与最终 cleanup，再输出跨 trial 中位数摘要。
- `run_fused_transaction_fast_path.py`：同 commit checked/trusted transaction A/B；覆盖 2/4 layers、batch 4/16、context 128/1024 与 FP16/BF16，non-instrumented wall 不创建 CUDA event，parity、rollback 与 profiler 独立执行。
- `summarize_fused_transaction_fast_path.py`：严格验证 checked/trusted 完整矩阵、交替顺序、seed、transaction/block/byte trajectory、parity、rollback 和 profiler item/local-scalar 证据，再输出跨 trial ratio 与绝对 attribution。
- `run_integrated_scheduled_multi_layer.py`：dynamic mixed-prefix multi-layer trace；组合 arrival、Scheduler、caller-supplied context、transaction rollback、finish/cancel、block reuse 与 terminal cleanup。
- `summarize_integrated_scheduled_multi_layer.py`：重建 dependency-free reference，严格验证 24-row matrix、observed/reference digest、transaction/prefix accounting、reuse 与 zero-used cleanup，再输出绝对 latency/TPS range。
- `run_flashinfer_baseline.py`：固定 `flashinfer-python==0.6.15.post1` 的有限 paged-decode 对比；同一 HND physical KV、page table、Q、scale 和 CUDA-event timing 下运行 FlashDec Triton、FlashInfer FA2 CUDA-core 与 tensor-core 三条预注册路径。
- `summarize_flashinfer_baseline.py`：严格验证 72-row case/dtype/backend/trial matrix、formal `3/10/50` sampling、runner command、版本、128 MiB workspace、clean worktree、布局、normalized tolerance ratio、page-table identity、轮转顺序和计时边界；报告绝对 p50/p90/p99 与 FlashInfer 相对 FlashDec 的 p50/TPS ratio range，logical workload GB/s 明确不是 DRAM bandwidth，不设置事后胜负门。
- `run_vllm_attention_microbench.py` / `summarize_vllm_attention_microbench.py`：在 vLLM 0.25.1 KV/metadata contract 内比较原生 Triton 与 FlashDec Qwen decode attention，使用时间窗口测量、完整 output parity 和冻结性能门槛。
- `run_vllm_model_correctness.py` / `summarize_vllm_model_correctness.py`：固定 Qwen prompts/seed，分别启动原生与 `CUSTOM` backend，验证第一步 greedy top-1 并描述完整 rollout 分叉。
- `run_vllm_model_latency.py` / `summarize_vllm_model_latency.py`：独立进程、交替 backend、默认 Inductor/CUDA Graph 的固定批量 `LLM.generate` A/B。
- `run_vllm_serving_benchmark.py` / `summarize_vllm_serving_benchmark.py`：管理独立 vLLM server 生命周期，运行标准 `vllm bench serve`，严格验证 TPOT、TTFT、throughput、零失败和配对稳定性门槛。

通用 benchmark/profile 默认配置为 `block_size=32, num_warps=2`。FP16 的少数小 shape 可显式使用 `block_size=16` 对照。

默认配置 profiling：

```bash
python benchmarks/profile_paged_decode.py \
  --case all \
  --dtype both \
  --kv-layout token_major \
  --block-size 32 \
  --num-warps 2 \
  --repeat 10 \
  --output-dir benchmarks/profiles/paged_decode_default \
  --summary-output benchmarks/results/paged_decode_default_profile_summary.md
```

`--case all` 覆盖 small、medium、large、large-batch；summary 每行包含完整 shape、dtype、layout、block size、num warps、GPU、PyTorch 和 CUDA 版本。

Paged decode `num_stages` 快速验证：

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
  --output benchmarks/results/paged_decode_staging_quick.csv
```

完整 sweep 使用 medium、large、large-batch，`warmup=5, repeat=30`。默认值只在候选相对 implicit default 的 p50 几何平均稳定提升超过 5%、主要 shape 无超过 5% 回退、FP16/BF16 方向一致时修改；否则保留 `num_stages=None`。

已提交的精简结果摘要：

- `results/paged_decode_block_size_summary.md`：RTX 5070 block-size correctness、quick sweep 和 block-size/warp 交叉实验结论。
- `results/paged_decode_kv_layout_summary.md`：token-major 与 dim-major KV layout 的完整 sweep 和默认布局决策。
- `results/paged_decode_default_profile_summary.md`：token-major、block32、2 warps 的 FP16/BF16 四场景 profiling 摘要。
- `results/paged_decode_staging_summary.md`：`default/1/2/3/4` staging full sweep、几何平均和配置选择。
- `results/rope_kv_append_backends_summary.md`：torch、独立 CUDA 与 fused CUDA append 路径的正式对比。
- `results/decode_engine_workload_trials3_summary.md`：动态 runtime 三轮 complete-step latency、吞吐和稳定性结论。
- `results/decode_engine_stage_profile_summary.md`：instrumented complete-step 阶段归因。

DecodeEngine dynamic runtime 快速验证：

```bash
python benchmarks/run_decode_engine_workload.py \
  --quick \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_quick.csv
```

完整运行使用 `--workload all`（默认）、`warmup_steps=5`，并默认比较 `torch` 与 `fused_cuda` append。计时是完整 runtime 的 wall-clock，包含 submit/admit、`Engine.step` 和 finish/cancel；它不能与 append-only CUDA-event 数字直接比较。

为减少尾延迟噪声和固定执行顺序偏差，正式结论使用：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_trials3.csv
```

相邻 trial 会反转 append backend 顺序，并使用 `seed + trial_index`；CSV 的 `trial`、`trial_count`、`backend_order` 和 `seed` 可用于严格配对。

三轮 CSV 同步回来后执行：

```bash
python benchmarks/summarize_decode_engine_trials.py \
  --input benchmarks/results/decode_engine_workload_trials3.csv \
  --output benchmarks/results/decode_engine_workload_trials3_summary.md
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
  --output-dir benchmarks/profiles/decode_engine_quick \
  --summary-output benchmarks/results/decode_engine_stage_profile_quick_summary.md
```

profiler 会先使用 disposable Engine 完成 JIT/warmup，再对新的同配置 Engine 记录 ranges。summary 中的 `engine_step` 是包含 preflight/append/decode 的 inclusive range，不能与子 range 相加；instrumented wall-clock 也不能替代 non-instrumented multi-trial CSV。

Scheduler 三策略快速验证：

```bash
python benchmarks/run_scheduler_workload.py \
  --case boundary_deadlock \
  --dtype float16 \
  --trials 1 \
  --output benchmarks/results/scheduler_workload_quick.csv
```

正式矩阵使用：

```bash
python benchmarks/run_scheduler_workload.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/scheduler_workload_trials3.csv

python benchmarks/summarize_scheduler_workload.py \
  --input benchmarks/results/scheduler_workload_trials3.csv \
  --output benchmarks/results/scheduler_capacity_progress_summary.md
```

相邻 trial 会轮转策略执行顺序。聚合器严格要求 36 行完整矩阵和统一 commit/device，并验证 boundary case 的策略语义。`completed_tokens` 包含随后被取消请求已经执行的无效工作；`useful_tokens` 只统计最终完成请求，因此策略吞吐结论优先使用 `useful_tokens_per_second`。任何 `resource_deadlocks > 0` 的行都必须与完成率、取消数一起解释，不能只比较 p50。

Multi-layer transaction 快速验证：

```bash
python benchmarks/run_multi_layer_engine.py \
  --case l2_b4_c128 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/multi_layer_transaction_quick.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/multi_layer_transaction_quick.csv \
  --output benchmarks/results/multi_layer_transaction_quick_summary.md \
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
  --output benchmarks/results/multi_layer_transaction_trials3.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/multi_layer_transaction_trials3.csv \
  --output benchmarks/results/multi_layer_transaction_summary.md
```

正式矩阵共 `12 cases x 2 dtypes x 2 backends x 3 trials = 144 rows`。输入生成、context seed、JIT build、profiler probe 和 rollback probe 均排除在正式 complete-token latency 外；profiler 字段只做 append/decode/launch 归因，rollback latency 不混入正常吞吐。Summary 同时报告 ratio 与 torch/fused 绝对 attribution median，任何低于 1 的 ratio 都必须结合绝对时间解释。

commit `fa0f89a` 的 RTX 5070 正式结果已通过 144 行严格校验。fused complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`；24 个 dtype/case 组合中 20 个三轮 p50 稳定胜出、4 个跨过 1.0。每轮仅 20 repeats，nearest-rank p99 接近单轮最大值，因此必须连同范围报告。

Shared-prefix quick 验证：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --quick \
  --hit-rate all \
  --dtype float16 \
  --trials 1 \
  --output benchmarks/results/shared_prefix_workload_quick.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/shared_prefix_workload_quick.csv \
  --output benchmarks/results/shared_prefix_workload_quick_summary.md \
  --expected-trials 1 \
  --expected-dtypes float16
```

正式矩阵：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/shared_prefix_workload_trials3.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/shared_prefix_workload_trials3.csv \
  --output benchmarks/results/shared_prefix_pre_metadata_cache_summary.md
```

正式矩阵共 `4 hit rates x 2 dtypes x 3 trials = 24 rows`。`capacity_admitted_requests` 来自固定 60% bounded pool 的第一次调度；decode latency 使用独立的无共享基线容量，使四档 hit rate 始终运行相同 batch。context 构建、materialization correctness、extension/Triton warmup、registration 和 attach probe 均排除在 complete-step latency 外。共享 prefix 不改变 attention 算法，因此 latency 小幅变化不能单独解释为 kernel 加速。

commit `fd36ed0` 的 RTX 5070 FP16 quick 已通过 4 行严格校验。0%/25%/50%/75% 的 context physical/logical blocks 分别为 `4/4`、`4/4`、`3/4`、`2/4`，bounded-pool admission 分别为 `2/4`、`2/4`、`3/4`、`3/4`。quick 每档只有 3 次正式 step 采样，latency/TPS 非单调，不形成正式性能结论。

commit `1d5d8d0` 的 RTX 5070 FP16/BF16 三轮正式矩阵共 24 行并通过严格校验。0%/25%/50%/75% 的 context physical blocks 为 `64/52/36/20`，bounded-pool admission 为 `9/12/15/16`，75% hit 避免占用 `68.8%` context KV capacity 和 `5.5 MiB`。summary 将每个 hit-rate 与同 dtype、同 trial 的 0% baseline 配对，报告 p50/p90/p99/TPS median `[min,max]`，并单独报告 scheduler/Engine p50 attribution。只有全三轮同向才标记 `shared_faster` 或 `shared_slower`。

Hot-path metadata cache correctness 通过后，使用 8 trials 做扩大样本的 confirmation：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 8 \
  --output benchmarks/results/shared_prefix_workload_trials8.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/shared_prefix_workload_trials8.csv \
  --output benchmarks/results/shared_prefix_capacity_summary.md \
  --expected-trials 8 \
  --expected-dtypes float16 bfloat16
```

commit `fe72e27` 的 RTX confirmation 共 64 行，seed `613-620`，四种 hit-rate 顺序各轮转两次。容量轨迹与旧 3-trial matrix 一致；所有非零 complete、scheduler 与 Engine p50 paired range 都跨过 1，因此性能结论是 near-neutral/no stable direction。旧 3-trial summary 作为优化前负结果基线保留，不能与新 run 直接相除声称 metadata cache 的因果 speedup。

Trusted transaction quick validation：

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case l2_b4_c128 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/trusted_transaction_quick.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/trusted_transaction_quick.csv \
  --output benchmarks/results/trusted_transaction_quick_summary.md \
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
  --output benchmarks/results/trusted_transaction_l4_stress.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/trusted_transaction_l4_stress.csv \
  --output benchmarks/results/trusted_transaction_l4_stress_summary.md \
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
  --output benchmarks/results/trusted_transaction_trials5.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/trusted_transaction_trials5.csv \
  --output benchmarks/results/trusted_transaction_summary.md \
  --expected-trials 5
```

正式矩阵共 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`，strict summary 报告 80 个 paired trials。`checked` 与 `trusted` 复用相同 Cache transaction API；runner 只在 benchmark context 中把 Cache 内部 raw launch 切换为 checked 或 trusted，因此状态机和 Engine 路由不变。每个 paired trial 先按轮换顺序完成两条 path 的 non-instrumented synchronized wall，再统一执行 profiler/rollback，避免 attribution 重采集夹在两侧 wall 之间。CPU-only profiler 先在 WARMUP cycle 执行并 abort 一个同形 token，再在唯一 active cycle采集；每个 layer 必须有精确一个 CPU user annotation，其 inclusive CPU total保留 checked 路径 `.item()` 的 host/stream 等待。checked 每个 profiled layer 必须有 5 次 `aten::item`/`aten::_local_scalar_dense`，trusted 为 0。少记 range/scalar 或缺少有效 CPU time可用相同 seed、全新 engine/profiler 重采集，固定最多三次并写入 `profile_attempt_count`；多出 range/scalar 则视为 active-work/fast-path 契约回归并立即终止，不能用重试掩盖。append/decode device time与 CUDA activity不再属于该 strict schema；若未来需要分段 GPU attribution，使用独立 CUDA Event/Nsight probe。summary validator 负责证据完整性，不替代人工性能 gate：overall p50 至少 `1.05x`，且全部 16 个 `dtype x case` 分组的五轮 p50 `[min,max]` 都不穿过 1。该证据只归因于 device-value validation；transaction-view H2D materialization/copy 由独立 persistent-metadata 实验评估。

commit `4018449` 的 RTX 5070/CUDA 12.8 矩阵通过严格校验：160 rows、80 paired trials，全部 16 个 `dtype x case` 分组均为 `trusted_faster` 且五轮 p50 最小值都大于 1。overall complete-token p50/p90/p99、TPS 和 append CPU/layer ratio 分别为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`；focused `73 passed, 23 subtests passed`，完整回归 `410 passed, 48 subtests passed`。7/16 分组的 p99 range 穿过 1，因此不声明稳定尾延迟收益。Trusted path 保持默认，canonical evidence 见[五轮配对摘要](results/trusted_transaction_summary.md)；persistent-metadata candidate 的负结果单独保留。

Integrated scheduled multi-layer quick：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case l2_c64 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/integrated_runtime_lifecycle_quick.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_quick.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_quick_summary.md \
  --expected-trials 1 \
  --expected-cases l2_c32 \
  --expected-dtypes float16
```

正式矩阵：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/integrated_runtime_lifecycle_trials3.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_trials3.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_summary.md \
  --expected-trials 3
```

formal 为 `2 layer counts x 2 contexts x 2 dtypes x 3 trials = 24 rows`。每 row 是完整 dynamic trace；strict summary 重建 dependency-free reference，验证 admission/defer/completion/cancel、layer-1 rollback、transaction/prefix 计数、released-block reuse、observed/reference digest 与 terminal zero-used cleanup。它只报告绝对 workload latency/TPS，不构造 shared-prefix speedup，也不重新比较 persistent-metadata candidate。完整 schema 见[integrated workload 设计](../docs/design_integrated_scheduled_multi_layer.md)。

FlashInfer formal baseline：

```bash
python benchmarks/run_flashinfer_baseline.py \
  --case all \
  --dtype both \
  --trials 3 \
  --warmup 10 \
  --repeat 50 \
  --require-clean \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv"

python benchmarks/summarize_flashinfer_baseline.py \
  --input "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv" \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_summary.md" \
  --expected-trials 3 \
  --expected-warmup 10 \
  --expected-repeats 50
```

完整矩阵为 `4 cases x 2 dtypes x 3 backends x 3 trials = 72 rows`。commit `d7d4feb` 记录 focused `93 passed, 37 subtests passed`、quick、formal、full `453 passed, 94 subtests passed` 与 clean-tree evidence check。它只比较共同 paged-decode kernel：输入、page table、layout 语义与 CUDA-event timing 对齐，FlashInfer plan/JIT 排除在计时外；不比较 scheduler、KV ownership、transaction 或完整 serving。固定 cu128/SM120a 环境、quick 命令与安装步骤见[FlashInfer 复现章节](../docs/reproducibility.md#flashinfer-有限外部基线)，canonical evidence 见[FlashInfer 摘要](results/flashinfer_paged_decode_baseline_summary.md)。

## vLLM Qwen2.5-3B 外部比较

R7 命令绑定历史闭环提交 `61836b6`，必须在该提交的独立 worktree/clone 与固定 `vLLM==0.25.1` 环境执行，并显式启用 plugin。当前 HEAD 的 model-latency runner/summarizer 已冻结为 R8 协议，不能用来重写 R7 的 12-row 负结果：

```bash
export VLLM_PLUGINS=flashdec
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export MODEL_DIR=/home/<user>/models/Qwen2.5-3B-Instruct
export RESULT_DIR=/home/<user>/flashdec_results/r7_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"
```

Attention formal gate：

```bash
python benchmarks/run_vllm_attention_microbench.py \
  --trials 5 --warmup 100 --repeat 500 \
  --output "$RESULT_DIR/vllm_attention.csv"

python benchmarks/summarize_vllm_attention_microbench.py \
  "$RESULT_DIR/vllm_attention.csv" \
  --output "$RESULT_DIR/vllm_attention_summary.md"
```

模型正确性需要两个独立 backend 进程：

```bash
python benchmarks/run_vllm_model_correctness.py \
  --backend TRITON_ATTN --model "$MODEL_DIR" \
  --output "$RESULT_DIR/model_correctness_triton.json"

python benchmarks/run_vllm_model_correctness.py \
  --backend CUSTOM --model "$MODEL_DIR" \
  --output "$RESULT_DIR/model_correctness_flashdec.json"

python benchmarks/summarize_vllm_model_correctness.py \
  "$RESULT_DIR/model_correctness_triton.json" \
  "$RESULT_DIR/model_correctness_flashdec.json" \
  --output "$RESULT_DIR/model_correctness_summary.md"
```

固定批量模型和在线 serving formal：

```bash
python benchmarks/run_vllm_model_latency.py \
  --model "$MODEL_DIR" \
  --output "$RESULT_DIR/model_latency.csv" \
  --trials 3 --warmup-iters 3 --num-iters 5 --require-clean

python benchmarks/summarize_vllm_model_latency.py \
  "$RESULT_DIR/model_latency.csv" \
  --output "$RESULT_DIR/model_latency_summary.md"

python benchmarks/run_vllm_serving_benchmark.py \
  --model "$MODEL_DIR" \
  --output "$RESULT_DIR/serving.csv" \
  --trials 3 --num-prompts 128 --num-warmups 8 \
  --input-len 4096 --output-len 128 --max-concurrency 8 \
  --port 8127 --require-clean

python benchmarks/summarize_vllm_serving_benchmark.py \
  "$RESULT_DIR/serving.csv" \
  --output "$RESULT_DIR/serving_summary.md"
```

历史提交 `61836b6` 的正式 R7 中，attention 与 model-correctness summary 返回 0；model-latency 与 serving summarizer 在写出完整报告后按冻结门槛返回非零。复现者应把这个非零结果视为被验证的负结果，不能删除失败项、降低 threshold，或用当前 R8 summarizer 覆盖原报告。四份 canonical evidence 见[结果索引](results/README.md#r7-vllm-qwen外部比较)，完整实现边界见 [vLLM backend 设计](../docs/design_vllm_backend.md)。

### R8 长上下文 fixed-batch formal

R8 保留上面的 R7 负结果，并新增一个单独预注册的长上下文目标。使用同一个固定 vLLM/Qwen 环境，显式启用 V1 multiprocessing 和 FlashDec plugin：

```bash
export VLLM_PLUGINS=flashdec
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
unset FLASHDEC_VLLM_NUM_SPLITS
export MODEL_DIR=/home/<user>/models/Qwen2.5-3B-Instruct
export RESULT_DIR=/home/<user>/flashdec_results/r8_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"

(cd "$MODEL_DIR" && sha256sum --check SHA256SUMS)

python benchmarks/run_vllm_model_latency.py \
  --model "$MODEL_DIR" \
  --output "$RESULT_DIR/model_latency.csv" \
  --case qwen_b8_i512_o2 \
  --case qwen_b8_i8192_o4096 \
  --trials 4 \
  --prime-iters 1 \
  --warmup-iters 1 \
  --num-iters 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --vllm-cache-base "$RESULT_DIR/vllm-cache" \
  --require-clean

python benchmarks/summarize_vllm_model_latency.py \
  "$RESULT_DIR/model_latency.csv" \
  --output "$RESULT_DIR/model_latency_summary.md"
```

commit `3ba68e3` 在 RTX 5070 上得到如下正式结果：

| case | paired ratio FlashDec/vLLM | 结果 |
| --- | ---: | --- |
| `qwen_b8_i512_o2` guard | `1.0029x [0.9890,1.0100]` | `<= 1.05x`，PASS |
| `qwen_b8_i8192_o4096` target | `0.9542x [0.9530,0.9560]` | latency `-4.58%`、TPS `+4.80%`；`<= 0.970x`，PASS |

runner 对每个 case 使用 4 个独立 process pairs（共 8 对）、balanced AB/BA、每进程 1 次 full-length JIT-prime、1 次 warmup 和 1 次 measured generation。8/8 `CUSTOM` workers 的 capture-time marker 都验证为 B8/Q16/KV2/D128/BF16/8 splits。canonical summary 见 [R8 长上下文模型摘要](results/vllm_qwen_long_context_model_latency_summary.md)。仓库外原始证据的 SHA-256 为：

- CSV：`ae57b1788abb61847e1faa4ee1a6ab57de0fba309c2cb5317d660e4913d503e2`；
- summary：`f511af02757b66cc75007768c5df7e9180ae31f3ed34853d00d00038e9354520`；
- run log：`8c1956118f877f4f006c4fba50f9c91c814dcc283457c84ebc3ec59723a4b7f7`；
- summary log：`1c8b5e95716fbd1a84528e8d39d6420d1d0b7b3a9023403c854edba2a4509bc6`；
- evidence manifest：`cf7ea96e39133ff4bf12959877b177c31547d9eb6f17273e94ab35d19946fd57`。

这里测量的是离线固定 B8 的 blocking `LLM.generate`：计时包含 model execution、scheduler、KV cache、sampling 和 API 调用，排除 startup/model load、full-length JIT-prime 和 warmup。它不是 online serving benchmark，不能从这组结果推导 TTFT、TPOT、并发请求吞吐或默认/最快 vLLM backend 的收益；外部基线明确固定为 `TRITON_ATTN`。
