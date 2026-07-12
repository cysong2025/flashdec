# Week 12 状态记录

## 本周主题

Dynamic DecodeEngine workload、端到端性能与 Paged KV runtime 可观测性。

## 当前已完成

- 新增 `WorkloadConfig` / `WorkloadResult` 与 deterministic synthetic workload runner。
- 覆盖 short churn、mixed steady context、long memory pressure 三类请求轨迹。
- 完整计时包含 submit/admit、allocator、RoPE/KV append、paged decode 和 finish/cancel；排除外部 Q/K/V 生成、prompt prefill、warmup 与 JIT。
- 新增 p50/p90/p99、tokens/s、active batch、block utilization/fragmentation、allocation/free/reuse 与 backpressure CSV 字段。
- RTX 5070 focused/full correctness 已反馈通过；quick/full workload CSV 已同步并完成分析。
- 12/12 full rows 均通过 engine/cache invariant，且 block accounting 完整。

## 正式 multi-trial 结果

- commit `3708b87` 生成 36 行：3 workloads x 2 dtypes x 2 backends x 3 trials。
- 36/36 行通过 engine/cache invariant、block accounting、pair trajectory、seed 和 backend-order 校验。
- fused complete-step p50/p90/TPS 几何平均：1.0668x/1.0317x/1.0811x vs torch。
- mixed-steady 与 long-pressure 的 FP16/BF16 p50 三轮均高于 1.0，median 为 1.0890x-1.0948x。
- short-churn 的 FP16/BF16 p50 均跨过 1.0，不能声明稳定 backend 胜出。
- p99 总体几何平均为 1.2590x，但单场景 trial ratio 范围为 0.2444x-5.0578x；只能作为高噪声尾延迟观测，不能用总体均值声明稳定收益。

详细表见 `benchmarks/results/week12_decode_engine_workload_trials3_summary.md`。

## Complete-step 阶段归因

- 正式 profiler 覆盖 3 workloads x 2 dtypes x 2 append backends，共 12/12 场景。
- named ranges 与执行轨迹一致：short/mixed 为全部成功，long-pressure 为 80 successful + 32 backpressure。
- fused 相对 torch 将 CUDA event 数减少 21.8%-45.6%。
- profiler wall p50 改善 1.0653x-1.3346x；该数只用于归因，不替代 non-instrumented multi-trial。
- paged decode device total 的变化仅为 -1.7%-+1.1%，说明 attention kernel 本身基本未改变。
- 因此 fusion 收益主要来自减少 RoPE/KV append 的中间操作、kernel launch 与 Python/runtime 路径，而不是让 paged attention 更快。
- long-pressure FP16 instrumented p99 回退到 0.8064x，继续证明 profiler 延迟和 p99 不适合用作 release 性能决策。

详细表见 `benchmarks/results/week12_decode_engine_profile_summary.md`。

## Runtime 结论

- short churn：120/120 successful steps，221 finished、24 cancelled、250 reused allocations，无 backpressure。
- mixed steady：160/160 successful steps，mean active batch 15.963，完成 76 个不同 context 请求。
- long pressure：80 successful + 32 backpressure steps；取消 32 个请求后复用 32 个 block，最终 16 used + 0 free，invariant 仍成立。
- short churn 的 FP16/BF16 p50 speedup 分别只有 0.9916x/0.9801x，是应保留的系统级负结果。

## 本次方法改进

单次 full run 的 BF16 fused p99 与 FP16 torch p99 都存在离群值，quick/full 的 short-churn 方向也不稳定。因此 benchmark 新增：

- `--trials N`。
- 相邻 trial 反转 backend 顺序。
- 每个 trial 使用递增 seed。
- CSV 记录 trial、trial_count、backend_order 和 seed。
- 新增 `summarize_decode_engine_trials.py`，只有通过完整矩阵、pair trajectory、block accounting、seed/order 和 invariant 校验后才输出聚合摘要。
- 聚合摘要同时报告 per-trial ratio、跨 trial median/min/max/geometric mean，并将 p50 跨过 1.0 的场景标记为 `unstable_crosses_1`。
- 新增 `DecodeEngine(profile_ranges=True)` 的 preflight/append/decode ranges；默认关闭且不增加 CUDA synchronization。
- 新增 `profile_decode_engine.py`，用专用 Engine 包装 submit/admit/complete step/finish/cancel，并输出 stage CPU/device totals、CUDA event count、profile table、summary 和可选 Chrome trace。
- profiler 结果只做归因；正式 latency 仍来自 non-instrumented multi-trial CSV。
- 新增 `docs/reproducibility.md`、Unreleased `CHANGELOG.md`、README quick start/support matrix 和 `scripts/check_release.py`。
- package 核心依赖精简为 torch/triton，pytest 与 Ninja 分别进入 `dev`/`cuda-extension` extras。
- release candidate 文档/检查器、正式 multi-trial 和 profiler evidence 已完成；clean WSL venv、版本 `0.1.0` 和 tag 明确保持 pending。

## RTX 5070 下一步：clean-install gate

```bash
cd ~/work/flashdec
git pull --ff-only origin main
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python scripts/run_r0_validation.py --phase local --phase focused --phase full
python scripts/run_r0_validation.py --phase trials-quick --phase profile-quick
```

以上命令必须在新目录和新 venv 中执行，并记录安装、环境、pytest 与 quick evidence 输出。正式 trial/profile 不需要重复生成。

release candidate 结构可先检查：

```bash
python scripts/check_release.py --require-clean
```

当前版本仍为 `0.0.0`；clean-install gate 通过后才升级版本并创建 `v0.1.0` tag。
