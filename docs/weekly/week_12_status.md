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

## 首轮结果

- fused complete-step p50 几何平均：1.0537x vs torch。
- p90 几何平均：1.0588x。
- tokens/s 几何平均：1.0674x。
- p99 几何平均：0.9641x，当前不能证明尾延迟改善。
- FP16/BF16 p50 几何平均分别为 1.0529x/1.0546x。
- append-only latency 降低约 18.2%，complete-step p50 只降低约 5.1%；说明 kernel fusion 收益被 attention、Python runtime 与同步成本明显稀释。

详细表见 `benchmarks/results/week12_decode_engine_workload_summary.md`。

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

## RTX 5070 下一步

```bash
cd ~/work/flashdec
git pull --ff-only origin main
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python -m pytest -vv \
  tests/test_workload.py \
  tests/test_workload_benchmark.py \
  tests/test_decode_engine_trial_summary.py \
  tests/test_decode_engine.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py

python benchmarks/run_decode_engine_workload.py \
  --quick \
  --trials 2 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials2_quick.csv

python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials3.csv

python benchmarks/summarize_decode_engine_trials.py \
  --input benchmarks/results/week12_decode_engine_workload_trials3.csv \
  --output benchmarks/results/week12_decode_engine_workload_trials3_summary.md
```

三 trial 数据回来后先由聚合器冻结稳定/不稳定结论，然后进入 complete-step profiler、`v0.1.0` reproducibility、CHANGELOG 和 release 文档。
