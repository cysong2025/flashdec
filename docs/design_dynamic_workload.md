# Dynamic Workload 与系统级指标设计

## 研究问题

Dynamic workload 回答的问题是：单算子优化进入动态 request lifecycle 后，能否在正确调度、Paged KV block 回收和不规则 active batch 的共同开销下，继续转化为可追溯的完整 decode-step 收益。

它仍是单 layer、单 token decode 原型，不包含模型 forward、tokenizer、sampling、HTTP/RPC 或完整 prefill kernel。

## 可复现协议

`flashdec.WorkloadConfig` 固定以下输入：

- `steps`：测量窗口中的逻辑 decode step 数。
- `max_active`、`arrivals_per_step`：active batch 上限与每 step 的新请求数。
- `decode_tokens_per_request`：每个请求的 decode token budget；耗尽后调用 `finish_request()`。
- `initial_context_tokens`、`context_stagger_tokens`：可选的预置 context。第 `request_id` 个请求使用 `initial + (request_id % max_active) * stagger`，因此不同请求具有可复现的 context 长度。
- `cancel_interval`：成功 step 的周期性取消；`cancel_probability`：以独立且带 seed 的 RNG 对每个剩余 active request 做取消判定；`cancel_on_backpressure`：内存不足时取消 active batch 中最早的请求，并在后续 step 自然重试。
- `seed`：Q/K/V 随机输入种子。

每个逻辑 step 的状态转移为：

```text
arrival -> add_request / admit -> (optional prompt prefill)
        -> construct active Q/K/V -> DecodeEngine.step
        -> finish | periodic cancel | backpressure cancel
```

prompt prefill 只建立已有 KV context，它不属于单 token decode 路径。若 workload 的 prefill 配置超出 `max_blocks`，它是 benchmark 配置错误，应增大 block pool，而不是把异常记成 decode backpressure。

## 计时边界

每条 CSV latency 是一个 measured logical step 的 wall-clock：

```text
included:  request submit/admit + allocator + RoPE/KV append
           + paged decode + finish/cancel + CUDA synchronization
excluded:  random Q/K/V generation + prompt prefill + warmup + JIT build
```

这个边界故意不同于 CUDA-event append microbenchmark：后者解释单条数据路径的 GPU 开销，本 workload 观察优化是否在 runtime 调度、allocator 与不规则 active batch 下仍能传递为系统收益。两者不能直接横向比较毫秒数。

所有 `WorkloadResult` 计数（completed tokens、request events、backpressure）只覆盖测量窗口；最终 `engine_metrics` 是运行结束时的累计快照，包含 warmup，避免把热身状态误认为测量数据。

## 三种标准负载

| workload | 主要行为 | 预期观察 |
| --- | --- | --- |
| `short_churn` | 短 decode budget、每 step 多次到达、周期性取消 | block 回收/reuse、较小且波动的 active batch、scheduler 固定成本 |
| `mixed_steady` | 16 路稳态、不同初始 context、较长 token budget | context 不规则下的稳定 step p50/p90/p99 和 fragmentation |
| `long_pressure` | 16 路持续 decode、仅每请求一个 physical block 的 pool | block 边界触发明确 backpressure，取消后恢复且无泄漏 |

`long_pressure` 有意只为初始 16 行各留一个 block，因此所有行同时跨越 token-32 block boundary 时无法扩容；这是验证调度接口，而非模拟生产 OOM。

## 命令

在 RTX 5070 WSL 环境：

```bash
cd ~/work/flashdec
git pull --ff-only origin main
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python -m pytest -vv \
  tests/test_workload.py \
  tests/test_decode_engine.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py

python benchmarks/run_decode_engine_workload.py \
  --quick \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_quick.csv
```

Quick validation 通过后运行完整实验：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_trials3.csv
```

默认同时测试 `append_backend=torch` 和 `append_backend=fused_cuda`，decode 使用受控实验选出的 Triton `block_size=32, num_warps=2`。如果只定位 fused GPU path，可添加 `--append-backends fused_cuda`；此时 CSV 不会包含相对 torch 的 speedup。

正式结果使用 `--trials 3`。trial 1/3 按命令行 backend 顺序执行，trial 2 反转顺序；seed 从基础值逐 trial 加一。CSV 明确记录 `trial`、`trial_count`、`backend_order` 和实际 seed，避免固定执行顺序或单一输入样本主导结论。

## 结果解释规则

1. 首先确认每行 `validated_invariants=True`，以及 `final_used_blocks + final_free_blocks == max_blocks`。
2. 检查 `long_pressure` 有非零 `backpressure_steps`，并且 `free_count/reuse_count` 随取消后续请求而增长；否则没有真正覆盖压力恢复。
3. 使用同一 dtype/workload 的 `speedup_vs_torch_p50` 判断 fused 的端到端传递效率；不要把 p50 与不同 workload 的 p50 混合比较。
4. `p99` 应与 `p50/p90` 一起看。短负载本身噪声较大；只有跨 workload、重复运行方向一致才做默认路径结论。
5. 保留负结果：若 append-only fused 更快但 complete-step speedup 接近 1，说明 attention、allocator 或 Python/synchronization 已成为主导。这与正向 speedup 一样，是判断优化传播边界的有效系统证据。

## Complete-step 阶段归因

正式 latency 仍由 non-instrumented workload CSV 给出；阶段归因单独使用：

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

显式 ranges：

```text
request_submit / request_admit
engine_step (inclusive)
  engine_preflight
  rope_kv_append
  paged_decode
request_finish / request_cancel
```

`rope_kv_append` 包含 Python allocator/metadata 与所选 append backend；`engine_step` 包含三个内部 range。PyTorch profiler 的 nested CPU/device totals 可能重叠，不能直接相加。Q/K/V generation 和 prompt prefill 会出现在全局 profiler 中，但刻意放在命名 Engine ranges 之外。
