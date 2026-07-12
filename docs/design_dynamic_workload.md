# Dynamic Workload 与系统级指标设计

## 目标

Week 12 不再把一次 CUDA kernel launch 当作系统性能，而是验证单 GPU DecodeEngine 在动态请求下能否正确调度、回收 Paged KV block，并提供可追溯的端到端 decode-step 指标。

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

这个边界故意不同于 Week 11 的 CUDA-event append microbenchmark：前者用于解释单条数据路径的 GPU 开销，后者用于观察优化是否在 runtime 调度、allocator 与不规则 active batch 下仍能传递为系统收益。两者不能直接横向比较毫秒数。

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
  --output benchmarks/results/week12_decode_engine_workload_quick.csv
```

quick 成功后运行完整实验：

```bash
python benchmarks/run_decode_engine_workload.py \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload.csv
```

默认同时测试 `append_backend=torch` 和 `append_backend=fused_cuda`，decode 固定为已冻结的 Triton `block_size=32, num_warps=2`。如果只定位 fused GPU path，可添加 `--append-backends fused_cuda`；此时 CSV 不会包含相对 torch 的 speedup。

## 结果解释规则

1. 首先确认每行 `validated_invariants=True`，以及 `final_used_blocks + final_free_blocks == max_blocks`。
2. 检查 `long_pressure` 有非零 `backpressure_steps`，并且 `free_count/reuse_count` 随取消后续请求而增长；否则没有真正覆盖压力恢复。
3. 使用同一 dtype/workload 的 `speedup_vs_torch_p50` 判断 fused 的端到端传递效率；不要把 p50 与不同 workload 的 p50 混合比较。
4. `p99` 应与 `p50/p90` 一起看。短负载本身噪声较大；只有跨 workload、重复运行方向一致才做默认路径结论。
5. 记录至少一个负结果：若 append-only fused 更快但 complete-step speedup 接近 1，说明 attention、allocator 或 Python/synchronization 已成为主导。这个结论同样是 AI Infra 项目的有效交付。
