# FlashDec Performance Report

本文汇总 FlashDec 的正式性能与系统实验。它回答的是设计问题，而不是按开发时间罗列结果；逐行数据、环境和命令以 [`benchmarks/results/`](../benchmarks/results/README.md) 中的 canonical summaries 为准。

## 解释规则

- 所有比值都必须与各表定义一起阅读；不同章节的 ratio 不能相乘。
- kernel-only CUDA-event、完整 DecodeEngine wall-clock 和 instrumented profiler 是不同计时边界。
- p50/p90 报告跨 trial 的中位数与范围；有限 repeats 的 p99 只用于展示波动。
- profiler 用于归因，不替代 non-instrumented latency。
- logical workload GB/s 是 shape-normalized proxy，不是硬件 DRAM bandwidth。
- 容量节省、admission 改善和 latency 是三个独立结果。

## 结果概览

| 研究问题 | 主要观察 | 权威证据 |
| --- | --- | --- |
| Paged-decode 默认配置 | token-major、block 32、2 warps；显式 staging 没有达到 5% 门槛 | [kernel experiments](kernel_experiments.md) |
| Append fusion 是否传递到完整 step | complete-step p50/TPS 几何平均 `1.0668x/1.0811x`，明显小于 append-only 收益 | [DecodeEngine multi-trial](../benchmarks/results/decode_engine_workload_trials3_summary.md) |
| Scheduler 在容量压力下是否保证进展 | lifetime policy 在 boundary case 完成 `100%`；cancel/greedy 为 `50%/0%` | [scheduler matrix](../benchmarks/results/scheduler_capacity_progress_summary.md) |
| 多层事务的主要开销在哪里 | fused complete-token p50/TPS 为 `1.2101x/1.2800x`，收益主要来自 append/launch | [multi-layer matrix](../benchmarks/results/multi_layer_transaction_summary.md) |
| Shared prefix 带来什么 | 75% hit 节省 `68.8%`/`5.5 MiB` context KV capacity，admission `9/16 → 16/16`；latency 无稳定方向 | [shared-prefix confirmation](../benchmarks/results/shared_prefix_capacity_summary.md) |
| Trusted validation 是否值得 | trusted/checked complete-token p50 `1.7307x`；persistent metadata 只有 13/16 组稳定，未采用 | [trusted](../benchmarks/results/trusted_transaction_summary.md) · [negative candidate](../benchmarks/results/persistent_metadata_candidate_summary.md) |
| 与 FlashInfer 的共同 kernel 对比如何 | FlashDec/FlashInfer p50 ratio `1.2003x/1.2284x`，方向有利于 FlashInfer；不外推到 runtime | [FlashInfer baseline](../benchmarks/results/flashinfer_paged_decode_baseline_summary.md) |
| 与 vLLM/Qwen 的外部优化如何 | R7 B8 decode-attention p50 降低约 20%；R8 长上下文 `LLM.generate` 延迟降低 `4.58%`、TPS 提升 `4.80%` 并通过 3% 门槛；R7 较短模型与 serving 负结果保留 | [vLLM kernel](../benchmarks/results/vllm_qwen_attention_summary.md) · [R8 long-context model](../benchmarks/results/vllm_qwen_long_context_model_latency_summary.md) · [R7 model](../benchmarks/results/vllm_qwen_model_latency_summary.md) · [serving](../benchmarks/results/vllm_qwen_serving_summary.md) |

## Paged-decode kernel

### 默认配置

| 维度 | 选择 | 核心证据 |
| --- | --- | --- |
| KV layout | token-major | 25/28 p50 和 25/28 p90 case 胜出 |
| block size | 32 | 24/28 p50 case 胜出；相对 block 16 的 p50 几何平均 `1.31x` |
| `num_warps` | 2 | 28/28 dtype/case 的 p50 最优 |
| `num_stages` | implicit (`None`) | 最佳显式候选只有 `1.0039x`，未达到预设 5% 门槛 |

完整假设与 sweep 见[受控 kernel 实验](kernel_experiments.md)。默认配置在 RTX 5070 上的代表性 CUDA-event p50：

| case | FP16 p50 | BF16 p50 |
| --- | ---: | ---: |
| batch 1 / context 128 | `0.015328 ms` | `0.038176 ms` |
| batch 16 / context 1024 | `0.155520 ms` | `0.160864 ms` |
| batch 16 / context 8192 | `0.884576 ms` | `0.928064 ms` |
| batch 64 / context 4096 | `1.934560 ms` | `1.961216 ms` |

长 context 的逻辑流量几乎全部来自 K/V 读取，latency 随 context 近似线性增长；估算有效带宽在较大工作量下约为 `1.1–1.75 TB/s`。这支持 memory-bound 解释，但没有 Nsight hardware counters，不能写成实测带宽结论。完整表见[default profile](../benchmarks/results/paged_decode_default_profile_summary.md)。

## Kernel 优化到完整 step 的传递

commit `3708b87` 的正式矩阵包含 3 workloads、FP16/BF16、torch/fused 两条 append path 和 3 trials，共 36 行。ratio 为 `torch/fused`，大于 1 表示 fused path 更低延迟或更高吞吐。

| dtype | workload | p50 median [min,max] | TPS median | 解释 |
| --- | --- | ---: | ---: | --- |
| FP16 | short-churn | `1.0001x [0.9311,1.0042]` | `1.0928x` | p50 跨 1 |
| FP16 | mixed-steady | `1.0927x [1.0837,1.1109]` | `1.1004x` | 稳定方向 |
| FP16 | long-pressure | `1.0890x [1.0614,1.1274]` | `1.0899x` | p50 稳定，p99 不稳定 |
| BF16 | short-churn | `1.0366x [0.9892,1.0508]` | `1.0735x` | p50 与 p99 不稳定 |
| BF16 | mixed-steady | `1.0948x [1.0882,1.2193]` | `1.1651x` | 稳定方向 |
| BF16 | long-pressure | `1.0744x [1.0741,1.1054]` | `1.0754x` | p50 稳定，p99 不稳定 |

总体 p50/p90/TPS 几何平均为 `1.0668x/1.0317x/1.0811x`。12-case profiler 显示 fusion 将 CUDA event 数减少 `21.8%–45.6%`，而 paged-decode device time 只变化 `-1.7%–+1.1%`；主要收益来自 append、launch 和 runtime 路径。long-pressure FP16 的 instrumented CPU total 仍有回退，不能宣称每个 workload 都改善。完整归因见[stage profile](../benchmarks/results/decode_engine_stage_profile_summary.md)。

## 容量压力下的调度进展

commit `16de9d4` 的 36-row policy matrix 比较 lifetime FIFO + aging、cancel-on-backpressure 和 greedy-step-only。

| workload | lifetime | cancel | greedy |
| --- | ---: | ---: | ---: |
| boundary deadlock completion | `100%` | `50%` | `0%` |
| forced cancellation | `0` | 有 | `0` |
| detected resource deadlock | `0` | `0` | 每 trial 1 次 |

finite queue 下三种 policy 都能完成，因此这里证明的是 capacity commitment、公平等待和进展保证，不是普通 workload 的无条件 latency speedup。

## 多层 token 事务

### Fused complete-token path

commit `fa0f89a` 的 144-row matrix 覆盖 12 cases、FP16/BF16、torch/fused 和 3 trials。fused/torch TPS 与 torch/fused latency 的几何平均分别为：

| metric | ratio |
| --- | ---: |
| complete-token p50 | `1.2101x` |
| complete-token p90 | `1.3826x` |
| TPS | `1.2800x` |
| per-layer append device | `1.6103x` |
| decode device | `1.0024x` |
| CUDA event count | `1.9784x` |

24 个 dtype/case 分组中 20 个 p50 三轮稳定胜出，4 个跨 1；decode device 基本不变，说明收益来自 append 和 launch 路径。每轮只有 20 repeats，p99 接近样本最大值，不作生产尾延迟结论。

### Cache-owned trusted validation

公开 raw append 必须检查 CUDA index 值域；Cache allocator 已在 host 上生成并拥有 transaction positions。trusted path 把这组证明放在 `begin_token()`，transaction API 回查内部 state 后调用 private raw launch，避免每 layer 重复 reduction 与 `.item()`。

commit `4018449` 的 160-row/80-pair 五轮矩阵中，16/16 个分组的 p50 最小值都高于 1：complete-token p50/TPS 为 `1.7307x/1.7131x`，append CPU/layer 为 `2.3612x`。7/16 个 p99 range 仍跨 1；CPU-only attribution 证明移除了 host scalar sync，不代表 device math 加速。

### Persistent metadata 负结果

commit `8047a9c` 的 persistent-metadata candidate 将 transaction views 从 l2/l4 的 `6/10` 降为 1，overall p50/TPS 为 `1.2493x/1.2392x`。但只有 13/16 个分组的五轮 p50 全部高于 1，未达到预设 16/16 条件，因此默认实现继续使用 materialized metadata。该负结果保留在正式证据中。

### 组合状态机验证

commit `6912894` 的 24-row matrix 把 scheduler、multi-layer transaction、shared prefix hit/miss、failure rollback、block reuse 和 cleanup 放进同一 trajectory。所有 reference digest、lifecycle 和最终零占用不变量均通过。轨迹只有 10 个 logical steps；p90/p99 受 context-import steps 主导，不解释为 steady-state decode tail。[完整摘要](../benchmarks/results/integrated_runtime_lifecycle_summary.md)

## Shared-prefix ownership 与容量

commit `fe72e27` 的 8-trial/64-row confirmation 固定 16 requests、128-token context、block size 32 和 48-block bounded pool。

| hit rate | context physical/logical blocks | saved capacity | admission |
| ---: | ---: | ---: | ---: |
| 0% | `64/64` | `0%` | `9/16` |
| 25% | `52/64` | `18.8%` | `12/16` |
| 50% | `36/64` | `43.8%` | `15/16` |
| 75% | `20/64` | `68.8%` / `5.5 MiB` | `16/16` |

75% hit 避免 44 个重复 context blocks；decode tail 仍是 request-private，因此 context saving 与 peak reduction 不是同一个百分比。所有非零 hit-rate 的 complete、scheduler 和 Engine p50 range 都跨 1，稳定结论仅限 ownership correctness、KV capacity 和 admission。

## FlashInfer 共同 kernel baseline

commit `d7d4feb` 的 72-row matrix 固定 Python 3.12.3、PyTorch `2.11.0+cu128`、Triton 3.6.0、CUDA 12.8 和 FlashInfer `0.6.15.post1`。三个 backend 共用 Q/K/V、logical pages、page table、sequence lengths、scale、dtype 和 seed；CUDA event 只包围 `run`/kernel dispatch，FlashInfer plan/JIT 与 metadata 构建在计时外。

| FlashInfer execution | FlashDec/FlashInfer p50 几何平均 | 三轮 range 高于 1 |
| --- | ---: | ---: |
| FA2 CUDA core | `1.2003x` | `8/8` |
| FA2 tensor core | `1.2284x` | `8/8` |

大于 1 表示 FlashInfer latency 更低。绝对 p99 有 7/16 个 range 重叠，且两组 tensor-core p99 中位数方向反转；每 row 只有 50 repeats。这个实验不能比较 scheduler、KV ownership、transaction、prefix cache 或完整 serving。

## vLLM Qwen2.5-3B 外部比较

R7/R8 使用 `vLLM==0.25.1` 的 `CUSTOM` out-of-tree attention backend。vLLM 继续拥有模型、prefill、KV cache、scheduler、sampling 和 HTTP server；FlashDec 只替换符合条件的 uniform single-token decoder attention，不支持的路径回退 vLLM Triton。实现合同见 [vLLM backend 设计](design_vllm_backend.md)。证据环境为 RTX 5070、Qwen2.5-3B BF16、PyTorch `2.11.0+cu130`、Triton `3.6.0` 和 PyTorch CUDA 13.0。

### R7 attention kernel gate：通过

commit `1cc25d4` 的 50-row matrix 包含 5 个 case、两个 backend 和每 case 5 个交替顺序的 paired trials。`triton.testing.do_bench` 使用 100 ms warmup、500 ms measurement；每个 pair 在计时前完成 full-output cross-backend correctness。

| case | vLLM Triton p50 | FlashDec p50 | FlashDec/vLLM |
| --- | ---: | ---: | ---: |
| B1 / ctx128 | `0.009696 ms` | `0.009952 ms` | `1.0264x` |
| B1 / ctx1024 | `0.013792 ms` | `0.013792 ms` | `1.0000x` |
| B4 / ctx1024 | `0.017888 ms` | `0.017888 ms` | `1.0000x` |
| B8 / ctx1024 | `0.030144 ms` | `0.024192 ms` | `0.8025x` |
| B8 / ctx2048 | `0.048608 ms` | `0.038528 ms` | `0.7926x` |

B8 两个预注册目标均低于 `0.90x`；B1/B4 和全 case `1.05x` guardrail 通过。这里可以宣称在冻结的 Qwen decode shapes 上，FlashDec attention p50 相对 vLLM Triton 降低 `19.75%` 和 `20.74%`。它不能被外推为模型或服务加速。

### R7 模型级固定批量：小幅改善，目标失败

commit `46c4a4b` 的默认 Inductor/CUDA Graph 运行采用两个 B8 case、每 backend 3 个独立进程、每进程 3 次 warmup 与 5 次 measured `LLM.generate`。模型加载和 compilation 排除在 latency 外。

| case | vLLM p50 | FlashDec p50 | paired ratio | 冻结结论 |
| --- | ---: | ---: | ---: | --- |
| input128/output128 | `1557.505 ms` | `1551.073 ms` | `0.9958x` | guardrail PASS |
| input2048/output128 | `3224.064 ms` | `3218.875 ms` | `0.9976x` | `<=0.995x` target FAIL |

两个 case 的绝对方向都略有改善，但目标 case 只有约 `0.16%` 的 p50 降幅，不能写成通过模型性能门槛。整体 external-model gate 为 `FAIL`。

### R7 在线 serving：TPOT 通过，吞吐目标略失

commit `7dcb19c` 的标准 `vllm bench serve` 比较固定 128 prompts、concurrency 8、input4096/output128、request rate `inf`、8 warmups、prefix cache off 和 3 个独立 server pairs。所有 run 都完成 128/128 requests 且零失败。

| metric | vLLM Triton | FlashDec | paired ratio | gate |
| --- | ---: | ---: | ---: | --- |
| median TPOT | `31.9200 ms` | `31.8152 ms` | `0.9969x` | PASS (`<=0.998x`) |
| p90 TPOT | `35.3043 ms` | `35.1923 ms` | `0.9973x` | PASS (`<=1.02x`) |
| median TTFT | `1113.221 ms` | `1113.990 ms` | `1.0007x` | PASS (`<=1.05x`) |
| output throughput | `197.816 tok/s` | `197.751 tok/s` | `1.0019x` | FAIL (`>=1.002x`) |

TPOT 三轮 paired ratio 都低于 1，median/p90 分别改善约 `0.31%/0.27%`；throughput paired median 提高约 `0.19%`，但比冻结目标低约 `0.01` 个百分点，因此 overall external-serving gate 必须保持 `FAIL`。这组结果说明 kernel 优化已经传到真实服务 decode latency，但 Amdahl 效应使整体收益很小。

### R8 长上下文固定批量：3% 端到端目标通过

R8 保留 vLLM 对模型、scheduler、KV cache、sampling 和 `LLM.generate` 的所有权，只继续优化 eligible single-token split decode：partial reduction 改为按 query head 并行；Qwen 的 B8/16-query-head/2-KV-head/head-dim-128 路径由 auto policy 选择 8 splits；只能产生单 split 时回退原生 Triton。KV update 合同恢复为 vLLM 官方顺序：`forward_includes_kv_cache_update=False`，vLLM 先执行统一 KV update；后续 custom split launch 不接管 cache lifecycle。每个 CUSTOM worker 还必须在 CUDA Graph capture 中产生唯一 activation marker；worker/parent runner 读盘验证，strict summarizer 再校验 CSV 中的 commit、dataset、shape、8-split geometry、canonical JSON 与 marker SHA 投影。

commit `3ba68e3` 的确认性矩阵包含短路径 guard 和长上下文 target，各 4 个独立 backend process pairs，按 balanced AB/BA 顺序运行。每个进程执行 1 次 full-length JIT prime、1 次 warmup 和 1 次 measured `LLM.generate`；模型加载、engine startup、JIT/graph capture 和结果 hashing 均不在计时区间。

| case | vLLM p50 | FlashDec p50 | paired ratio | latency reduction | output TPS uplift | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| input512/output2 | `401.357 ms` | `403.915 ms` | `1.0029x [0.9890,1.0100]` | `-0.29%` | `-0.29%` | `<=1.05x` guard PASS |
| input8192/output4096 | `78118.907 ms` | `74578.237 ms` | `0.9542x [0.9530,0.9560]` | `4.58%` | `4.80%` | `<=0.970x` target PASS |

因此可以在冻结的 B8/input8192/output4096 离线 fixed-batch workload 上声明至少 3% 的完整 `LLM.generate` latency 收益；4 个 paired ratios 全部落在 `0.9530–0.9560x`。短 guard 只证明没有超过允许的 5% 回退，并不代表短请求加速。activation marker 证明同一 B8 decode graph 在计时前成功捕获了 FlashDec 8-split 路径；它不是对每一次 measured graph replay 的逐次 device trace。

该结果的“端到端”包括 Qwen transformer execution、scheduler、KV-cache access、sampling 和 Python API overhead，但排除 startup/JIT；它不是在线 serving 的 TTFT/TPOT/throughput 结论，也不能覆盖 R7 serving overall gate 的失败。完整命令、16-row matrix、外置 raw evidence SHA 与边界见 [R8 canonical summary](../benchmarks/results/vllm_qwen_long_context_model_latency_summary.md)。

### 正确性与解释边界

R7 固定 8 个 prompts 的第一步 greedy top-1 为 8/8 一致。完整 32-token rollout 为 5/8 一致，共享前缀为 217/256 tokens；诊断运行中 split 与 non-split FlashDec 的 8/8 完整 rollout 一致。R8 formal run 使用固定 token-ID datasets，并要求每个请求至少共享前两个输出 token；短 guard 为 8/8 完整一致，长 target 的最小共同前缀为 49 tokens/request、完整序列为 7/8 一致。跨实现不同 reduction 顺序可能在近似并列 logits 处改变一个 token，随后所有自回归输入不同。逐元素 kernel tolerance、前缀门槛与完整 rollout hash 是三个不同证据层，不能互相替代。

## 证据边界

- 核心/FlashInfer 代表环境为 RTX 5070 与 CUDA 12.8；vLLM/Qwen 证据使用同一 GPU 与 PyTorch CUDA 13.0。其他硬件或软件栈必须重新测量。
- 正式 Markdown summary 可提交，原始 CSV/log/profile 默认留在仓库外或由 `.gitignore` 排除；R8 外置 evidence 的路径和 SHA-256 记录在 canonical summary 中。
- 测试计数证明某个 commit 的 correctness，不是性能指标。
- 当前 `0.0.0` 不承诺稳定安装、API 或生产 serving 行为。
