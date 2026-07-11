# FlashDec AI Infra 项目定位

## 一句话目标

FlashDec 不是单个 PagedAttention 算子的集合，而是一个面向单 GPU、单 token decode 阶段的执行与 KV Cache 管理原型。项目要贯通请求状态、Paged KV 内存管理、K/V 写入、attention kernel、动态 batch 执行和端到端性能评测。

## 系统分层

```text
Synthetic Workload / Request Lifecycle
                  |
                  v
        Decode Engine / Batch Builder
                  |
        +---------+---------+
        |                   |
        v                   v
Paged KV Cache Runtime   Decode Metadata
allocate/free/reuse      block_tables/seq_lens
        |                   |
        +---------+---------+
                  |
                  v
       RoPE + KV Append Data Path
                  |
                  v
       Triton Paged Decode Kernel
                  |
                  v
 correctness / latency / throughput / memory metrics
```

## 项目深度来自哪里

### 1. Kernel 层

- PyTorch dense/paged reference 定义语义。
- Triton dense decode 与 paged decode kernel。
- FP16/BF16、MHA/GQA/MQA、变长 batch、block size 与 layout。
- online softmax、FP32 accumulation、block table 间接索引。
- 参数实验、profiling 和负结果记录。

这一层回答“单个 GPU kernel 如何正确并高效地计算”。当前主线已基本完成，后续只保留有边界的 `num_stages` 和索引路径实验。

### 2. KV 内存管理层

- physical block pool。
- request 到 logical block list 的映射。
- block allocation、capacity check、free 与 reuse。
- request finish/cancel 后回收 block。
- 使用率、剩余容量、内部碎片和回收次数统计。

这一层回答“动态请求到达和结束时，KV Cache 如何管理显存”。当前只完成 append 与容量检查，free/reuse 和生命周期仍待实现。

### 3. Decode 数据路径

- 新 token 的 RoPE。
- K/V 写入 physical block。
- block table 与 seq_len 更新。
- append 与 paged attention 的接口衔接。
- PyTorch fallback 与 CUDA extension 路径。

这一层回答“一个 decode step 的状态如何从新 K/V 流入 attention”。计划使用 fused RoPE + paged KV append CUDA extension；若范围过大，先完成独立 CUDA KV append。

### 4. 执行引擎层

- request add、active、finish、cancel 状态。
- 从 active requests 构建 batch、block tables 和 seq_lens。
- 每一步执行 append -> paged decode -> 状态更新。
- 不同 context 的请求进入和离开 batch。
- cache capacity 不足时的 admission/backpressure 行为。

这一层回答“多个动态请求如何共同驱动底层 kernel”，是项目从算子走向 AI Infra 原型的关键。

### 5. 端到端评测与可观测性

- kernel latency 与完整 decode-step latency 分开记录。
- tokens/s、active requests、batch size、context 分布。
- block 使用率、内部碎片、allocation/reuse 次数。
- request churn 下的 p50/p90/p99 step latency。
- OOM/admission failure、空 batch、request cancel 等错误路径。
- 所有结果绑定环境、commit、配置和随机种子。

这一层回答“系统在真实动态 workload 下是否稳定、可解释、可复现”。

## 明确不做什么

为了保证深度和完成度，`v0.1.0` 不实现：

- tokenizer、采样、logits processor 和完整 Transformer 模型。
- HTTP/RPC 服务、鉴权、流式返回等网络层。
- tensor parallel、pipeline parallel 或多机调度。
- prefix cache、swap、CPU offload 和生产级抢占。
- 完整 prefill kernel 与生产级 continuous batching scheduler。
- 完整多 layer 模型执行；`v0.1.0` 的 DecodeEngine 固定验证单 layer 路径。

这些内容可以作为后续路线，但不应稀释当前单 GPU decode runtime 主线。

## 当前状态与缺口

| 层次 | 当前状态 | 主要缺口 |
| --- | --- | --- |
| Reference / Kernel | 已完成主要 correctness、参数实验与最终 profiling | 有边界的 `num_stages`/索引实验 |
| Paged KV Runtime | 已完成 allocate-on-append、block table、seq_len、capacity check | free/reuse、request lifecycle、指标 |
| Decode Data Path | PyTorch append 与 Triton decode 已分别存在 | CUDA RoPE/KV append 与统一 step |
| Execution Engine | 未实现 | request state、batch builder、admission、step orchestration |
| End-to-End Evaluation | 已有 kernel benchmark/profiler | 动态 workload、step latency、内存效率、p99 |
| Reproducibility | 已有环境和实验记录 | 一键运行、干净环境验证、release |

## v0.1.0 完成标准

只有满足以下条件，FlashDec 才算完成一个有足够深度的 AI Infra 项目：

1. kernel correctness 与最终配置有完整证据。
2. Paged KV Cache 支持 request add、append、finish/cancel、block free/reuse 和容量统计。
3. DecodeEngine 能对动态 active batch 执行 append -> paged decode，并正确更新状态。
4. 至少一条原生 CUDA 数据路径能构建、测试和 benchmark。
5. 动态 workload 能输出 step latency、tokens/s、block utilization、fragmentation 和 reuse 指标。
6. request churn、容量耗尽、释放后复用等状态机测试通过。
7. 新环境能够按文档复现 correctness 和 quick end-to-end benchmark。

单个 kernel 更快、参数 sweep 更多，不能单独满足上述完成标准。
