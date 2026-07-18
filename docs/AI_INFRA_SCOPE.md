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

这一层回答“单个 GPU kernel 如何正确并高效地计算”。当前配置已经冻结；除 correctness 或明确性能回归外，不再重复参数 sweep。

### 2. KV 内存管理层

- physical block pool。
- request 到 logical block list 的映射。
- block allocation、capacity check、free 与 reuse。
- request finish/cancel 后回收 block。
- 使用率、剩余容量、内部碎片和回收次数统计。

这一层回答“动态请求到达和结束时，KV Cache 如何管理显存”。PagedKVCache v2 已完成 free/reuse、finish/cancel、容量原子性、metrics 和 request churn 验证。

### 3. Decode 数据路径

- 新 token 的 RoPE。
- K/V 写入 physical block。
- block table 与 seq_len 更新。
- append 与 paged attention 的接口衔接。
- PyTorch fallback 与 CUDA extension 路径。

这一层回答“一个 decode step 的状态如何从新 K/V 流入 attention”。当前实现同时保留 PyTorch reference、独立 CUDA append 和 fused RoPE + paged KV append；GPU Engine 默认使用 fused 路径。

当前 PyTorch RoPE + paged KV append reference 和统一返回接口已通过 RTX 5070 correctness。独立 CUDA KV append 的 JIT extension、raw op 与 runtime integration 也已通过 RTX 5070 验证（focused `34 passed in 3.59s`，full `198 passed in 5.13s`）；RoPE 的 `append_backend="cuda"` integration 已通过（focused `56 passed in 3.85s`，full `204 passed in 4.47s`）；fused RoPE + KV append 已通过（focused `66 passed in 44.35s`，full `214 passed in 4.52s`）。三路径 full CUDA-event benchmark 已完成，fused p50 几何平均为 1.2226x vs torch。

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
- 完整多 layer 模型 forward；当前只实现调用方提供逐层 Q/K/V 的顺序 token transaction。

这些内容可以作为后续路线，但不应稀释当前单 GPU decode runtime 主线。

## 当前状态

| 层次 | 当前状态 | 证据边界 |
| --- | --- | --- |
| Reference / Kernel | correctness、shape sweep、profiling 与默认配置冻结已完成 | 仅在明确回归时重新进入参数实验 |
| Paged KV Runtime | lifecycle、free/reuse、capacity atomicity、metrics、churn 与 multi-layer transaction 已完成 | Cache 是 block ownership 与事务状态的唯一来源 |
| Decode Data Path | torch/CUDA/fused 三条路径通过 RTX correctness；fused append-only p50 为 `1.2226x` | native kernel 不修改 allocator 或 seq_len |
| Execution Engine | dynamic batch、backpressure、Scheduler R1 与 multi-layer R2 已完成 | 不包含模型 forward、sampling 或网络层 |
| End-to-End Evaluation | 36-row Engine、36-row Scheduler、144-row Multi-layer 与 64-row Shared Prefix confirmation 完成 | profiler 只做归因；p99 保留范围 |
| Reproducibility | 环境检查、分层验证、严格 summary 与 release checker 已完成 | clean-machine install、版本与 tag 留在最终发布阶段 |

## Release candidate 完成标准

只有满足以下条件，FlashDec 才算完成一个有足够深度的 AI Infra 项目：

1. kernel correctness 与最终配置有完整证据。
2. Paged KV Cache 支持 request add、append、finish/cancel、block free/reuse 和容量统计。
3. DecodeEngine 能对动态 active batch 执行 append -> paged decode，并正确更新状态。
4. 至少一条原生 CUDA 数据路径能构建、测试和 benchmark。
5. 动态 workload 能输出 step latency、tokens/s、block utilization、fragmentation 和 reuse 指标。
6. request churn、容量耗尽、释放后复用等状态机测试通过。
7. 新环境能够按文档复现 correctness 和 quick end-to-end benchmark。

单个 kernel 更快、参数 sweep 更多，不能单独满足上述完成标准。

## 选择性扩展边界

Block-aware Scheduler、multi-layer KV token transaction 与 shared prefix blocks 均已完成。当前没有自动启动下一条功能主线；仓库保持 private，FlashInfer/vLLM 有限公开对比与 release 工作都等待所有者明确决定。完整优先级与验收门槛见 `docs/ROADMAP.md`。
