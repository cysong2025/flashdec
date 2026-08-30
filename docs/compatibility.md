# FlashDec Compatibility

本文说明 FlashDec `0.0.0` 研究原型覆盖的环境、shape、dtype 和运行时语义。表中的“支持”表示仓库包含实现与 correctness coverage，不等于稳定 API 或生产部署承诺。

## 已验证参考环境

| Component | Version / device |
| --- | --- |
| OS | WSL2 Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 5070, compute capability 12.0 |
| Python | 3.12.3 |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA | 12.8 |
| Triton | 3.6.0 |
| CUDA Toolkit / NVCC | 12.8 / 12.8.93 |
| Ninja | 1.13.0 |
| GCC/G++ | 13.3.0 |

Python 3.10 和 3.12 均进入仓库级 dependency-free checks。GPU 数值与性能证据来自上表环境；其他组合需要重新运行 correctness，不能直接继承性能结论。

R7 vLLM/Qwen 外部证据使用独立环境：Python 3.12.3、`vLLM==0.25.1`、PyTorch `2.11.0+cu130`、PyTorch CUDA 13.0 和 Triton 3.6.0，同一 RTX 5070。它不替换上表核心/CUDA-extension 的 cu128 环境；两组性能数字不能跨环境直接相除。

## Paged-decode kernel

| Capability | Supported range |
| --- | --- |
| Operation | single-token decode attention |
| Q shape | `[batch, num_q_heads, head_dim]` |
| KV physical layout | token-major `[page, kv_head, token, dim]` |
| Page table | `[batch, max_pages_per_sequence]` |
| Page size | 8, 16, 32；默认 32 |
| Head dimension | 64, 128 |
| Input/output dtype | FP16, BF16 |
| Attention mapping | MHA, GQA, MQA |
| Sequence lengths | per-row variable length，包含 zero length |
| Physical pages | 允许非连续 pages |
| Triton launch default | `num_warps=2`, implicit `num_stages` |

限制：

- 不支持 Triton FP32 paged decode 或任意 head dimension。
- dim-major layout 只用于受控实验与 correctness 对照，不是 runtime 默认。
- 没有自动 block-size/layout/staging autotune；默认值来自固定矩阵，详见[kernel experiments](kernel_experiments.md)。
- 当前 kernel 不包含 attention 内部 RoPE、ALiBi、sliding window 或 speculative decode。

## PagedKVCache

`PagedKVCache` 支持：

- request-scoped logical page list 与 physical page pool；
- append、finish、cancel、free/reuse 和 capacity preflight；
- padded block tables、committed `seq_len`、utilization/fragmentation metrics；
- capacity failure 原子性和 allocator invariant validation；
- multi-layer token transaction：shared location、sequential layer writes、single commit、batch abort；
- caller-provided immutable full-page shared prefixes、active refcount、private tail、inactive LRU；
- Cache-owned trusted transaction provenance。

限制：

- finished/cancelled request id 不能重新激活。
- legacy `append()`、RoPE helper 和 `DecodeEngine.step()` 只适用于单层；多层使用 `begin_step()` / `step_layer()` / `commit_step()`。
- shared prefix 必须在 request submission 前由调用方注册并覆盖完整 initial context。
- 不包含 prefix content hashing、模型 prefill、swap/offload 或多进程 cache ownership。

## RoPE 与 KV append

| Path | Dtype | Scope |
| --- | --- | --- |
| PyTorch reference | FP16/BF16/FP32 | split-half RoPE + paged write |
| Native CUDA append | FP16/BF16/FP32 | location-only K/V write；RoPE 在 PyTorch |
| Fused CUDA | FP16/BF16/FP32 | split-half RoPE + paged K/V write |

共同约束：token-major contiguous cache，偶数 `rotary_dim`，position 来自 append 前的 request length。当前不实现 RoPE scaling、YaRN、NTK-aware scaling 或 interleaved-pair convention。

原生 extension 使用 lazy JIT，要求：

- NVIDIA GPU 与 CUDA-compatible PyTorch；
- 与 PyTorch CUDA build 匹配的 Toolkit/NVCC；
- Ninja 与可用的 host compiler。

## DecodeEngine 与 Scheduler

DecodeEngine 支持 waiting/active/finished/cancelled lifecycle、deterministic row mapping、explicit backpressure、torch/native/fused append path、reference/Triton decode path，以及 multi-layer token transaction。

Block-aware Scheduler 支持 lifetime block commitment、FIFO + aging、bounded runnable subset、deferred requests，以及 policy/snapshot-bound decision validation。Decision 携带 request ids、原始 K/V-free metadata snapshot 与 config；Scheduler 不持有 K/V tensor 或 physical pages。`apply_scheduler_decision()` 必须显式接收生成 decision 的 scheduler 与 snapshot，旧的单参数调用不兼容。

核心 runtime 不支持：

- 完整 Transformer forward、prefill 或 logits/sampling；
- HTTP/RPC serving 与 streaming；
- priority API、生产级抢占、continuous batching service loop；
- tensor/pipeline parallel、多 GPU 或多机；
- CUDA Graph capture contract。

可选 vLLM plugin 复用 vLLM 的完整模型、prefill、continuous batching、sampling、HTTP streaming 和 CUDA Graph；这些是外部 runtime 能力，不是 FlashDec 核心实现。FlashDec fast path 仍只覆盖符合 [vLLM backend 合同](design_vllm_backend.md)的 single-token decode。

## vLLM out-of-tree backend

| Capability | 已验证范围 |
| --- | --- |
| vLLM | exactly `0.25.1` |
| Model | Qwen2.5-3B-Instruct BF16 |
| Attention shape | 16 query heads / 2 KV heads / head dimension 128 |
| KV layout | vLLM NHD cache 的零拷贝 strided view |
| Decode | uniform single-token decoder batch |
| CUDA Graph | vLLM default PIECEWISE/FULL capture |
| Fallback | prefill、mixed batch 和 unsupported features 回退 vLLM Triton |

限制：

- backend registry 与 metadata contract 绑定 vLLM 0.25.1；升级必须重新验证。
- FlashDec fast path 不支持 quantized KV、ALiBi、sinks、sliding window、logit soft cap、decode LSE、output scale 或非 decoder attention。
- TP/PP、多机、speculative decoding 和其他模型没有正式证据。
- WSL 证据协议设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 与 `VLLM_USE_FLASHINFER_SAMPLER=0`；它们不表示核心 kernel 依赖 FlashInfer sampler。

## FlashInfer baseline environment

FlashInfer 是可选实验依赖，不进入核心 runtime。可复现 baseline 使用：

| Package / setting | Required value |
| --- | --- |
| `flashinfer-python` | `0.6.15.post1` |
| Python | 3.12 |
| PyTorch | `2.11.0+cu128` |
| Triton | 3.6.0 |
| CUDA Toolkit | 12.8.1 |
| `FLASHINFER_CUDA_ARCH_LIST` | `12.0a` |
| Constraints | [`constraints/flashinfer-cu128.txt`](../constraints/flashinfer-cu128.txt) |

共同范围是 FP16/BF16、GQA 32/8 heads、head dimension 128、page size 32 的 batch paged decode。FlashDec token-major tensor 与 FlashInfer `HND` 共享逻辑数据，不在计时区间 permute/copy。FlashInfer planning/JIT/workspace lifecycle、FlashDec scheduler/transaction 和完整 serving 都不在比较范围内。方法与结果见[baseline design](design_flashinfer_baseline.md)和[canonical summary](../benchmarks/results/flashinfer_paged_decode_baseline_summary.md)。

## 安装与版本边界

- PyPI 默认解析出的 PyTorch/CUDA 组合不一定适配本机；必要时先安装匹配的 PyTorch，再用 `--no-deps` 安装 FlashDec extras。
- `baseline` extra 必须放在独立 virtualenv，避免解析器替换既有 cu128 Torch/CUDA stack。
- `0.0.0` 是研究原型版本；兼容性以本页矩阵和实际 correctness 复核为准，不承诺稳定 API 或通用二进制 wheel。
- 完整安装、环境探针和分层测试命令见[复现指南](reproducibility.md)。
