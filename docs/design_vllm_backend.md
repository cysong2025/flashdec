# vLLM Out-of-tree Attention Backend

## 目标与边界

R7 把 FlashDec 的 paged single-token decode attention 接入固定的 `vLLM==0.25.1`；R8 继续收窄并优化 Qwen2.5-3B 的 grouped-GQA split-KV 路径，用正式的长上下文固定批量生成验证至少 3% 的端到端收益。两阶段工作的共同问题是：在同一个真实模型、同一个 vLLM scheduler/KV cache/runtime 中，只替换 eligible decode attention，FlashDec 能否保持正确性并改善外部标准实现的性能。

FlashDec 不接管模型加载、prefill、KV cache 分配、continuous batching、sampling 或 HTTP API。它是一个 out-of-tree attention backend；vLLM 继续拥有完整 serving runtime。

```text
Qwen2.5-3B request
        │
        ▼
vLLM scheduler / model runner / KV cache
        │
        ├── prefill, mixed batch, unsupported feature
        │       └── vLLM Triton attention
        │
        └── uniform single-token decoder batch
                └── FlashDec grouped-GQA split-KV decode
```

## 注册与选择

`pyproject.toml` 通过 `vllm.general_plugins` 注册 `flashdec.vllm_plugin:register`。插件把 vLLM 的 `CUSTOM` registry slot 映射到 `FlashDecAttentionBackend`；该类继承 vLLM Triton backend 的 metadata、state 和 KV-cache contract。

运行时必须显式选择 backend：

```bash
export VLLM_PLUGINS=flashdec
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

vllm serve /home/<user>/models/Qwen2.5-3B-Instruct \
  --attention-backend CUSTOM \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16
```

后两个环境变量是本项目已验证 WSL/RTX 5070 环境的固定协议：前者避免 FlashInfer sampler 对 SM120/CUDA 组合的独立探测问题，后者启用 vLLM 的 WSL pin-memory 路径。它们不是 FlashDec kernel API 的通用要求。

## Fast path eligibility

`FlashDecAttentionImpl` 只替换同时满足下列条件的调用：

- decoder self-attention；
- uniform single-token decode，`max_query_len == 1`；
- causal、非 cascade；
- FP16/BF16 query 与同 dtype 的 5-D paged KV cache；
- head dimension 128，block size 16/32；
- query heads 可被 KV heads 整除，且每个 KV head 对应 4/8/16 个 query heads；
- 无 ALiBi、attention sinks、sliding window、logit soft cap、output scaling 或 decode LSE 返回要求。

任何条件不满足时都调用原生 `TritonAttentionImpl.forward()`。即使 shape 合法，最终只能选择一个 split，或者 vLLM 的 persistent workspace 容量不足时，也回退原生 Triton。因此 prefill、mixed prefill/decode、非 decoder attention 和尚未实现的 feature 不会进入一个近似兼容路径。`max_seq_len < 512` 是对当前 metadata 的额外 guard；CUDA Graph 模式在 capture 时按 bucket metadata 决策，不能把它解释为“所有实际 prompt 小于 512 的 replay 都必然回退”。

## KV layout 与零拷贝视图

vLLM 0.25.1 的 KV cache 是 `[block, 2, token, kv_head, dim]`，其中第二维选择 K/V plane。backend 把这个 5-D tensor 原样传给 FlashDec；kernel 通过显式 stride 直接寻址 K/V，不做 `permute()`、物理 layout conversion 或 hot-path copy。block table、sequence lengths 和 workspace 均复用 vLLM metadata。

`FlashDecAttentionBackend.forward_includes_kv_cache_update=False` 保留 vLLM 官方的独立 `unified_kv_cache_update` 和 torch.compile 显式数据依赖。当前 FlashDec eligible kernel 会再次写入同一个当前 token 的 K/V；这是冗余的同值 store，但不会绕过 vLLM 的图级更新顺序合同。fallback 则完全沿用原生 Triton 的更新与 attention 路径。

## Grouped GQA 与 split-KV

Qwen2.5-3B 使用 16 个 query heads、2 个 KV heads和 head dimension 128。FlashDec 每个 Triton program 处理共享同一 KV head 的 8 个 query heads，使 K/V tile 只加载一次，再分别维护每个 query head 的 online-softmax state。

高并发、长上下文使用两阶段 split-KV：

1. 第一阶段把 logical blocks 分成多个 segment，写入局部 max、exp-sum 和 FP32 accumulator。
2. 第二阶段用 log-sum-exp 规则合并 segment，并写入最终 FP16/BF16 output。

workspace 直接复用 vLLM `TritonAttentionMetadata` 的 persistent softmax buffers，不在 decode hot path 分配张量。自动策略从 `1/2/4/8/16` 中选择最接近约 128 个并行 programs 的 2 的幂：

```text
argmin(s ∈ {1,2,4,8,16}) |num_requests * num_kv_heads * s - 128|
```

最终 split 数限制在 `[1, 16]` 和 logical-block 数以内；eligibility 检查要求当次 metadata 的 `max_seq_len >= 512`。在 CUDA Graph 模式中，这个值可能来自 capture bucket 或 `max_model_len`，而不是 replay 时的实际 prompt 长度。`FLASHDEC_VLLM_NUM_SPLITS=1/2/4/8/16` 可用于有界实验，`0` 或未设置表示自动选择。Qwen2.5-3B 的正式 B8 路径为 16 个 query heads、2 个 KV heads、group size 8、head dimension 128；此时自动策略选择 8 splits，第一阶段为 `8 requests × 2 KV heads × 8 splits = 128` 个 grouped-GQA programs。

## 正式 split 激活证明

R8 的 fixed-batch runner 为每一个 `CUSTOM` worker 生成唯一的绝对 marker 路径和 nonce。只有 multi-split launcher 成功返回，并且当前 stream 正在 CUDA Graph capture 时，backend 才以 `O_EXCL`（并在平台支持时加 `O_NOFOLLOW`）创建权限 `0600` 的 canonical JSON marker。marker 绑定 case、trial、dataset SHA-256、commit、engine PID、shape、dtype、logical blocks 和 split 数。worker/parent runner 会读盘校验 canonical bytes、SHA-256 和绑定，再把经验证的投影写入 CSV；summarizer 随后复核 CSV 中的 canonical JSON、SHA、路径唯一性、绑定与展平字段，不声称自行重新读取原始 marker 文件。

正式 4-trial 证据要求 8 个 `CUSTOM` marker 全部唯一，且都记录 B8、Q16/KV2、D128、block16/32、BF16、8 splits 和 `cuda_graph_capture=true`。这项证明的严格边界是：对应 decode CUDA Graph 在 capture 阶段包含成功的 FlashDec custom split launch。它不等同于对每一次 measured graph replay 的 device-side 直接观测，不能据此宣称每次 replay 都单独产生了运行时证明。

## 正确性合同

正确性分三层验证：

1. PyTorch reference 对 FP16/BF16、不同 split 数和边界 shape 做逐元素比较。
2. vLLM attention microbenchmark 对每个正式 case 的完整 output 做 cross-backend tolerance check。
3. Qwen2.5-3B 固定 prompt/seed 的端到端生成验证第一步 greedy top-1 和共同前缀；完整 rollout 差异作为描述性结果保留。

不同并行 reduction 不保证 bitwise identical。一次接近并列的 greedy logits 扰动会改变后续自回归输入，因此完整 rollout identity 不替代逐元素 kernel accuracy，也不作为跨实现的唯一 pass/fail 条件。R7 的 32-token 诊断中，开启和关闭 split-KV 的 8 组完整 rollout 全部一致；R8 长上下文正式结果的最小共同前缀为 49 tokens、完整一致 7/8，且同一 backend 的 warmup/measured 完整输出 hash 在各 trial 内稳定。

## 性能结论

R7 在 RTX 5070 上的正式 Qwen attention shape matrix 中：

- B8/ctx1024 p50 为 vLLM 的 `0.8025x`，即 FlashDec latency 低约 19.75%；
- B8/ctx2048 p50 为 `0.7926x`，即低约 20.74%；
- B1/B4 case 保持在冻结的 1.05x 非回归边界内。

R7 证明的是外部 vLLM KV/metadata contract 内的 decode-attention kernel 优化。其完整 Qwen model 和在线 serving 只得到小幅正向观测，并分别未通过预注册的模型 target 和 throughput target；这个历史负结果仍然保留，不能把约 20% kernel 数字写成 20% model 或 serving speedup。

R8 commit `3ba68e3` 的 4-trial、balanced AB/BA 正式 fixed-batch 结果则通过了新的预注册门槛：

- target `qwen_b8_i8192_o4096` 的 paired latency ratio 为 `0.9542x [0.9530,0.9560]`，即 latency 降低 `4.58%`、output TPS 提升 `4.80%`，通过 `<= 0.970x` target；
- guard `qwen_b8_i512_o2` 为 `1.0029x [0.9890,1.0100]`，通过 `<= 1.05x` 非回归门槛；
- 8/8 `CUSTOM` workers 通过 capture-time split attestation。

这里的“端到端”是离线、固定 B8、blocking `LLM.generate` 的完整调用：包含模型执行、scheduler、KV cache、sampling 和 Python API 路径，排除进程启动、模型加载、JIT-prime 和 warmup。它不是在线请求负载，也没有给出 TTFT/TPOT 或 serving throughput 结论。权威数字、协议和证据边界见[性能报告](performance_report.md)、[R8 正式摘要](../benchmarks/results/vllm_qwen_long_context_model_latency_summary.md)和[结果索引](../benchmarks/results/README.md)。

## 已知限制

- 固定验证版本是 `vLLM==0.25.1`；vLLM backend registry 和 metadata 是版本化接口，升级后必须重新做集成测试。
- 只替换 uniform single-token decoder batch；prefill 和 mixed batch 仍由 vLLM Triton backend 执行。
- 当前证据限于单 RTX 5070、BF16 Qwen2.5-3B、单 GPU 和指定 batch/context。
- 不支持 tensor/pipeline parallel、distributed serving、quantized KV、speculative decoding 或非默认 attention feature 的 FlashDec fast path。
- out-of-tree backend 是研究集成，不把 FlashDec `0.0.0` 变成生产 serving 发行版。
