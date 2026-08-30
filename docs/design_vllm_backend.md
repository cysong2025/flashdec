# vLLM Out-of-tree Attention Backend

## 目标与边界

R7 把 FlashDec 的 paged single-token decode attention 接入固定的 `vLLM==0.25.1`，用于回答一个更严格的问题：在同一个真实 Qwen2.5-3B 模型、同一个 vLLM scheduler/KV cache/API server 中，只替换 eligible decode attention，FlashDec 能否保持正确性并改善外部标准实现的性能。

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
- head dimension 64 或 128，block size 8/16/32；
- query heads 可被 KV heads 整除；
- 无 ALiBi、attention sinks、sliding window、logit soft cap、output scaling 或 decode LSE 返回要求。

任何条件不满足时都调用原生 `TritonAttentionImpl.forward()`。因此 prefill、mixed prefill/decode、非 decoder attention 和尚未实现的 feature 不会进入一个近似兼容路径。

## KV layout 与零拷贝视图

vLLM 0.25.1 的 NHD cache half 是 `[block, token, kv_head, dim]`。FlashDec kernel 接收 `[block, kv_head, token, dim]`；backend 通过 `permute()` 创建 stride view，不做物理 layout conversion，也不在 hot path 复制 K/V。block table、sequence lengths 和 workspace 均复用 vLLM metadata。

## Grouped GQA 与 split-KV

Qwen2.5-3B 使用 16 个 query heads、2 个 KV heads和 head dimension 128。FlashDec 每个 Triton program 处理共享同一 KV head 的 8 个 query heads，使 K/V tile 只加载一次，再分别维护每个 query head 的 online-softmax state。

高并发、长上下文使用两阶段 split-KV：

1. 第一阶段把 logical blocks 分成多个 segment，写入局部 max、exp-sum 和 FP32 accumulator。
2. 第二阶段用 log-sum-exp 规则合并 segment，并写入最终 FP16/BF16 output。

workspace 直接复用 vLLM `TritonAttentionMetadata` 的 persistent softmax buffers，不在 decode hot path 分配张量。自动策略以约 128 个并行 programs 为目标：

```text
ceil(128 / (num_requests * num_kv_heads))
```

最终 split 数限制在 `[1, 16]` 和 logical-block 数以内；context 小于 512 时禁用 split。`FLASHDEC_VLLM_NUM_SPLITS=1..16` 可用于有界实验，`0` 或未设置表示自动选择。

## 正确性合同

正确性分三层验证：

1. PyTorch reference 对 FP16/BF16、不同 split 数和边界 shape 做逐元素比较。
2. vLLM attention microbenchmark 对每个正式 case 的完整 output 做 cross-backend tolerance check。
3. Qwen2.5-3B 固定 prompt/seed 的端到端生成验证第一步 greedy top-1 为 8/8 一致；完整 32-token rollout 为 5/8 一致并作为描述性结果保留。

不同并行 reduction 不保证 bitwise identical。一次接近并列的 greedy logits 扰动会改变后续自回归输入，因此完整 rollout identity 不替代逐元素 kernel accuracy，也不作为跨实现的唯一 pass/fail 条件。开启和关闭 split-KV 的 8 组完整 rollout 在诊断实验中全部一致，说明 split 优化没有额外改变生成序列。

## 性能结论

RTX 5070 上的正式 Qwen shape matrix 中：

- B8/ctx1024 p50 为 vLLM 的 `0.8025x`，即 FlashDec latency 低约 19.75%；
- B8/ctx2048 p50 为 `0.7926x`，即低约 20.74%；
- B1/B4 case 保持在冻结的 1.05x 非回归边界内。

这证明的是外部 vLLM KV/metadata contract 内的 decode-attention kernel 优化。完整 Qwen model 和在线 serving 只得到小幅正向观测，并分别未通过预注册的模型 target 和 throughput target；不能把 20% kernel 数字写成 20% model 或 serving speedup。权威数字与失败门槛见[性能报告](performance_report.md)和[R7 结果摘要](../benchmarks/results/README.md#r7-vllm-qwen外部比较)。

## 已知限制

- 固定验证版本是 `vLLM==0.25.1`；vLLM backend registry 和 metadata 是版本化接口，升级后必须重新做集成测试。
- 只替换 uniform single-token decoder batch；prefill 和 mixed batch 仍由 vLLM Triton backend 执行。
- 当前证据限于单 RTX 5070、BF16 Qwen2.5-3B、单 GPU 和指定 batch/context。
- 不支持 tensor/pipeline parallel、distributed serving、quantized KV、speculative decoding 或非默认 attention feature 的 FlashDec fast path。
- out-of-tree backend 是研究集成，不把 FlashDec `0.0.0` 变成生产 serving 发行版。
