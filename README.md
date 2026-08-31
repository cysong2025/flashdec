<div align="center">

# ⚡ FlashDec

**A transactional, paged-decode runtime for single-GPU LLM inference**

从 Triton kernel、Paged KV Cache 到 vLLM/Qwen2.5-3B 端到端验证。

[![Repository checks](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml/badge.svg)](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8%20%7C%2013.0-76B900?logo=nvidia&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-3.6-654FF0)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[Quick start](#快速开始) · [Architecture](docs/design.md) · [Performance](docs/performance_report.md) · [API](docs/API.md) · [Reproduce](docs/reproducibility.md)

</div>

FlashDec 优化大模型逐 token 生成阶段的数据路径：它用事务化的 Paged KV Cache 管理请求状态，以 Triton/CUDA kernel 完成 RoPE、KV append 和 paged attention，并通过 out-of-tree 插件接入 vLLM。vLLM 继续负责模型、prefill、调度、sampling 与 serving；FlashDec 只替换满足条件的 single-token decode attention，并在不支持的场景自动回退。

| | 代表性结果 | 测量边界 |
| --- | ---: | --- |
| 🚀 Qwen2.5-3B 完整生成延迟 | **−4.58%** | B8 / input 8192 / output 4096，offline `LLM.generate` |
| ⚡ 输出吞吐 | **+4.80%** | 与上面相同的 4 轮 balanced AB/BA 实验 |
| 🧩 vLLM decode-attention p50 | **−19.75% / −20.74%** | B8 / context 1024、2048，kernel-only |
| 🧠 Shared-prefix KV 容量 | **−68.8%** | 固定 48-block pool，admission `9/16 → 16/16` |

> [!NOTE]
> 端到端结果来自 RTX 5070 上固定的 Qwen2.5-3B BF16 长上下文离线 workload，baseline 为显式 vLLM `TRITON_ATTN`，不代表所有模型、在线流量或 vLLM 默认最快配置。完整协议、范围和负结果见[性能报告](docs/performance_report.md)。

## 架构

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flashdec-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/flashdec-architecture-light.svg">
    <img src="docs/assets/flashdec-architecture-light.svg" width="100%" alt="FlashDec integration architecture from vLLM and the plugin router through the transactional runtime to Triton and CUDA kernels.">
  </picture>
</p>

| 层次 | FlashDec 提供的能力 |
| --- | --- |
| vLLM 集成 | `general_plugins` 注册、`CUSTOM` attention backend、严格 eligibility 检查和原生 Triton fallback |
| Decode runtime | block-aware scheduler、`DecodeEngine`、跨层 token transaction、失败回滚与 lifecycle validation |
| KV memory | physical block allocator、block table、shared-prefix refcount/LRU、request-private tail |
| GPU data path | fused RoPE + KV append、grouped-GQA split-KV paged attention、query-head reducer |
| Evidence | PyTorch reference、跨 backend parity、执行路径 attestation、commit-bound benchmark summaries |

### vLLM 是如何接入的？

FlashDec **没有维护修改版 vLLM，也没有接管整套推理引擎**。插件复用 vLLM 的模型执行和 KV layout，在 uniform single-token decode、支持的 dtype/head shape 等条件满足时选择 FlashDec kernel；prefill、mixed batch 或其他不支持路径继续使用 vLLM Triton backend。正式实验还会校验 CUDA Graph capture marker，避免“配置了插件但实际没有执行”的假阳性。

实现合同与 fallback 边界见 [vLLM backend design](docs/design_vllm_backend.md)。

## 性能概览

<p align="center">
  <a href="docs/assets/flashdec-results-overview-light.svg">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flashdec-results-overview-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/flashdec-results-overview-light.svg">
      <img src="docs/assets/flashdec-results-overview-light.svg" width="100%" alt="FlashDec results across Qwen end-to-end generation, vLLM attention, KV capacity and runtime mechanisms.">
    </picture>
  </a>
</p>

| 结果层次 | FlashDec 对比对象 | 结果 | 结论 |
| --- | --- | ---: | --- |
| 完整 Qwen `LLM.generate` | vLLM `TRITON_ATTN` | `0.9542x` latency | 长上下文目标通过；短路径 guard 为 `1.0029x` |
| Qwen attention kernel | vLLM Triton | `0.8025x / 0.7926x` p50 | B8 两个目标 shape 通过 |
| Cache-owned transaction | checked dispatch | `1.7307x` p50 speedup | 仅适用于 cache-owned metadata |
| Shared prefix | private context blocks | `68.8%` KV capacity saved | 容量收益，不宣称稳定 latency 收益 |
| Boundary scheduler | greedy step-only | `100% vs 0%` completion | correctness/progress，不是 latency 对比 |
| External kernel check | FlashInfer 0.6.15 | `1.2003x / 1.2284x` latency ratio | FlashInfer 更快；只代表共同 kernel shape |

图表由受版本控制的数据快照和 canonical summaries 确定性生成。不同层次的 ratio 方向和计时范围不能互相换算；原始结论见[结果索引](benchmarks/results/README.md)。

## 快速开始

推荐 Linux/WSL、Python 3.10+，以及与 PyTorch CUDA build 匹配的 CUDA Toolkit、NVCC 和 Ninja。

```bash
git clone https://github.com/cysong2025/flashdec.git
cd flashdec

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"

python scripts/check_env.py
python -m pytest -q -ra
```

先查看[兼容性矩阵](docs/compatibility.md)。FlashInfer 和 vLLM 使用隔离的固定环境，避免 pip 替换已经验证的 PyTorch/CUDA stack；对应安装和 Qwen 命令见[复现指南](docs/reproducibility.md)。

## API 示例

```python
import flashdec

output = flashdec.decode(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=q.shape[-1] ** -0.5,
    block_size=32,
)
```

事务化多层调用：

```python
tx = engine.begin_step(request_ids)
for layer_idx, (q, k, v) in enumerate(zip(q_by_layer, k_by_layer, v_by_layer)):
    engine.step_layer(tx, layer_idx, q, k, v)
result = engine.commit_step(tx)
```

任一 layer 失败都会 abort 整个 token；调用方必须丢弃此前返回的 layer outputs。完整张量约定和状态契约见 [API documentation](docs/API.md)。

## 文档

| 想了解什么 | 从这里开始 |
| --- | --- |
| 系统分层与数据流 | [Architecture](docs/design.md) |
| vLLM 插件和 fallback | [vLLM backend](docs/design_vllm_backend.md) |
| API 与状态语义 | [API reference](docs/API.md) |
| 性能数字和适用边界 | [Performance report](docs/performance_report.md) |
| 环境、测试与 benchmark | [Reproducibility](docs/reproducibility.md) |
| 全部设计和实验文档 | [Documentation index](docs/INDEX.md) |

## 项目边界

FlashDec 核心面向单 GPU、single-token decode、FP16/BF16 和 caller-provided Q/K/V。它不自行实现 tokenizer、完整 Transformer/prefill、sampling、HTTP/RPC 服务、生产级 continuous batching、TP/PP、多机或 swap/offload；可选 vLLM 插件复用其中一部分外部能力，但不把它们变成 FlashDec 自有实现。

欢迎提交 issue 或 pull request。贡献前请阅读[贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md)和[安全政策](SECURITY.md)。

## License

FlashDec 采用 [Apache License 2.0](LICENSE) 开源。
