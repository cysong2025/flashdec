<div align="center">

# ⚡ FlashDec

**从 PagedAttention kernel 到可验证的单 GPU LLM Decode Runtime**

*A correctness-first, evidence-driven decode runtime built with PyTorch, Triton and CUDA.*

[![Repository checks](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml/badge.svg)](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-validated%2012.8-76B900?logo=nvidia&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-validated%203.6-654FF0)
![Milestone](https://img.shields.io/badge/R1--R5-complete-2EA44F)
![Stage](https://img.shields.io/badge/stage-private%200.0.0-6E7781)

</div>

> [!IMPORTANT]
> R1–R5 的研究与工程交付均已完成，默认实现和 canonical evidence 已冻结。仓库当前仍是 private `0.0.0` development candidate；新环境复现、版本升级、公开设置与 tag 尚未启动。完整边界见[交付状态](docs/DELIVERY_STATUS.md)。

## 🧭 项目概览

FlashDec 研究在请求长度、batch 和 KV 容量持续变化时，如何把 **Paged KV ownership、动态调度、多层 token 事务和 GPU kernel** 组织为一条可验证的数据路径，并量化 kernel 优化能否传递到完整 decode step。

| | |
| --- | --- |
| 🎯 **范围** | 单 GPU、每个 request 每 step 一个 decode token |
| 🧩 **核心栈** | Python · PyTorch · Triton · C++/CUDA extension |
| 🧠 **正确性锚点** | PyTorch dense/paged reference + dependency-free state-machine reference |
| 🧱 **运行时状态** | Paged KV lifecycle、block ownership、transaction、shared prefix、scheduler |
| 🧪 **验证环境** | NVIDIA GeForce RTX 5070 · PyTorch `2.11.0+cu128` · CUDA 12.8 · Triton 3.6 |
| 📦 **当前阶段** | R1–R5 complete；release gate 暂停 |

### 为什么做 FlashDec？

| ✅ Correctness first | 🧱 Ownership first | ⚡ Kernel → System | 🔬 Evidence first |
| --- | --- | --- | --- |
| 先定义 reference，再实现 Triton/CUDA 路径 | Cache 独占 block、lifecycle 与事务状态 | 同时测 kernel、完整 step 和 host/runtime 开销 | 固定 commit、shape、seed、trial 与计时边界 |

## 🏗️ 架构

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flashdec-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/flashdec-architecture-light.svg">
    <img src="docs/assets/flashdec-architecture-light.svg" width="100%" alt="FlashDec architecture showing the caller, scheduler, DecodeEngine, transactional PagedKVCache, fused RoPE and KV append, paged decode attention, and the evidence boundary.">
  </picture>
</p>

Scheduler 只输出版本化容量决策；DecodeEngine 组织执行；PagedKVCache 是 block ownership、事务、`seq_len` 和 lifecycle 的唯一权威来源。RoPE/append 产生 rotated Q 并写入 K/V，paged attention 再读取 Cache-owned paged K/V；Kernel 不推进 request 状态，benchmark 也不拥有运行时对象。完整语义见[总体设计](docs/design.md)。

## ✨ R1–R5 交付矩阵

| 阶段 | 交付能力 | 正式证据 | 冻结结论 |
| --- | --- | --- | --- |
| **Core / R0** | dense/paged reference、Triton decode、Paged KV lifecycle、CUDA/fused append、动态 Engine | [Week 8–12 evidence](benchmarks/results/README.md) | token-major、block 32、2 warps、implicit stages、fused append |
| **R1 Scheduler** | lifetime commitment、FIFO + aging、公平 runnable subset、stale decision 拒绝 | [36-row matrix](benchmarks/results/r1_scheduler_workload_trials3_summary.md) | boundary case completion：lifetime 100%、cancel 50%、greedy 0% |
| **R2 Multi-layer** | 多层共享 token 位置、顺序写入、单次 seq_len commit、失败 rollback | [144-row matrix](benchmarks/results/r2_multi_layer_engine_trials3_summary.md) | torch/fused p50/p90 `1.2101x/1.3826x`；fused/torch TPS `1.2800x` |
| **R3 Shared Prefix** | immutable full-block 共享、refcount、private tail、inactive LRU、shared-aware admission | [64-row confirmation](benchmarks/results/r3_shared_prefix_workload_trials8_summary.md) | 75% hit 节省 `68.8%`/`5.5 MiB` KV-pool capacity，admission `9/16 → 16/16`；latency 无稳定方向 |
| **R4 Trusted / Integrated** | Cache-owned trusted validation、正式 candidate gate、统一 scheduled multi-layer trajectory | [R4-A](benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md) · [R4-B negative](benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md) · [R4-C](benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md) | R4-A checked/trusted p50 `1.7307x`、trusted/checked TPS `1.7131x`；R4-B 仅 13/16 稳定，已回滚；R4-C lifecycle 全通过 |
| **R5 External Baseline** | 固定 FlashInfer `0.6.15.post1` 的共同 paged-decode kernel-only 对比 | [72-row matrix](benchmarks/results/r5_flashinfer_paged_decode_trials3_summary.md) | FlashDec/FlashInfer p50 latency `1.2003x/1.2284x`，>1 有利 FlashInfer；16/16 p50 range 高于 1，绝对 p99 有 7/16 range 重叠 |

> [!NOTE]
> Ratio 的方向、绝对值、range 与不可比边界以各 strict summary 为准。R3 的稳定收益是 KV capacity/admission；R5 只比较共同 kernel scope。完整解释见[性能报告](docs/performance_report.md)。

## 🚀 快速开始

推荐 Linux/WSL、Python 3.10+。CUDA extension 需要与 PyTorch CUDA build 匹配的 Toolkit、NVCC 和 Ninja。

> [!CAUTION]
> 当前 fresh-environment release gate 已暂停。以下命令是开发环境入口，不代表已经验证的全新环境安装保证；private 仓库 clone 需要 GitHub 访问权限。

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

文档、证据与源码编译门：

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence
python -m compileall -q flashdec tests benchmarks scripts
```

GitHub CI 还运行选定的 dependency-free pytest；完整列表见 [repository checks workflow](.github/workflows/quality.yml)。

> [!WARNING]
> R5 FlashInfer 基线必须使用独立 Python 3.12 virtualenv、[`constraints/r5-cu128.txt`](constraints/r5-cu128.txt) 和 `FLASHINFER_CUDA_ARCH_LIST=12.0a`。不要在既有 cu128 环境中直接让 `baseline` extra 重新解析 Torch/CUDA。完整流程见[R5 复现章节](docs/reproducibility.md#r5-flashinfer-有限公开基线证据)。

## 🧩 API 速览

Paged decode：

```python
import flashdec

head_dim = q.shape[-1]
out = flashdec.decode(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=head_dim**-0.5,
    block_size=32,
)
```

Multi-layer token transaction：

```python
tx = engine.begin_step(request_ids)
for layer_idx, (q, k, v) in enumerate(zip(q_by_layer, k_by_layer, v_by_layer)):
    engine.step_layer(tx, layer_idx, q, k, v)
result = engine.commit_step(tx)
```

任一 `step_layer()` 失败都会 abort 整个 token；调用方必须丢弃此前 layer outputs。完整接口见[公开 API](docs/API.md)和[DecodeEngine 设计](docs/design_decode_engine.md)。

## 🗂️ 仓库地图

| 路径 | 内容 |
| --- | --- |
| [`flashdec/`](flashdec) | Python package、Paged KV runtime、scheduler、workload 与 Triton kernels |
| [`flashdec/csrc/`](flashdec/csrc) | C++/CUDA append extensions |
| [`tests/`](tests) | reference、kernel、runtime、失败路径与 evidence-validator tests |
| [`benchmarks/`](benchmarks) | benchmark runners、profilers 与 strict summaries |
| [`benchmarks/results/`](benchmarks/results) | 审核后提交的 canonical Markdown evidence |
| [`docs/`](docs) | 设计、性能、兼容性、复现、路线图与阶段历史 |
| [`scripts/`](scripts) | 环境检查、文档检查、验证编排与 release gate |

## 📚 文档导航

| 从这里开始 | 深入设计 | 证据与复现 |
| --- | --- | --- |
| [交付状态](docs/DELIVERY_STATUS.md) | [Paged KV Cache](docs/design_paged_kv.md) | [结果索引](benchmarks/results/README.md) |
| [公开 API](docs/API.md) | [DecodeEngine](docs/design_decode_engine.md) | [性能报告](docs/performance_report.md) |
| [系统范围](docs/AI_INFRA_SCOPE.md) | [Scheduler](docs/design_scheduler.md) | [复现指南](docs/reproducibility.md) |
| [文档索引](docs/INDEX.md) | [Multi-layer Transaction](docs/design_multi_layer_kv_transaction.md) | [兼容性矩阵](docs/compatibility.md) |
| [当前状态与下一步](docs/NEXT_STEPS.md) | [Shared Prefix](docs/design_shared_prefix_blocks.md) | [阶段日志](docs/weekly/README.md) |

## 🎯 范围边界

### 当前包含

- 单 GPU single-token decode、FP16/BF16、MHA/GQA/MQA。
- caller-provided Q/K/V 与 caller-built immutable shared-prefix blocks。
- caller-provided multi-layer context K/V 的事务性导入。
- PyTorch reference、Triton decode、CUDA/fused append 与 strict benchmark evidence。

### 当前不包含

- 完整 Transformer/prefill forward、tokenizer、sampling 或 logits processor。
- HTTP/RPC serving、生产级 continuous batching 或抢占。
- TP/PP、多机、swap/offload 或自动 prefix content hashing/admission-time online prefix eviction。
- 与不同 shape、layout、scheduler 或计时边界的工业 serving 系统做直接 speedup 声明。

FlashDec 是研究型 runtime 原型，不应被解释为生产 serving framework。贡献前请阅读[贡献指南](CONTRIBUTING.md)；当前维护范围限于 correctness、回归、文档与证据可追溯性改进。
