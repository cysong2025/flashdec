# FlashDec

[![quality](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml/badge.svg)](https://github.com/cysong2025/flashdec/actions/workflows/quality.yml)

FlashDec 是一个面向单 GPU LLM decode 的研究型运行时原型，覆盖 Paged KV Cache、Triton paged decode attention、CUDA RoPE/KV append、动态请求调度和多层 token 事务。

项目关注的核心问题是：在请求长度与 batch 持续变化的情况下，如何维护可验证的 KV 所有权和失败原子性，并证明底层 kernel 优化能够转化为完整 decode step 的系统收益。

> 当前状态：R1 Block-aware Scheduler、R2 Multi-layer KV Token Transaction、R3 Shared Prefix Blocks 与 R4-A Trusted CUDA Transaction Fast Path 均已完成。R4-A commit `4018449` 在 RTX 5070 通过 focused `73 passed, 23 subtests`、full `410 passed, 48 subtests` 和 160-row/80-pair 正式矩阵；Cache-owned trusted path 的 complete-token p50/TPS 几何平均为 `1.7307x/1.7131x`，16/16 个 dtype/case 分组的五轮 p50 range 均稳定胜出。R4-B persistent transaction metadata 在 commit `8047a9c` 完成 correctness 与 160-row/80-pair 正式矩阵，overall p50/TPS 为 `1.2493x/1.2392x`，但只有 13/16 分组五轮全部胜出，未通过预注册稳定性门，因此主线恢复 R4-A/materialized 默认。R4-C 组合 workload 的 reference trace、原子 multi-layer prompt prefill、runner 与 strict validator 已实现，等待 RTX 5070 correctness/24-row 正式验证。仓库继续保持 private 和 `0.0.0`。

## 架构

```text
Synthetic / Trace-driven Workload
                |
                v
       Block-aware Scheduler
  admission / fairness / commitment
                |
                v
           DecodeEngine
  request rows / token transaction
                |
        +-------+-------+
        |               |
        v               v
Paged KV Runtime    Observability
ownership/reuse     latency/events
        |
        v
Fused RoPE + KV Append (CUDA)
        |
        v
Paged Decode Attention (Triton)
```

Scheduler 只决定可进入本轮执行的 request ids；DecodeEngine 组织数据路径；Paged KV Runtime 独占 block ownership、事务与生命周期状态。Kernel 不维护请求状态，benchmark 也不拥有运行时对象。

## 核心能力

- **Reference 与 kernel**：PyTorch dense/paged reference；Triton dense/paged decode attention；支持 FP16/BF16、MHA/GQA/MQA、head dimension 64/128 和变长 context。
- **Paged KV Runtime**：physical block allocate/free/reuse、finish/cancel、容量预检、碎片与使用率指标、request churn invariant。
- **Shared Prefix R3**：immutable multi-layer full-block 注册与共享、active refcount、request-private tail、inactive LRU、saved blocks/bytes 指标，以及 shared residency 与 private commitment 分离的 scheduler admission。
- **CUDA 数据路径**：独立 CUDA KV append 与 fused RoPE + paged KV append，保留 PyTorch fallback。
- **DecodeEngine**：稳定 request-row 映射、动态 admission、显式 backpressure、append → paged decode 执行链。
- **Block-aware Scheduler**：lifetime block commitment、FIFO + aging、公平 runnable subset、stale decision 拒绝和 boundary-deadlock 对照实验。
- **Multi-layer transaction**：一个 token 只预留一次位置；各 layer 共用 block id/offset；全部成功后 seq_len 只增长一次；任一 layer 失败自动 rollback。
- **Trusted transaction R4**：公开 fused primitive 保留完整索引检查；Cache-owned transaction path 可跳过 allocator 已证明的 device-value reduction，避免每 layer 的重复 host/stream sync。
- **Integrated workload R4-C**：动态到达、fixed-prefix hit/private miss、multi-layer prompt、逐层 token transaction、rollback、finish/cancel 与 block reuse 使用同一 dependency-free reference trajectory 验证。
- **证据链**：固定 commit/seed/trial/shape/timing scope 的 benchmark、严格 summary validator、独立 profiler attribution 和 release gate。

## 结果摘要

所有性能数字均来自 NVIDIA GeForce RTX 5070，PyTorch `2.11.0+cu128`、CUDA Toolkit 12.8。Profiler 字段只用于归因，正式延迟来自 non-instrumented 路径。

| 层次 | 结果 | 结论 |
| --- | --- | --- |
| Paged decode kernel | token-major、`block_size=32`、`num_warps=2` | 固定配置来自完整 shape sweep |
| Fused append-only | p50 几何平均 `1.2226x` vs torch | fusion 减少 launch 与中间数据路径 |
| Dynamic DecodeEngine | p50/p90/TPS `1.0668x/1.0317x/1.0811x` | kernel 收益部分传递到完整 step |
| Scheduler R1 | boundary case 完成率：lifetime 100%，cancel 50%，greedy 0% | 默认策略保证容量安全与进展，不宣称无条件更快 |
| Multi-layer R2 | complete-token p50/p90/TPS `1.2101x/1.3826x/1.2800x` | 24 个 dtype/case 中 20 个三轮稳定胜出 |
| R2 最终 RTX 回归 | `337 passed, 25 subtests passed` | 无 skipped 或 failure |
| R3-B RTX 回归 | focused `56 passed, 14 subtests passed`；full `352 passed, 25 subtests passed` | shared-prefix Engine/scheduler 集成验收通过 |
| R3-D RTX 回归 | targeted `1 passed`；focused `61 passed, 8 subtests passed`；full `361 passed, 25 subtests passed` | hot-path lookup invariant 与完整功能回归通过 |
| R3 最终 confirmation | 75% hit：context 节省 `68.8%`/`5.5 MiB`，admission `9/16 -> 16/16` | 64 行通过；所有非零 complete/scheduler/Engine p50 range 均跨 1，无稳定性能方向 |
| R4-A RTX 回归 | focused `73 passed, 23 subtests passed`；full `410 passed, 48 subtests passed` | public checked 与 Cache-owned trusted 边界、parity、rollback 和完整功能回归通过 |
| R4-A 正式矩阵 | p50/p90/TPS `1.7307x/1.6751x/1.7131x`；append CPU `2.3612x` | 160 行、80 pairs、16/16 p50 ranges 稳定胜出；7/16 p99 ranges 跨 1，不声明稳定尾延迟 |
| R4-B 正式负结果 | p50/TPS/append CPU `1.2493x/1.2392x/3.0366x` | correctness 通过，但仅 13/16 p50 ranges 稳定胜出；keep gate 失败并恢复 materialized 默认 |
| R4-B 回滚边界回归 | focused `89 passed, 23 subtests passed`；full `410 passed, 48 subtests passed` | commit `36225d1` clean tree 与 release evidence gate 通过，确认 candidate 未残留 |

R2 的 decode device ratio 为 `1.0024x`，而 append device 与 CUDA event ratio 分别为 `1.6103x` 和 `1.9784x`，说明系统收益主要来自 append/launch 路径。每轮仅 20 repeats，p99 接近单轮最大值，因此所有尾延迟结论都保留场景范围，不作生产级稳定性声明。

详细结果见[性能报告](docs/performance_report.md)、[R1 正式摘要](benchmarks/results/r1_scheduler_workload_trials3_summary.md)、[R2 正式摘要](benchmarks/results/r2_multi_layer_engine_trials3_summary.md)、[R3 最终摘要](benchmarks/results/r3_shared_prefix_workload_trials8_summary.md)、[R4-A 正式摘要](benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)和[R4-B 正式负结果](benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)。R3 的 ownership、容量口径与离群点边界记录在[Shared Prefix Blocks 设计](docs/design_shared_prefix_blocks.md)，R4-C 预注册 trace 与 evidence schema 见[组合 workload 设计](docs/design_integrated_scheduled_multi_layer.md)。

## 快速开始

推荐 Linux/WSL、Python 3.10+。CUDA extension 需要与 PyTorch CUDA build 匹配的 Toolkit、NVCC 和 Ninja。

```bash
git clone https://github.com/cysong2025/flashdec.git
cd flashdec

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"

python scripts/check_env.py
```

CPU/reference 验证：

```bash
python -m pytest -q \
  tests/test_decode_reference.py \
  tests/test_paged_cache.py \
  tests/test_multi_layer_transaction.py \
  tests/test_shared_prefix_blocks.py \
  tests/test_scheduler.py
```

RTX/CUDA 验证：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python -m pytest -q -ra
```

完整的分层验证、正式 benchmark 和结果导出命令见[复现指南](docs/reproducibility.md)。

## API 示例

Paged decode public API：

```python
import flashdec

out = flashdec.decode(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    sm_scale=1.0 / head_dim**0.5,
    block_size=32,
)
```

Multi-layer token transaction：

```python
tx = engine.begin_step(request_ids)
for layer_idx, (q, k, v) in enumerate(zip(q_by_layer, k_by_layer, v_by_layer)):
    layer_result = engine.step_layer(tx, layer_idx, q, k, v)
result = engine.commit_step(tx)
```

`step_layer()` 的输入、写入或 decode 异常会自动 abort 整个 token；调用方应丢弃此前 layer output。

## 仓库结构

```text
flashdec/                 Python package、runtime、scheduler 与 kernels
flashdec/csrc/            C++/CUDA extension
tests/                    reference、kernel、runtime 与证据校验测试
benchmarks/               benchmark、profiler 与严格 summary 工具
benchmarks/results/       审核后的精简 Markdown 结果
docs/                     设计、性能、兼容性、复现与工程历史
scripts/                  环境检查、验证编排和 release gate
```

## 文档

- [文档索引](docs/INDEX.md)
- [公开 API](docs/API.md)
- [系统范围与分层](docs/AI_INFRA_SCOPE.md)
- [Paged KV Cache 设计](docs/design_paged_kv.md)
- [DecodeEngine 设计](docs/design_decode_engine.md)
- [Scheduler 设计](docs/design_scheduler.md)
- [Multi-layer transaction 设计](docs/design_multi_layer_kv_transaction.md)
- [Shared Prefix Blocks 设计](docs/design_shared_prefix_blocks.md)
- [性能报告](docs/performance_report.md)
- [兼容性矩阵](docs/compatibility.md)
- [复现指南](docs/reproducibility.md)
- [路线图](docs/ROADMAP.md)
- [贡献与本地验证](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## 范围与限制

- 单 GPU、每 request 每 step 一个 decode token。
- Q/K/V 由调用方提供；不执行完整 Transformer forward、tokenizer、sampling 或网络服务。
- Multi-layer API 是顺序 token transaction，不是完整模型执行器；multi-layer prompt prefill 尚未实现。
- R3 prefix cache 只接收调用方已经构建的 full blocks；R3-B 要求 prefix 覆盖完整 initial context，且 scheduler-managed request 开始前 resident set 已固定。尚不包含模型 prefill、content hashing、admission-time prefix eviction、swap/offload、生产级抢占、tensor/pipeline parallel 或多机执行。
- CUDA extension 使用 lazy JIT，首次运行包含构建成本。
- 公开结果只来自仓库代码、公开工具链与个人硬件；不包含任何第三方非公开实现或数据。

FlashDec 是研究型原型，不应被解释为生产 serving framework，也不与不同计时边界的工业系统做直接速度宣称。
