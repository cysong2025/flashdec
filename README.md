# FlashDec

FlashDec 是一个 12 周 AI Infra 工程项目，主题是 **单 GPU LLM decode 执行、PagedAttention 与 Paged KV Cache 管理**。

这个项目的目标不是做一个完整推理服务框架，也不是只优化一个 GPU 算子，而是完成一个小而深、公开可复现的 decode runtime 原型：

- 用 PyTorch 写清楚、可靠的 reference 实现。
- 用 Triton 实现 dense decode attention 和 paged decode attention。
- 实现支持 block allocation、free/reuse 和请求生命周期的 Paged KV Cache runtime。
- 建立动态 batch 的 decode execution path，贯通 KV append、block table 和 paged attention。
- 用 CUDA extension 实现 RoPE/KV append 数据路径。
- 建立 kernel 与端到端两层 correctness、benchmark、profiling 和内存效率指标。

## 当前状态

Week 1-3 已在 RTX 5070 上完成 correctness 与 benchmark 记录。Week 4 dense decode Triton kernel 已在 RTX 5070 上通过 correctness，并完成默认 benchmark。Week 5 Paged KV Cache runtime 与 paged PyTorch reference 已在 RTX 5070 上通过 correctness。Week 6 paged decode Triton kernel v1 已在 RTX 5070 上通过 correctness，并完成第一版 benchmark。Week 7 head_dim 128、BF16、GQA/MQA correctness 已在 RTX 5070 上通过，并完成 batch/context shape sweep。Week 8 已完成 `num_warps`、block size 和 KV layout 实验。Week 9 已完成最终默认配置的 FP16/BF16 四场景 profiling。Week 10 冻结通用 kernel 配置并完成 Paged KV runtime v2。Week 11 的 fused append-only p50 几何平均为 `1.2226x`。Week 12 的首轮动态 workload 已完成 12/12 invariant 验证；fused complete-step p50/p90/tokens-s 几何平均为 `1.0537x/1.0588x/1.0674x`，但 p99 为 `0.9641x`，因此已进入多 trial 与交替 backend 顺序的稳定性验证。

主要文档：

- [12 周详细执行计划](docs/PROJECT_PLAN.md)
- [AI Infra 项目定位](docs/AI_INFRA_SCOPE.md)
- [接下来工作计划](docs/NEXT_STEPS.md)
- [v0.1-v0.3 深度路线图](docs/ROADMAP.md)
- [可复现安装与验证](docs/reproducibility.md)
- [版本变化记录](CHANGELOG.md)
- [准备清单](docs/PREP_CHECKLIST.md)
- [中文学习资料导航](docs/CHINESE_RESOURCES.md)
- [环境记录](docs/environment.md)
- [Week 1 状态记录](docs/weekly/week_1_status.md)
- [Week 2 状态记录](docs/weekly/week_2_status.md)
- [Week 3 状态记录](docs/weekly/week_3_status.md)
- [Week 4 状态记录](docs/weekly/week_4_status.md)
- [Week 5 状态记录](docs/weekly/week_5_status.md)
- [Week 6 状态记录](docs/weekly/week_6_status.md)
- [Week 7 状态记录](docs/weekly/week_7_status.md)
- [Week 8 状态记录](docs/weekly/week_8_status.md)
- [Week 9 状态记录](docs/weekly/week_9_status.md)
- [Week 10 状态记录](docs/weekly/week_10_status.md)
- [Week 11 状态记录](docs/weekly/week_11_status.md)
- [Week 12 状态记录](docs/weekly/week_12_status.md)
- [性能实验记录](docs/perf_experiments.md)
- [性能报告](docs/performance_report.md)
- [Paged KV Cache 设计说明](docs/design_paged_kv.md)
- [RoPE + KV Append 设计说明](docs/design_rope_kv_append.md)
- [Fused RoPE + Paged KV Append 设计说明](docs/design_fused_rope_kv_append.md)
- [DecodeEngine v1 设计说明](docs/design_decode_engine.md)
- [Block-aware Scheduler v2 设计说明](docs/design_scheduler.md)
- [Multi-layer KV Token Transaction 设计说明](docs/design_multi_layer_kv_transaction.md)
- [动态 Workload 设计说明](docs/design_dynamic_workload.md)
- [CUDA KV Append 设计说明](docs/design_cuda_kv_append.md)
- [兼容性记录](docs/compatibility.md)

## Quick start

推荐环境是 Linux/WSL。创建独立环境并安装开发/native extras：

```bash
git clone https://github.com/cysong2025/flashdec.git
cd flashdec

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"

python scripts/check_env.py
python scripts/check_release.py
```

CPU/reference focused 验证：

```bash
python -m pytest -q \
  tests/test_decode_reference.py \
  tests/test_paged_cache.py \
  tests/test_rope_append.py \
  tests/test_scheduler.py \
  tests/test_workload.py \
  tests/test_decode_engine_trial_summary.py \
  tests/test_profile_decode_engine.py
```

RTX 5070 native/Triton 验证前：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python -m pytest -vv \
  tests/test_cuda_kv_append.py \
  tests/test_fused_rope_kv_append.py \
  tests/test_paged_decode.py \
  tests/test_decode_engine.py \
  tests/test_public_api.py
```

完整的 clean-install、full regression、multi-trial 和 profiler 命令见 [reproducibility guide](docs/reproducibility.md)。

## 支持矩阵

| 层次 | 当前支持 | 当前默认/状态 |
| --- | --- | --- |
| dtype | FP16、BF16；部分 reference/native op 支持 FP32 | GPU decode 默认 FP16/BF16 |
| head dim | 64、128 | benchmark 主线 128 |
| attention heads | MHA、GQA、MQA | `num_q_heads % num_kv_heads == 0` |
| KV layout | token-major、dim-major correctness | token-major 默认 |
| block size | 8、16、32 | 32 默认 |
| Triton launch | num_warps 2/4/8、可选 num_stages | 2 warps、implicit stages |
| KV runtime | allocate/free/reuse、finish/cancel、backpressure | 单 GPU、单 layer |
| append backend | torch、独立 CUDA、fused CUDA | GPU Engine 显式 fused CUDA |
| scheduler | lifetime commitment/FIFO + aging 纯策略 planner | R1-A preview，尚未接入 Engine |
| workload | short-churn、mixed-steady、long-pressure | wall-clock + memory/lifecycle metrics |
| profiling | paged kernel、完整 Engine ranges、Chrome trace | instrumented 数据只做归因 |

## 当前限制

- `v0.1.0` 固定为单 GPU、单 layer、每 request 每 step 一个 decode token。
- Q/K/V 由调用方提供；不包含完整 Transformer forward、tokenizer、sampling 或网络服务。
- Block-aware Scheduler v2 的 R1-A 纯策略 planner 已实现但尚未接入 Engine；当前压力 workload baseline 仍会显式记录 backpressure/cancellation。
- Multi-layer KV Token Transaction 已冻结设计但尚未实现；当前 Cache/Engine 继续显式限制单 layer。
- 不包含 prefix cache、swap/offload、抢占、tensor/pipeline parallel 或多机。
- CUDA extension 使用 lazy JIT，需要匹配 PyTorch CUDA build 的 Toolkit、NVCC、host compiler 和 Ninja。
- macOS 工作区只能执行静态检查；正式 correctness/benchmark/profiling 以 RTX 5070 WSL 为准。
- 当前 package version 仍为 `0.0.0`；multi-trial、profiler 和 clean-install gate 完成前不创建 `v0.1.0` tag。

## 项目边界

这个仓库只包含公开、自写、可复现的内容。

不得包含：

- 公司内部源码。
- 公司内部 benchmark 数据。
- 公司芯片、编译器、运行时、内部 API 的非公开细节。
- 任何来自实习工作的保密材料。

公开 benchmark 只基于个人 RTX 5070 开发板或后续租借的公开云 GPU。

## 目标 API

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

## 计划里程碑

- Week 0-2：环境、Triton 基础、小算子、测试与 benchmark 框架。
- Week 3-4：PyTorch reference 与 dense decode Triton kernel。
- Week 5-7：Paged KV Cache 与 paged decode Triton kernel。
- Week 8-9：性能优化、profiling、对比实验。
- Week 10：冻结 kernel 配置并完成 Paged KV runtime 生命周期。
- Week 11：CUDA RoPE/KV append 与 decode execution engine。
- Week 12：动态 workload、端到端评测、发布与复现验证。

## 中文资料入口

学习主线优先使用中文材料：

- Triton 中文文档：https://triton-lang.cn/main/index.html
- PyTorch 中文站：https://pytorch.ac.cn/
- PyTorch 自定义 C++/CUDA 算子中文教程：https://docs.pytorch.ac.cn/tutorials/advanced/cpp_custom_ops.html
- vLLM 中文文档：https://docs.vllm.com.cn/
- vLLM Paged Attention 中文页面：https://docs.vllm.com.cn/en/latest/design/paged_attention/

更详细的阅读顺序见 [中文学习资料导航](docs/CHINESE_RESOURCES.md)。
