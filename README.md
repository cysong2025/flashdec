# FlashDec

FlashDec 是一个 12 周 AI Infra 工程项目，主题是 **LLM decode 阶段的 PagedAttention 与 Paged KV Cache 高性能算子**。

这个项目的目标不是做一个完整推理服务框架，而是完成一个小而深、公开可复现的 GPU 算子工程：

- 用 PyTorch 写清楚、可靠的 reference 实现。
- 用 Triton 实现 dense decode attention 和 paged decode attention。
- 实现一个简单的 Paged KV Cache 运行时，支持 block table。
- 建立 correctness、benchmark、profiling 三件套。
- 补一个小型 CUDA extension，展示底层算子经验。

## 当前状态

Week 1-3 已在 RTX 5070 上完成 correctness 与 benchmark 记录。Week 4 dense decode Triton kernel 已在 RTX 5070 上通过 correctness，并完成默认 benchmark。Week 5 Paged KV Cache runtime 与 paged PyTorch reference 已在 RTX 5070 上通过 correctness。Week 6 paged decode Triton kernel v1 已在 RTX 5070 上通过 correctness，并完成第一版 benchmark。Week 7 head_dim 128、BF16、GQA/MQA correctness 已在 RTX 5070 上通过，并完成 batch/context shape sweep。Week 8 已完成 `num_warps`、block size 和 KV layout 实验，当前通用配置为 `token-major + block_size=32 + num_warps=2`。Week 9 已完成最终默认配置的 FP16/BF16 四场景 profiling，correctness 为 `76 passed in 4.49s`；Nsight 硬件计数因当前环境缺少 `ncu`/`nsys` 暂未补充。

主要文档：

- [12 周详细执行计划](docs/PROJECT_PLAN.md)
- [接下来工作计划](docs/NEXT_STEPS.md)
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
- [性能实验记录](docs/perf_experiments.md)
- [性能报告](docs/performance_report.md)
- [Paged KV Cache 设计说明](docs/design_paged_kv.md)
- [兼容性记录](docs/compatibility.md)

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
    block_size=16,
)
```

## 计划里程碑

- Week 0-2：环境、Triton 基础、小算子、测试与 benchmark 框架。
- Week 3-4：PyTorch reference 与 dense decode Triton kernel。
- Week 5-7：Paged KV Cache 与 paged decode Triton kernel。
- Week 8-9：性能优化、profiling、对比实验。
- Week 10：小型 CUDA extension。
- Week 11-12：README、设计文档、benchmark 报告、发布与复现验证。

## 中文资料入口

学习主线优先使用中文材料：

- Triton 中文文档：https://triton-lang.cn/main/index.html
- PyTorch 中文站：https://pytorch.ac.cn/
- PyTorch 自定义 C++/CUDA 算子中文教程：https://docs.pytorch.ac.cn/tutorials/advanced/cpp_custom_ops.html
- vLLM 中文文档：https://docs.vllm.com.cn/
- vLLM Paged Attention 中文页面：https://docs.vllm.com.cn/en/latest/design/paged_attention/

更详细的阅读顺序见 [中文学习资料导航](docs/CHINESE_RESOURCES.md)。
