# FlashDec 12 周详细执行计划

> 开始日期：2026-06-17
>
> 目标完成日期：2026-09-13
>
> 每周投入：12-18 小时
>
> 项目定位：公开、可复现的单 GPU LLM decode 执行与 KV Cache 管理项目。

> Week 12 之后的 scheduler、multi-layer KV transaction、shared prefix 与公开基线路线见 `docs/ROADMAP.md`；本文件保留原始 12 周计划与验收基线。

## 1. 项目目标

FlashDec 的目标是围绕 **LLM decode 阶段的执行路径、PagedAttention 与 Paged KV Cache** 做一个小而深的 AI Infra 项目。高性能算子是底层核心，但最终交付必须覆盖内存管理、请求生命周期、动态 batch 执行和端到端评测。

它要证明五件事：

1. 理解 LLM decode 的计算、显存和动态请求瓶颈，而不是只实现零散 kernel。
2. 用 PyTorch + Triton + CUDA 实现并验证核心数据路径。
3. 实现 Paged KV block 的分配、释放、复用、容量管理和请求生命周期。
4. 用轻量 DecodeEngine 组织动态 active batch 与单步执行。
5. 把 kernel 指标、端到端 step latency、吞吐和内存效率做成工程闭环。

项目不追求完整 serving engine。核心范围是：

- 单 token decode attention。
- Paged KV Cache 的 block table 索引。
- physical block pool 的 allocate/free/reuse。
- request add/finish/cancel 生命周期。
- 变长 batch。
- 动态 active batch 的 decode-step orchestration。
- GQA / MQA。
- FP16 / BF16。
- kernel 与端到端 workload 在 RTX 5070 上的可复现 benchmark。

## 2. 最终成果

12 周结束时，仓库应包含：

- `flashdec` Python package。
- PyTorch reference 实现。
- Triton dense decode attention kernel。
- Triton paged decode attention kernel。
- `PagedKVCache` 运行时：分配/释放/复用 block、append KV、生成 block table、统计内存状态。
- `DecodeEngine`：管理 request 状态、构建 active batch、执行 append -> paged decode。
- 单元测试：覆盖主要 shape 和边界条件。
- benchmark 脚本：输出 kernel 与端到端 workload 的 CSV / Markdown 表格。
- profiling 报告：解释主要瓶颈和优化效果。
- 一个小型 CUDA extension，优先做 fused RoPE + KV append。
- 中文 README、中文设计文档、中文性能报告和兼容性说明。
- 可复现的安装、测试、benchmark 与 profiling 命令。

## 3. 成功标准

最低可交付版本：

- PyTorch reference 正确且可读。
- dense decode Triton kernel 与 paged decode Triton kernel 均能在 RTX 5070 上运行。
- Paged KV Cache 支持变长序列和非连续物理 block。
- correctness tests 覆盖 FP16/BF16、head_dim 64/128、batch 1-128、context 128-8192。
- benchmark 至少明显快于 naive PyTorch reference。
- README 能清楚解释 decode attention 为什么偏 memory-bound，以及 paged KV cache 解决什么问题。

完整工程版本：

- 支持 GQA/MQA。
- 支持 request finish/cancel、block free/reuse 和容量耗尽错误路径。
- 支持动态 active batch 的多步 decode execution。
- 性能报告包含 latency/token、有效内存带宽、shape sweep。
- 系统报告包含 step latency、tokens/s、block utilization、fragmentation 和 reuse。
- profiling 能解释至少 3 个有效优化和 1 个无效优化。
- 有一个能构建、能测试、能 benchmark 的 CUDA extension。
- 文档能从算法、kernel、内存管理、执行引擎和工程验证五层解释实现。

进阶版本：

- 能和 FlashInfer 或 vLLM 的公开实现做部分 shape 对比。
- 做一个简单 autotune：按 shape 选择 block size / num warps / num stages。
- 写一篇中文技术文章：《从零实现 Paged Decode Attention》。

## 4. 技术路线

### 为什么主线用 Triton

Triton 可以理解成一种面向 GPU kernel 的 Python 风格 DSL。它比手写 CUDA 更容易快速实现和迭代，又能暴露 block size、program id、mask load/store、num warps、autotune 等算子优化关键点。

本项目采用：

- Triton：主力实现 attention kernel。
- PyTorch：reference、测试、benchmark 驱动。
- CUDA：小范围实现一个 extension，展示底层能力。

### 核心 API 方向

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

缓存管理接口：

```python
cache = flashdec.PagedKVCache(
    num_layers=1,
    num_kv_heads=8,
    head_dim=128,
    block_size=32,
    max_blocks=4096,
    dtype=torch.float16,
    device="cuda",
)

block_tables = cache.append(layer_idx=0, request_ids=request_ids, k=k, v=v)
```

接口可以在实现中微调，但 README、tests、benchmark 应保持一致。

## 5. 里程碑

### 里程碑 A：基础能力跑通

截止：2026-07-05

成果：

- 项目骨架完整。
- 环境可用。
- 能写简单 Triton kernel。
- pytest 与 benchmark 框架可运行。

### 里程碑 B：Dense Decode Attention

截止：2026-07-19

成果：

- PyTorch reference 正确。
- dense decode Triton kernel 正确。
- 可以测 correctness 与 latency。

### 里程碑 C：Paged Decode Attention

截止：2026-08-09

成果：

- Paged KV Cache 运行时可用。
- block table 索引正确。
- paged decode Triton kernel 支持主要 shape。

### 里程碑 D：优化与 profiling

截止：2026-08-23

成果：

- 有性能实验记录。
- 有 profiling 报告。
- 默认 kernel 参数来自实测结果。

### 里程碑 E：公开发布

截止：2026-09-13

成果：

- Paged KV Runtime v2 支持 request lifecycle 和 block free/reuse。
- DecodeEngine 能运行单 layer 动态 active batch。
- synthetic workload 能报告 step latency、吞吐和内存效率。
- 中文 README、中文设计文档、中文性能报告完整。
- benchmark 可复现。
- 安装、测试、benchmark 和已知限制可以由新环境复现。

## 6. 每周计划

### Week 0：2026-06-17 至 2026-06-21

主题：仓库、环境、学习路径准备。

目标：

- 把项目从想法落到仓库。
- 明确公开边界，避免任何公司保密风险。
- 在 RTX 5070 开发板上验证 PyTorch / CUDA / Triton。
- 建立中文资料学习路线。

任务：

- 建立目录：
  - `flashdec/`
  - `tests/`
  - `benchmarks/`
  - `docs/`
  - `scripts/`
- 在 5070 开发板运行：
  - `python scripts/check_env.py`
  - `nvidia-smi`
  - `nvcc --version`
- 把环境输出记录到 `docs/environment.md`。
- 阅读中文资料导航的第 0 阶段和第 1 阶段。
- 写第一篇笔记：`docs/notes/triton_basics.md`。

交付物：

- `README.md`
- `docs/PROJECT_PLAN.md`
- `docs/PREP_CHECKLIST.md`
- `docs/CHINESE_RESOURCES.md`
- `docs/environment.md`

验收标准：

- 你能用 2 分钟说清楚 FlashDec 做什么、不做什么。
- 5070 开发板上能跑一个最小 PyTorch CUDA snippet。
- 确认 Triton 是否能在当前驱动和 Python 环境中正常 import。

### Week 1：2026-06-22 至 2026-06-28

主题：Triton 入门与测试/benchmark 框架。

目标：

- 掌握 Triton 最基本的编程模型。
- 建立以后所有 kernel 共用的 correctness 和 benchmark 方法。

中文学习输入：

- Triton 中文文档：安装、教程、`triton.language` 基本 API。
- PyTorch 中文教程：Tensor、CUDA tensor、基本 profiling/计时。

编码任务：

- 实现 3 个小 kernel：
  - vector add。
  - row-wise softmax。
  - RMSNorm forward。
- 为每个 kernel 写 PyTorch reference。
- 为每个 kernel 写 pytest correctness test。
- 实现 benchmark helper：
  - warmup。
  - CUDA event 计时。
  - p50 / p90 / mean latency。
  - CSV 输出。

交付物：

- `flashdec/kernels/vector_add.py`
- `flashdec/kernels/softmax.py`
- `flashdec/kernels/rmsnorm.py`
- `tests/test_triton_basics.py`
- `benchmarks/run_microbench.py`
- `docs/notes/triton_basics.md`

验收标准：

- `pytest` 通过。
- microbench 能输出 CSV。
- 你能解释：
  - `program_id`
  - block size
  - mask load/store
  - stride
  - contiguous/coalesced memory access

### Week 2：2026-06-29 至 2026-07-05

主题：matmul、访存、autotune、profiling 入门。

目标：

- 建立 GPU kernel 性能直觉。
- 知道什么时候算子是 compute-bound，什么时候是 memory-bound。

中文学习输入：

- Triton 中文 matmul 教程。
- Triton 中文 autotune 相关内容。
- CUDA 中文入门资料中关于 grid/block/thread、global/shared/register memory 的章节。

编码任务：

- 实现一个小型 FP16 matmul Triton kernel。
- 和 `torch.matmul` 对比 correctness。
- 做 M/N/K shape sweep。
- 加入小范围 Triton autotune。
- 跑一次 PyTorch profiler 或 Nsight。

笔记任务：

- 写 `docs/notes/gpu_memory_basics.md`，回答：
  - global memory、shared memory、register 分别是什么。
  - 为什么合并访存重要。
  - 为什么 cuBLAS 很难被手写 matmul 打败。
  - 为什么 decode attention 常常比 prefill 更偏内存带宽瓶颈。

交付物：

- matmul kernel。
- matmul benchmark CSV。
- 一份 profiler 截图或文本摘要。
- `docs/notes/gpu_memory_basics.md`

验收标准：

- 你能读懂一个简单 profiler timeline。
- 你能说明 kernel launch time 和 GPU execution time 的区别。

### Week 3：2026-07-06 至 2026-07-12

主题：attention reference 与 dense decode baseline。

目标：

- 先把 attention 语义定义清楚。
- 后续所有 Triton kernel 都能对齐这个 reference。

中文学习输入：

- Transformer attention 中文讲解。
- FlashAttention / online softmax 中文讲解。
- vLLM 中文 Paged Attention 页面中的输入、概念、QK、Softmax 部分。

编码任务：

- 实现 dense decode PyTorch reference：
  - `q`: `[num_seqs, num_q_heads, head_dim]`
  - `k/v`: `[num_seqs, max_seq_len, num_kv_heads, head_dim]`
  - `seq_lens`: `[num_seqs]`
  - `out`: `[num_seqs, num_q_heads, head_dim]`
- 支持 GQA 映射：
  - `kv_head = q_head // (num_q_heads // num_kv_heads)`
- 实现数值稳定 softmax。
- 写随机 shape tests。

交付物：

- `flashdec/reference.py`
- `tests/test_decode_reference.py`
- `benchmarks/run_decode_reference.py`
- `docs/design.md` 初稿。

验收标准：

- reference 代码清晰、朴素、可信。
- 可以一键生成随机 shape，与手写小例子对齐。

### Week 4：2026-07-13 至 2026-07-19

主题：dense decode attention Triton kernel。

目标：

- 写出第一个真正的 decode attention kernel。
- 掌握 online softmax 在 kernel 内部的写法。

编码任务：

- 实现 dense decode kernel：
  - 每个 program 处理一个 `(sequence, q_head)`。
  - 沿 context 维度分 block 遍历 K/V。
  - 使用 FP32 accumulation。
  - 实现 running max、running exp sum、running output accumulator。
- 先支持 head_dim 64，再支持 head_dim 128。
- 写 correctness tests。
- 和 naive PyTorch reference benchmark。

笔记任务：

- 写 `docs/notes/online_softmax.md`：
  - safe softmax。
  - online softmax。
  - 为什么 attention 不能真的 materialize 整个 attention matrix。

交付物：

- `flashdec/kernels/dense_decode.py`
- `tests/test_dense_decode.py`
- `benchmarks/run_dense_decode.py`
- `docs/notes/online_softmax.md`

验收标准：

- dense Triton decode 正确。
- 在中等 batch/context 上快于 naive PyTorch reference。
- 你能在白板上推导 online softmax 的更新公式。

### Week 5：2026-07-20 至 2026-07-26

主题：Paged KV Cache 数据结构。

目标：

- 先把 block table 和 cache 语义做清楚，再写 paged kernel。

中文学习输入：

- vLLM 中文 Paged Attention 页面。
- vLLM 中文文档中 KV cache、prefix cache、性能分析相关内容。
- 操作系统分页/虚拟内存的中文资料，用来辅助理解 block table。

编码任务：

- 实现 `PagedKVCache`：
  - 固定大小 physical block。
  - 每个 request 维护 logical block list。
  - append 一个 token。
  - 生成 padded block table tensor。
  - 维护 `seq_lens`。
- 实现基于 block table 的 PyTorch paged reference。
- 测试 dense KV 与 paged KV 输出一致。

交付物：

- `flashdec/cache.py`
- `flashdec/paged_reference.py`
- `tests/test_paged_cache.py`
- `docs/design_paged_kv.md`

验收标准：

- 你能画出 logical token index 到 physical block / offset 的映射。
- paged reference 与 dense reference 对齐。

### Week 6：2026-07-27 至 2026-08-02

主题：paged decode kernel v1。

目标：

- 写出第一个可正确运行的 paged decode Triton kernel。

编码任务：

- 把 dense decode 逻辑改成 block table 索引。
- 每个 `(sequence, q_head)`：
  - 读取 `seq_len`。
  - 遍历 logical block。
  - 查 block table 得到 physical block。
  - 从 physical KV cache 读取 K/V。
  - 对最后一个 block 做 mask。
- v1 限定：
  - `block_size = 16`
  - `head_dim = 64`
  - FP16
  - GQA 可以先不做。

交付物：

- `flashdec/kernels/paged_decode.py`
- `tests/test_paged_decode.py`
- 第一版 paged decode benchmark CSV。

验收标准：

- 变长序列 correctness 通过。
- 你能指出 v1 慢在哪里：访存、索引、launch overhead，还是 occupancy。

### Week 7：2026-08-03 至 2026-08-09

主题：真实 decode shape 补全。

目标：

- 让 paged decode 支持更贴近 LLM 的 shape。

编码任务：

- 支持 head_dim 128。
- 支持 BF16。
- 支持 GQA/MQA：
  - 例如 32 q heads / 8 kv heads。
  - 例如 16 q heads / 1 kv head。
- 做 batch sweep：
  - 1, 2, 4, 8, 16, 32, 64, 128。
- 做 context sweep：
  - 128, 256, 512, 1024, 2048, 4096, 8192。
- 明确记录暂不支持的 shape。

交付物：

- 更新后的 paged decode kernel。
- 更新后的 tests。
- `benchmarks/results/week7_paged_decode.csv`
- `docs/compatibility.md`

验收标准：

- 主 shape matrix correctness 通过。
- benchmark 不再只是 toy shape。

### Week 8：2026-08-10 至 2026-08-16

主题：优化第一轮：参数、布局、访存。

目标：

- 用实验而不是猜测来决定默认 kernel 配置。

实验任务：

- sweep：
  - block size：8/16/32。
  - `BLOCK_N`。
  - `num_warps`。
  - `num_stages`。
- 比较 KV cache layout：
  - `[num_blocks, num_kv_heads, block_size, head_dim]`
  - `[num_blocks, num_kv_heads, head_dim, block_size]`
  - vLLM 风格 layout 可作为参考。
- 优化 K/V load 的合并访存。
- 减少 benchmark 中 Python 侧开销。

记录任务：

- 建立 `docs/perf_experiments.md`：
  - 假设。
  - 改动。
  - 测试 shape。
  - 结果。
  - 结论。

交付物：

- `docs/perf_experiments.md`
- 更新后的 benchmark CSV。
- RTX 5070 默认 kernel config。

验收标准：

- 至少 3 个性能实验有清晰 before/after。
- 默认配置来自测量结果。

### Week 9：2026-08-17 至 2026-08-23

主题：profiling 与对比。

目标：

- 用可复现数据解释 kernel 的性能瓶颈。

实验任务：

- profile 三类场景：
  - 小 batch / 短 context。
  - 中 batch / 中 context。
  - 大 batch / 长 context。
- 记录：
  - kernel latency。
  - latency/token。
  - 有效内存带宽。
  - occupancy 或 achieved occupancy。
  - memory throughput。
- 对比：
  - naive PyTorch reference。
  - dense Triton baseline。
  - FlashInfer，如果安装顺利。
  - vLLM，至少做设计层面对比。

文档任务：

- 写 `docs/performance_report.md`：
  - 瓶颈是什么。
  - 哪些优化有效。
  - 哪些优化无效。
  - 下一步会怎么做。

交付物：

- `docs/performance_report.md`
- `benchmarks/results/week9_summary.md`
- profiler artifact，体积太大则只保留截图/摘要。

验收标准：

- 你能用数据解释性能。
- 报告里至少包含一个负结果，体现真实工程判断。

### Week 10：2026-08-24 至 2026-08-30

主题：冻结 kernel 配置与 Paged KV Runtime v2。

目标：

- 结束无边界参数调优，把项目重心转向请求生命周期和显存管理。

编码任务：

- 完成 `num_stages` 的 baseline/1/2/3/4 受控实验并冻结 kernel config。
- 实现 `finish_request()`、`cancel_request()`。
- 实现 physical block free/reuse。
- 增加 request state query 和 cache metrics。
- 保证批量 append 容量不足时不发生 partial mutation。
- 增加 request churn 与 block leak 测试。

交付物：

- `PagedKVCache` runtime v2。
- runtime state-machine tests。
- `num_stages` 实验摘要。
- 更新 `docs/design_paged_kv.md`。

验收标准：

- add/append/finish/cancel/reuse 状态机正确。
- request churn 后不存在 block leak。
- kernel 默认配置被代码、测试和文档共同固定。

### Week 11：2026-08-31 至 2026-09-06

主题：CUDA 数据路径与 DecodeEngine。

目标：

- 贯通新 token K/V 写入、Paged KV metadata 和 paged attention execution。

编码任务：

- 建立 PyTorch C++/CUDA extension。
- 先实现 CUDA KV append，再按进度融合 RoPE。
- 注册 Python op并保留 PyTorch fallback。
- 定义 request waiting/active/finished/cancelled 状态。
- 实现 active batch builder。
- 实现 append -> block table/seq_len -> paged decode 的单步执行。
- 覆盖请求在不同 step 加入、完成和取消。

文档任务：

- 写 `docs/cuda_extension.md`：
  - 为什么 attention kernel 用 Triton。
  - 为什么 append kernel 用 CUDA。
  - build 流程。
  - 主要瓶颈。
- 写 `docs/decode_engine.md`，记录 request、batch 和 cache 的所有权边界。

交付物：

- `flashdec/csrc/`
- `flashdec/cuda_ops.py`
- `tests/test_cuda_extension.py`
- `flashdec/engine.py`
- `tests/test_decode_engine.py`
- `docs/cuda_extension.md`
- `docs/decode_engine.md`

验收标准：

- extension 能构建和运行，CUDA 写入结果与 reference 对齐。
- DecodeEngine 多步动态 batch 输出与逐请求 reference 对齐。
- request finish/cancel 后 block 能被新请求复用。

### Week 12：2026-09-07 至 2026-09-13

主题：端到端 workload、发布与复现验证。

目标：

- 用动态 workload 验证系统行为，并固化为可安装、可测试、可 benchmark 的 `v0.1.0`。

任务：

- 实现 short/high-churn、mixed/steady、long/memory-pressure 三种 synthetic workload。
- 记录完整 decode-step p50/p90/p99、tokens/s、block utilization、fragmentation 和 reuse。
- 区分 kernel latency、runtime overhead 与 execution overhead。
- 清理 package API，完善安装说明和环境检查脚本。
- 完善 README 架构、quick start、benchmark 表、限制和 roadmap。
- 在干净环境复跑安装、correctness、quick benchmark。
- 固化最终结果表和已知限制，打 `v0.1.0` release tag。

交付物：

- `benchmarks/run_decode_workload.py`
- `docs/system_performance.md`
- 完整中文 `README.md`
- `CHANGELOG.md`
- `docs/reproducibility.md`
- GitHub release `v0.1.0`

验收标准：

- 新环境能按 README 跑通 correctness、动态 workload quick benchmark。
- kernel、runtime、engine 和 workload 四层边界清楚。
- 系统、算法、内存管理和性能结论都能追溯到代码和数据。

## 7. 每周时间分配

每周 12-18 小时建议这样分：

- 3-4 小时：中文资料阅读与笔记。
- 5-8 小时：编码。
- 2-3 小时：测试与 debug。
- 1-2 小时：benchmark / profiling。
- 1 小时：周总结和文档。

不要跳过文档时间。项目结论需要能追溯到代码、测试命令和 benchmark 数据。

## 8. 每周复盘模板

每周结束写一篇：

```markdown
# Week N 复盘

## 本周完成

## 正确性结果

## 性能结果

## Bug / 阻塞

## 下周调整
```

每周最好有一个 commit，包含：

- 代码。
- 测试。
- benchmark 或实验结果。
- 对应文档。

## 9. 风险与应对

### 风险：Triton 学习成本高

应对：

- Week 1-2 只做小 kernel。
- 先 dense decode，再 paged decode。
- 永远先 correctness，再 performance。

### 风险：RTX 5070 环境兼容问题

应对：

- Week 0 先验证环境。
- 能跑后立刻 pin 版本。
- PyTorch reference 与 CPU 侧测试独立存在。
- 必要时租一块常见云 GPU 做最终 benchmark。

### 风险：PagedAttention 太复杂

应对：

- 先写 paged reference。
- v1 限定 block_size 16、head_dim 64、FP16。
- v1 正确后再加 GQA、head_dim 128、BF16。

### 风险：性能比不过成熟库

应对：

- 主要对比 naive PyTorch 和自己的 dense baseline。
- FlashInfer/vLLM 用作设计参考和可选对比。
- 报告如实记录从零实现、正确性、benchmark、profiling 和优化过程，不夸张宣称超过工业库。

### 风险：系统范围膨胀

应对：

- `v0.1.0` 固定为单 GPU、单 layer、单 token decode execution。
- 优先做深 request lifecycle、block allocator、dynamic batch 和 workload 指标。
- 不实现网络服务、完整模型、多机并行和生产级 scheduler。
- 新功能必须对应 correctness、系统指标或端到端路径，否则不进入主线。

### 风险：公司保密边界

应对：

- 只使用公开中文资料、公开英文论文的中文笔记、公开代码和个人实验。
- 不发布公司内部代码、数据、硬件细节、性能结论。
- 所有公开 benchmark 明确标注个人 RTX 5070 或公开云 GPU。

## 10. 中文学习顺序

1. PyTorch tensor、stride、CUDA tensor、CUDA event 计时。
2. Triton vector add、mask load/store。
3. Triton softmax、reduction。
4. Triton matmul、autotune。
5. dense attention 与 online softmax。
6. KV cache、GQA、MQA。
7. Paged KV Cache、block table、allocator 与 request lifecycle。
8. dynamic batch、decode execution 和 backpressure。
9. profiling、memory bandwidth 与端到端 workload 指标。
10. PyTorch C++/CUDA custom operators、README 与复现说明。

具体链接见 `docs/CHINESE_RESOURCES.md`。

## 11. 最终完成定义

- 公共 API、reference、Triton kernel 和 Paged KV Cache 语义保持一致。
- FP16/BF16、MHA/GQA/MQA、变长 batch 与主要错误路径有 correctness 覆盖。
- 默认配置由完整 shape sweep 决定，并在文档中保留原始命令和结果摘要。
- Paged KV Runtime 支持 request finish/cancel、block free/reuse、容量与碎片指标。
- 单 layer DecodeEngine 支持动态 active batch，并通过逐请求 reference 对齐。
- CUDA extension 能构建、测试并与 PyTorch reference 对齐。
- synthetic workload 能报告完整 step latency、tokens/s 和内存效率。
- README、设计、性能、兼容性和复现文档互相一致。
