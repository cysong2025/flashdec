# 中文学习资料导航

这份资料导航服务于 FlashDec 项目。原则是：**主学习路径全部用中文资料和中文笔记推进**。英文论文或官方英文页面只作为必要时核对术语和 API 的备用资料，不作为每天学习主线。

## 0. 使用方式

每看一份资料，都输出一页中文笔记。笔记不要抄原文，按下面结构写：

```markdown
# 资料标题

## 我学到了什么

## 和 FlashDec 的关系

## 关键概念

## 我还没懂的问题

## 下一步要写的代码或实验
```

学习资料必须和项目产出绑定。只看不写代码、只看不做实验，都不算完成。

## 1. 优先资料入口

### Triton

- Triton 中文文档：https://triton-lang.cn/main/index.html
- Triton 中文教程入口：https://triton-lang.cn/main/getting-started/tutorials/index.html
- Triton 中文 API：`triton.language`、`triton.testing`、Triton 语义。

重点看：

- 安装。
- vector add。
- fused softmax。
- matmul。
- fused attention。
- autotune。
- debugging。

目标：

- Week 1 能写 vector add / softmax / RMSNorm。
- Week 2 能写 matmul 和小范围 autotune。
- Week 4 起能写 attention kernel。

### PyTorch

- PyTorch 中文站：https://pytorch.ac.cn/
- PyTorch 中文教程：https://docs.pytorch.ac.cn/tutorials/
- PyTorch 自定义 C++/CUDA 算子中文教程：https://docs.pytorch.ac.cn/tutorials/advanced/cpp_custom_ops.html

重点看：

- Tensor 基础、shape、stride。
- CUDA tensor。
- `torch.cuda.Event` 计时。
- PyTorch profiler。
- 自定义 C++/CUDA 算子。

目标：

- Week 1 建立 correctness tests 和 benchmark。
- Week 3 写 attention reference。
- Week 10 写 CUDA extension。

### vLLM 与 PagedAttention

- vLLM 中文文档：https://docs.vllm.com.cn/
- vLLM Paged Attention 中文页面：https://docs.vllm.com.cn/en/latest/design/paged_attention/

重点看：

- Paged Attention 的输入。
- Query / Key / Value 的内存布局。
- QK 计算。
- Softmax 归约。
- block、thread block、warp、thread group 的区别。
- Paged KV Cache 与 block table 的思想。

目标：

- Week 5 能设计 `PagedKVCache`。
- Week 6 能实现 paged decode v1。
- Week 9 能写 profiling 报告。profiling 相关页面从 vLLM 中文首页搜索“性能分析”进入，避免直接依赖不稳定深链接。

### CUDA

中文资料优先级：

1. 中文书：《CUDA 编程：基础与实践》。
2. 中文书：《CUDA C 编程权威指南》。
3. NVIDIA 中文/中国站文档中关于 CUDA 编程模型、内存层次、性能优化的章节。
4. 高质量中文博客，仅作辅助，不把博客结论当最终事实。

重点看：

- grid / block / thread。
- warp。
- global memory。
- shared memory。
- register。
- coalesced memory access。
- occupancy。
- CUDA event。
- `nvcc` 与 PyTorch extension 构建。

目标：

- Week 2 能解释 GPU memory hierarchy。
- Week 8-9 能理解 profiling 指标。
- Week 10 能写一个小 CUDA extension。

### Attention / FlashAttention / Online Softmax

中文资料优先级：

1. Transformer attention 中文讲解。
2. FlashAttention 中文原理解析。
3. online softmax 中文推导。
4. vLLM Paged Attention 中文页面里的 Softmax 部分。

重点看：

- attention 公式。
- safe softmax。
- online softmax。
- 为什么不 materialize 完整 attention matrix。
- 为什么 FlashAttention 强调 IO-aware。

目标：

- Week 3 写 dense attention reference。
- Week 4 写 dense decode Triton kernel。
- Week 9 解释性能瓶颈。

## 2. 分阶段学习路线

### 第 0 阶段：环境与概念，Week 0

阅读：

- PyTorch 中文站安装页。
- Triton 中文安装页。
- vLLM 中文 Paged Attention 页面开头、输入、概念。

输出：

- `docs/environment.md`
- `docs/notes/triton_basics.md`
- 一张手画或 Markdown 图：dense KV vs paged KV。

必须弄懂：

- 什么是 decode。
- 什么是 KV cache。
- 为什么 decode 每次只产生一个 token。
- 为什么 KV cache 会随着上下文增长。

### 第 1 阶段：Triton 小算子，Week 1

阅读：

- Triton 中文 vector add。
- Triton 中文 softmax。
- Triton 中文 API：`tl.arange`、`tl.load`、`tl.store`、mask、program id。

输出：

- vector add kernel。
- softmax kernel。
- RMSNorm kernel。
- `docs/notes/triton_basics.md` 完整版。

必须弄懂：

- 一个 Triton program 对应什么计算块。
- mask load/store 为什么必要。
- stride 与 contiguous tensor 的关系。

### 第 2 阶段：矩阵乘与性能直觉，Week 2

阅读：

- Triton 中文 matmul。
- Triton 中文 autotune。
- CUDA 中文资料中的内存层次与线程层次。

输出：

- matmul kernel。
- shape sweep benchmark。
- `docs/notes/gpu_memory_basics.md`

必须弄懂：

- 为什么 matmul 要 tiling。
- 为什么 global memory 慢、shared/register 快。
- 为什么 coalesced load 重要。
- 为什么算子性能不能只看 FLOPS。

### 第 3 阶段：Attention 语义，Week 3

阅读：

- Transformer attention 中文讲解。
- FlashAttention 中文原理解析。
- vLLM Paged Attention：输入、QK、Softmax。

输出：

- PyTorch dense decode reference。
- correctness tests。
- `docs/design.md` 初稿。

必须弄懂：

- q/k/v shape。
- causal attention 在 decode 阶段如何简化。
- GQA/MQA 的 head 映射。
- safe softmax。

### 第 4 阶段：Dense Decode Kernel，Week 4

阅读：

- Triton 中文 fused attention 教程。
- online softmax 中文推导。

输出：

- dense decode Triton kernel。
- dense decode benchmark。
- `docs/notes/online_softmax.md`

必须弄懂：

- running max 如何更新。
- running sum 如何重标定。
- output accumulator 如何随 max 变化重标定。

### 第 5 阶段：Paged KV Cache，Week 5

阅读：

- vLLM 中文 Paged Attention 全文。
- 操作系统分页/虚拟内存中文资料。

输出：

- `PagedKVCache`。
- paged reference。
- `docs/design_paged_kv.md`

必须弄懂：

- logical block。
- physical block。
- block table。
- 为什么物理内存不连续也能表示连续上下文。

### 第 6 阶段：Paged Decode Kernel，Week 6-7

阅读：

- vLLM Paged Attention 的 Key、Value、QK、Softmax、LV、Output 部分。
- Triton 中文 debugging。

输出：

- paged decode v1。
- head_dim 64/128。
- GQA/MQA。
- BF16。
- `docs/compatibility.md`

必须弄懂：

- 如何从 logical token index 找到 physical block 和 block offset。
- 最后一个 block 如何 mask。
- 为什么不同 layout 会影响访存。

### 第 7 阶段：性能优化，Week 8-9

阅读：

- CUDA 中文性能优化资料。
- vLLM 中文性能分析资料。
- PyTorch profiler 中文资料。

输出：

- `docs/perf_experiments.md`
- `docs/performance_report.md`
- profiler 截图或摘要。

必须弄懂：

- latency/token。
- memory bandwidth。
- occupancy。
- kernel launch overhead。
- shape sweep 为什么重要。

### 第 8 阶段：CUDA Extension，Week 10

阅读：

- PyTorch 自定义 C++/CUDA 算子中文教程。
- CUDA 中文资料中的 kernel、memory、build 部分。

输出：

- CUDA extension。
- correctness tests。
- append benchmark。
- `docs/cuda_extension.md`

必须弄懂：

- PyTorch extension 如何注册 op。
- host code 与 device code 的区别。
- `nvcc` 在构建里做什么。

### 第 9 阶段：项目包装，Week 11-12

阅读：

- 优秀中文开源项目 README。
- 高性能算子 / AI Infra 项目复盘文章。

输出：

- 中文 README。
- 中文设计文档。
- 中文性能报告。
- 中文面试问答。
- 中英文简历 bullet。

必须弄懂：

- 面试官最关心的是“为什么这样设计”和“怎么证明有效”。
- benchmark 结论必须可复现。
- 不要夸张宣称超过工业库。

## 3. 每周固定输出

每周至少输出下面 3 类证据中的 2 类：

- 代码：kernel、reference、cache runtime、benchmark、tests。
- 数据：CSV、Markdown 表格、profiler 截图/摘要。
- 文档：中文笔记、设计文档、周复盘。

周复盘路径：

```text
docs/weekly/week_N_review.md
```

## 4. 资料质量判断标准

中文博客很多，质量不一。使用时按这个顺序判断：

1. 官方中文文档优先。
2. 有代码、有实验、有图示的文章优先。
3. 能和官方文档或源码互相印证的文章优先。
4. 只讲结论、不讲 shape、不讲代码、不讲实验的文章只作启发。

遇到中文资料互相矛盾时：

- 以官方文档和源码为准。
- 自己写最小实验验证。
- 在笔记里记录“我验证了什么”和“还没验证什么”。

## 5. 面试表达积累

每周把学到的内容压缩成 3 句话：

```markdown
## 本周面试表达

1. 我解决的问题是：
2. 我的实现选择是：
3. 我的实验结论是：
```

到 Week 12 时，这些句子就是简历和面试回答的原料。
