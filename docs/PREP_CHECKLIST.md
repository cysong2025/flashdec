# FlashDec 准备清单

这份清单用于 Week 0 和 Week 1 开始前的准备。

## 1. 公开边界

所有公开内容必须满足：

- [ ] 不使用公司内部源码。
- [ ] 不发布公司内部 benchmark 数据。
- [ ] 不描述公司芯片、编译器、运行时、私有 API 或未公开行为。
- [ ] 只使用公开资料、公开代码和个人实验。
- [ ] benchmark 只基于个人 RTX 5070 开发板或公开云 GPU。
- [ ] 所有文档默认中文撰写。

## 2. 环境验证

在 RTX 5070 开发板上运行：

```bash
python scripts/check_env.py
```

如果脚本不可运行，手动执行：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import triton; print(triton.__version__)"
nvcc --version
```

把输出记录到：

```text
docs/environment.md
```

## 3. 建议 Python 环境

初始建议：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch triton pytest pandas matplotlib tabulate
```

如果 PyTorch CUDA wheel 需要指定安装源，以 PyTorch 中文站给出的当前 CUDA 安装命令为准：

```text
https://pytorch.ac.cn/
```

环境能跑通后，把版本固定记录在 `docs/environment.md`。

## 4. 仓库结构

目标结构：

```text
flashdec/
  __init__.py
  reference.py
  cache.py
  kernels/
tests/
benchmarks/
  results/
  profiles/
docs/
  notes/
  weekly/
scripts/
```

## 5. 中文资料阅读顺序

先读：

1. [中文学习资料导航](CHINESE_RESOURCES.md)
2. Triton 中文文档：https://triton-lang.cn/main/index.html
3. PyTorch 中文站：https://pytorch.ac.cn/
4. vLLM 中文文档：https://docs.vllm.com.cn/
5. vLLM Paged Attention 中文页面：https://docs.vllm.com.cn/en/latest/design/paged_attention/
6. PyTorch 自定义 C++/CUDA 算子中文教程：https://docs.pytorch.ac.cn/tutorials/advanced/cpp_custom_ops.html

每份资料只抓和项目有关的内容。不要为了“学完一整套”而拖慢实现。

## 6. Week 0 要写的笔记

创建并填写：

- `docs/notes/triton_basics.md`
- `docs/notes/gpu_memory_basics.md`
- `docs/notes/attention_decode.md`
- `docs/notes/paged_kv_cache.md`

每篇笔记回答：

- 它解决什么问题？
- 关键 tensor shape 是什么？
- 性能瓶颈可能在哪里？
- 我应该 benchmark 什么？
- 我还有什么不懂？

## 7. 每周固定节奏

建议节奏：

- 周一或周二：确认本周交付物。
- 周中：实现最小正确版本。
- 周末：测试、benchmark、写笔记。
- 周日晚上：写 `docs/weekly/week_N_review.md`。

不要让一周只留下代码。每周至少留下：

- 测试结果。
- benchmark 结果。
- 设计笔记。
- profiling 笔记。

## 8. Week 1 完成标准

Week 1 只有在下面事项完成后才算结束：

- [ ] Triton vector add kernel 正确。
- [ ] Triton row-wise softmax kernel 正确。
- [ ] RMSNorm forward kernel 正确。
- [ ] CUDA event timing helper 可用。
- [ ] benchmark 能导出 CSV。
- [ ] 能解释 program id、block size、mask、stride、coalesced memory access。

## 9. 立即行动

1. 在 5070 开发板运行 `python scripts/check_env.py`。
2. 把输出补到 `docs/environment.md`。
3. 阅读 `docs/CHINESE_RESOURCES.md` 的第 0 阶段和第 1 阶段。
4. 写 `docs/notes/triton_basics.md`。
5. 开始实现 vector add reference/kernel/test。
