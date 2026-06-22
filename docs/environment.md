# 环境记录

这里记录 FlashDec 使用过的开发环境。

当前 Codex 工作区是 macOS 环境，没有 NVIDIA CUDA。真正的 RTX 5070 开发板环境需要你在开发板上单独运行 `scripts/check_env.py` 后补充。

## 当前 Codex 工作区

### 机器

- 日期：2026-06-17
- OS：macOS-26.2-arm64-arm-64bit
- GPU：当前工作区未检测到 NVIDIA GPU
- GPU 显存：不适用
- Driver：不适用

### 软件

- Python：3.9.6
- PyTorch：当前工作区未安装
- CUDA available：false
- PyTorch CUDA runtime：不适用
- Triton：当前工作区未安装
- NVCC：not found

### 原始输出

```text
FlashDec environment check
==========================
Python: 3.9.6
Platform: macOS-26.2-arm64-arm-64bit
PyTorch: not available (ModuleNotFoundError: No module named 'torch')
Triton: not available (ModuleNotFoundError: No module named 'triton')

torch import failed: ModuleNotFoundError: No module named 'torch'

nvidia-smi
----------
not found

nvcc --version
--------------
not found
```

### 备注

- 当前工作区不能运行 CUDA/Triton benchmark。
- 最终版本应以 RTX 5070 开发板环境为准。

## RTX 5070 开发板

在开发板上运行：

```bash
python scripts/check_env.py
```

然后填写：

- 日期：
- OS：
- GPU：
- GPU 显存：
- Driver：
- Python：
- PyTorch：
- PyTorch CUDA：
- Triton：
- NVCC：

### 原始输出

```text
粘贴开发板上的输出。
```

### 兼容性结论

- 是否能 import torch：
- 是否能 import triton：
- 是否能运行最小 CUDA tensor：
- 是否需要调整 PyTorch / CUDA / Triton 版本：
