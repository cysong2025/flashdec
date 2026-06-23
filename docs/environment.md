# 环境记录

这里记录 FlashDec 使用过的开发环境。

当前 Codex 工作区是 macOS 环境，没有 NVIDIA CUDA。RTX 5070 台式机环境通过 Windows 11 + WSL2 Ubuntu 24.04 运行 PyTorch/Triton。

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

## RTX 5070 台式机（WSL2 Ubuntu 24.04）

运行命令：

```bash
python scripts/check_env.py
```

### 机器

- 日期：2026-06-23
- OS：Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
- GPU：NVIDIA GeForce RTX 5070
- GPU 显存：11.94 GiB
- Driver：581.29

### 软件

- Python：3.12.3
- PyTorch：2.11.0+cu128
- PyTorch CUDA：12.8
- Triton：3.6.0
- CUDA available：true
- NVCC：not found

### 原始输出

```text
FlashDec environment check
==========================
Python: 3.12.3
Platform: Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
PyTorch: 2.11.0+cu128
Triton: 3.6.0

CUDA available: True
PyTorch CUDA: 12.8
CUDA device count: 1
GPU 0: NVIDIA GeForce RTX 5070, 11.94 GiB, sm_12.0

nvidia-smi
----------
Tue Jun 23 11:22:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.10              Driver Version: 581.29         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070        On  |   00000000:01:00.0  On |                  N/A |
|  0%   49C    P0             30W /  250W |    1003MiB /  12227MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

nvcc --version
--------------
not found
```

### 兼容性结论

- 是否能 import torch：是。
- 是否能 import triton：是。
- 是否能运行最小 CUDA tensor：是，PyTorch 可识别 RTX 5070。
- 是否需要调整 PyTorch / CUDA / Triton 版本：Week 1 不需要。`nvcc` 暂未安装，不影响 PyTorch/Triton 小算子验证；后续 CUDA extension 阶段需要补 CUDA Toolkit。
