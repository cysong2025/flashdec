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

## Mac 开发与 RTX 5070 验证工作流

从 2026-07-11 起，后续学习、kernel 实现和 GPU 验证统一采用以下流程：

1. Mac 工作区负责代码修改、本地静态检查、文档整理和 Git 提交。
2. GitHub `origin/main` 是 Mac 与 Windows/WSL 之间唯一的代码同步中介。
3. Windows/WSL 工作区保持干净，通过 `git pull --ff-only` 获取待验证提交。
4. RTX 5070 负责 CUDA/Triton correctness、benchmark 和 profiling。
5. GPU 输出回传后，由 Mac 更新实验摘要和学习记录，再提交到 GitHub。

### Mac：提交前检查

```bash
git status --short
git diff --check
python3 -m compileall flashdec tests benchmarks
```

提交时必须显式列出工程文件，不使用 `git add .`。这样可以避免把本地生成物、临时文件或其他在途文件混入 kernel 提交。

```bash
git add <本次 kernel、test、benchmark、docs 文件>
git diff --cached --check
git diff --cached --stat
git commit -m "<本次工程目标>"
git push origin main
```

### Windows/WSL：拉取待验证提交

进入 RTX 5070 的 WSL 仓库后先检查本地状态：

```bash
cd ~/work/flashdec
git status --short
```

只有工作树干净时才拉取：

```bash
git pull --ff-only origin main
source .venv/bin/activate
python scripts/check_env.py
```

如果 `git status --short` 有输出，先确认这些修改的来源，不使用 `git reset --hard` 覆盖未保存工作。

### RTX 5070：验证顺序

每次验证按 correctness、quick benchmark、full benchmark 的顺序执行。correctness 未通过时不继续性能测试。

本轮 block size 验证命令：

```bash
python -m pytest -vv tests/test_paged_decode.py tests/test_public_api.py
python benchmarks/run_block_size_sweep.py --quick --output benchmarks/results/week8_paged_decode_block_size_quick.csv
python benchmarks/run_block_size_sweep.py --output benchmarks/results/week8_paged_decode_block_size.csv
```

上板后需要保留：

- 当前 Git commit id：`git rev-parse --short HEAD`。
- `python scripts/check_env.py` 的关键版本信息。
- pytest passed/failed 数量和耗时。
- benchmark 的完整命令、CSV 路径、p50/p90/mean、shape、dtype 和日期。
- 失败时的完整 traceback，不只保留最后一行。

`benchmarks/results/*.csv` 和 `*.log` 默认被 Git 忽略。公开仓库只提交精简后的 Markdown 摘要；不要把大体积 Chrome trace 直接提交到 Git。

当前 SSH 入口是 Windows OpenSSH，因此 Mac 不能直接调用 WSL 内的 `rsync`。结果回传使用 Windows 目录中转：

```bash
# WSL：复制结果到 Windows 用户目录
mkdir -p /mnt/c/Users/<windows-user>/flashdec_results
cp benchmarks/results/<result-files> /mnt/c/Users/<windows-user>/flashdec_results/

# Mac：通过 Windows OpenSSH 拉取
scp -r <windows-user>@<windows-host>:flashdec_results/. benchmarks/results/
```

如果后续配置了 WSL 自己的 SSH 端口，才直接使用 `rsync` 从 WSL 路径拉取。

### 工作流注意事项

- 不在 Mac 和 WSL 同时修改同一批工程文件。
- 不使用手工复制作为常规代码同步方式，避免验证到旧版本或不完整文件。
- GPU 结果必须绑定 commit id，确保数字与代码可追溯。
- 未在 RTX 5070 验证的结果一律标记为 pending，不写成已完成结论。
- GPU 验证完成后再决定默认 kernel config，不根据单次 quick 结果直接修改默认值。
