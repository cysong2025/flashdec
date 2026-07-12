# Week 11 状态记录

## 本周主题

RoPE + paged KV append 数据路径，以及后续 CUDA extension。

## 当前已完成

- 新增 `flashdec/rope.py`。
- 实现 split-half `apply_rope()` PyTorch reference。
- RoPE 使用 FP32 计算，支持 FP16/BF16/FP32 输出和 partial `rotary_dim`。
- 新增 `PagedKVCache.next_positions()`，明确 pre-append position 语义。
- 实现 `rope_paged_kv_append_ref()`：rotated Q/K、raw V append 和统一 metadata 返回。
- 新增 `RopeAppendResult` 与公开 `flashdec` API。
- 新增手算公式、norm/tail、runtime dtype、block boundary、capacity atomicity 和 terminal request 测试。

## 当前环境限制

Codex macOS 环境没有 torch/pytest/CUDA，只能执行 AST、compileall 和静态检查。RTX 5070 WSL 已成功完成 native extension 首次 JIT build；公开 API 修复后仍需复跑 focused/full correctness。所有性能结论仍为 pending。

## 需要在 RTX 5070 完成

```bash
cd ~/work/flashdec
git pull origin main
source .venv/bin/activate

python -m pytest -vv \
  tests/test_rope_append.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py

python -m pytest -vv
```

验证结果：

```text
focused: 38 passed in 3.60s
full:    186 passed in 4.96s
```

## CUDA extension 前置检查

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

需要记录 `nvcc`、PyTorch CUDA build 和 `CUDA_HOME` 是否一致可用。若缺少 `nvcc`，先决定安装匹配 Toolkit，不能直接开始 extension 编译。

首次检查结果（安装 Toolkit 前）：

```text
nvcc: command not found
torch: 2.11.0+cu128
torch CUDA: 12.8
CUDA_HOME: None
```

已完成的 Toolkit 检查：

```text
nvcc: CUDA compilation tools, release 12.8, V12.8.93
PyTorch: 2.11.0+cu128
PyTorch CUDA: 12.8
CUDA_HOME: /usr/local/cuda-12.8
Ninja: 1.13.0
gcc/g++: 13.3.0
```

Toolkit 使用 WSL 的 toolkit-only 安装方式；不要在 WSL 安装 `cuda`、`cuda-12-x` 或 `cuda-drivers`，避免尝试安装 Linux GPU driver。

## 本次代码：独立 CUDA KV append

- 新增 lazy JIT extension：`flashdec/csrc/kv_append.cpp` 和 `flashdec/csrc/kv_append_kernel.cu`。
- 新增 `flashdec.cuda_kv_append()`：接收 token-major cache、physical `block_ids`/`block_offsets` 与一轮 K/V。
- 新增 `PagedKVCache.append_cuda()`：保留 Python allocator/lifecycle/capacity 语义，仅将 K/V slot 写入交给 CUDA。
- 新增 raw op、runtime 对齐、capacity atomicity、contiguity 错误路径测试。
- 首次 native path 只做 K/V copy，不融合 RoPE；这是为了先建立可解释的 reference/native 对照。

上板命令：

```bash
cd ~/work/flashdec
git pull --ff-only origin main
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export FLASHDEC_CUDA_VERBOSE=1

python -m pytest -vv \
  tests/test_cuda_kv_append.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py

unset FLASHDEC_CUDA_VERBOSE
python -m pytest -vv
```

第一次运行会在 PyTorch extension cache 中编译 CUDA 源码；后续同一环境、同一源码通常复用 build cache。需要记录 commit id、pytest 输出和任何 compiler traceback。

## 首次 RTX 5070 上板结果（待 API 修复后复跑）

首次 focused 命令共收集 34 项，用时 `44.20s`：

- 32 项通过：CUDA extension 已完成 JIT build；FP16/BF16/FP32 raw physical slot 写入、out-of-range metadata、`append_cuda()` 与 Python allocator/cache 内容对齐、capacity atomicity，以及 paged cache/decode 回归均通过。
- 2 项失败：`flashdec.cuda_kv_append` 被 Python 同名子模块覆盖成 module，导致公开函数 API 不是 callable。这是 Python package namespace 冲突，不是 CUDA 编译、kernel 映射或数值正确性问题。

修复策略：将内部实现从 `flashdec/cuda_kv_append.py` 改为 `flashdec/_cuda_kv_append.py`，继续以 `flashdec.cuda_kv_append(...)` 暴露函数。修复提交后必须复跑同一 focused 命令和完整回归，才能把 native correctness 标记为通过。

## 下一步

1. 在 RTX 5070 完成独立 CUDA KV append 的首次 JIT build、focused correctness 和 full regression。
2. 记录 FP16/BF16/FP32 的 raw slot 与 runtime 对齐结果。
3. 若正确性稳定，为 `rope_paged_kv_append_ref()` 增加可选 native append 路径或单独的 native API。
4. 实现 fused RoPE + KV append，再比较 PyTorch assignment、independent CUDA append 和 fused path 的 latency/launch 数。

官方参考：

- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
