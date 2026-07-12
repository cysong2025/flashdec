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

Codex macOS 环境没有 torch/pytest/CUDA，只能执行 AST、compileall 和静态检查。RTX 5070 WSL 已成功完成三条 RoPE append 路径的 native extension JIT build、focused correctness 和完整回归；当前进入 CUDA-event benchmark。

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
- 新增 `rope_paged_kv_append(..., append_backend="torch" | "cuda")`；reference API 保持固定 PyTorch append，正式接口才可选择已验证的 CUDA K/V 写入。
- 新增 `append_backend="fused_cuda"` 和独立 fused extension：一次 kernel launch 输出 rotated Q、写入 rotated K 与 raw V。

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
  tests/test_rope_append.py \
  tests/test_cuda_kv_append.py \
  tests/test_fused_rope_kv_append.py \
  tests/test_paged_cache.py \
  tests/test_public_api.py

unset FLASHDEC_CUDA_VERBOSE
python -m pytest -vv
```

第一次运行会在 PyTorch extension cache 中编译 CUDA 源码；后续同一环境、同一源码通常复用 build cache。需要记录 commit id、pytest 输出和任何 compiler traceback。

## RTX 5070 上板结果

首次 focused 命令共收集 34 项，用时 `44.20s`：

- 32 项通过：CUDA extension 已完成 JIT build；FP16/BF16/FP32 raw physical slot 写入、out-of-range metadata、`append_cuda()` 与 Python allocator/cache 内容对齐、capacity atomicity，以及 paged cache/decode 回归均通过。
- 2 项失败：`flashdec.cuda_kv_append` 被 Python 同名子模块覆盖成 module，导致公开函数 API 不是 callable。这是 Python package namespace 冲突，不是 CUDA 编译、kernel 映射或数值正确性问题。

修复策略：将内部实现从 `flashdec/cuda_kv_append.py` 改为 `flashdec/_cuda_kv_append.py`，继续以 `flashdec.cuda_kv_append(...)` 暴露函数。

修复后的最终验证：

```text
focused: 34 passed in 3.59s
full:    198 passed in 5.13s
```

结论：独立 CUDA KV append 已在 RTX 5070 上完成 JIT 构建和 correctness。它与 Python allocator/reference 在测试覆盖的 FP16/BF16/FP32、跨 block 分配、physical slot 写入、capacity failure 和公开 API 语义上一致；这一结论不代表性能更快，性能比较仍待 benchmark。

RoPE 的 `torch`/`cuda` 双 backend 集成验证结果：

```text
focused: 56 passed in 3.85s
full:    204 passed in 4.47s
```

这确认普通 CUDA append 已正确接入 RoPE 数据路径；随后已完成 `fused_cuda` 的独立验证。

fused RoPE + KV append 验证结果：

```text
focused: 66 passed in 44.35s
full:    214 passed in 4.52s
```

`44.35s` 包含 fused extension 的首次 JIT 编译；完整回归复用 build cache。结论：`torch`、`cuda`、`fused_cuda` 三条路径均已在 RTX 5070 correctness 通过。仍不能由 correctness 时间推导性能，必须使用 CUDA event 单独测量。

## 三路径 benchmark（待执行）

本次不改变 RoPE 数学或 CUDA kernel，而是新增正式接口：

```python
flashdec.rope_paged_kv_append(..., append_backend="torch")
flashdec.rope_paged_kv_append(..., append_backend="cuda")
flashdec.rope_paged_kv_append(..., append_backend="fused_cuda")
```

`torch` 是默认值，和 `rope_paged_kv_append_ref()` 一致；`cuda` 仍用 PyTorch 计算 Q/K RoPE，只将 rotated K/raw V 写入切换到 `PagedKVCache.append_cuda()`；`fused_cuda` 用一个 native kernel 计算 rotated Q/K 并写入 K/V。新增测试要求 native 路径在 GQA、跨 block、FP16/BF16/FP32 下与 reference 对齐，并检查 unknown backend、CPU cache 和 fused capacity failure 时不发生 allocator mutation。

新增 `benchmarks/run_rope_kv_append_bench.py`，对如下完整 append GPU 路径计时：

```text
torch       : PyTorch RoPE + Python append
cuda        : PyTorch RoPE + independent CUDA append
fused_cuda  : fused CUDA RoPE + K/V append
```

每个 case 在计时前完成 cache prefill、extension preload 与 reference 对齐检查；CSV 的 CUDA-event latency 不包含 JIT build 和 prefill。它测量的是 GPU work，不包含 Python allocator 的 CPU wall-clock 开销。

首次 quick：

```bash
python benchmarks/run_rope_kv_append_bench.py \
  --quick \
  --dtype both \
  --output benchmarks/results/week11_rope_kv_append_quick.csv
```

quick 正常后执行完整三场景：

```bash
python benchmarks/run_rope_kv_append_bench.py \
  --dtype both \
  --output benchmarks/results/week11_rope_kv_append.csv
```

## 下一步

1. 在 RTX 5070 执行 quick/full 三路径 benchmark，并同步 CSV。
2. 根据 p50/p90、speedup、dtype 和 case 分析 fusion 是否真的减少 GPU work。
3. 再进入 DecodeEngine，测量含 Python allocator/scheduler 的 complete decode-step wall-clock latency。

官方参考：

- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
