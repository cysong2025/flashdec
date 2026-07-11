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

Codex macOS 环境没有 torch/pytest/CUDA，只能执行 AST、compileall 和静态检查。RTX 5070 correctness 尚未执行，CUDA extension 尚未开始。

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

## CUDA extension 前置检查

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

需要记录 `nvcc`、PyTorch CUDA build 和 `CUDA_HOME` 是否一致可用。若缺少 `nvcc`，先决定安装匹配 Toolkit，不能直接开始 extension 编译。

## 下一步

1. 记录 RTX 5070 focused/full correctness。
2. 固定 CUDA op 输入输出与 `RopeAppendResult` 语义。
3. 先实现独立 CUDA KV append，对齐 rotated K/raw V 写入。
4. 正确性稳定后再融合 RoPE，并建立 separate vs fused benchmark。
