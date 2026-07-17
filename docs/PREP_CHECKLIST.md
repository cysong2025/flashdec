# FlashDec 开发与验证清单

本文用于新开发环境和正式实验前的快速检查。完整安装与 release 流程以[复现指南](reproducibility.md)为准。

## 1. 环境要求

- Python 3.10+。
- PyTorch 与 Triton。
- GPU 路径需要 NVIDIA GPU、与 PyTorch build 匹配的 CUDA Toolkit/NVCC。
- CUDA extension 构建需要 Ninja 和可用的 host compiler。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"
```

## 2. 环境检查

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python scripts/check_env.py
git status --short
git rev-parse --short HEAD
```

正式 GPU 结果必须记录 Python、PyTorch、Triton、CUDA、NVCC、GPU 和 Git commit。

## 3. 分层验证

Dependency-free 检查：

```bash
python -m unittest discover -s tests -p 'test_scheduler.py' -v
python -m unittest discover -s tests -p 'test_multi_layer_workload_*.py' -v
python -m compileall flashdec tests benchmarks scripts
git diff --check
```

CPU/reference：

```bash
python -m pytest -q \
  tests/test_decode_reference.py \
  tests/test_paged_cache.py \
  tests/test_multi_layer_transaction.py \
  tests/test_scheduler.py
```

GPU/full：

```bash
python -m pytest -q -ra
```

CUDA 可用的正式回归不应因缺少 `CUDA_HOME` 或 NVCC 跳过 native tests。使用 `-ra` 审核所有 skip reason。

## 4. Benchmark 前检查

- 工作树中的 tracked source 必须干净。
- warmup/repeat/trial/seed 固定并写入结果。
- torch/fused 对照使用同一 shape、输入轨迹和 backend-order 轮转。
- 输入生成、JIT、profiler 与错误路径 probe 不混入正式 latency。
- CSV/log 作为本地原始证据；公开仓库只提交审核后的 Markdown summary。

## 5. 提交前检查

```bash
git diff --check
python scripts/check_release.py --require-evidence
```

确认 README、设计文档、性能报告与 summary 使用同一组数字和边界描述。

## 6. 公开内容边界

- 只提交仓库自有代码、公开资料和个人实验结果。
- 不包含第三方非公开源码、数据、硬件行为或内部 API。
- 不把不同硬件、layout、计时范围或功能边界的结果放入同一 speedup 结论。
- 不把研究型原型描述为生产 serving framework。
