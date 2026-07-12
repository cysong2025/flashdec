# FlashDec Reproducibility Guide

## 目的与证据边界

本文定义第三方复现 FlashDec `v0.1.0` 候选版本的唯一流程。复现分为四层，不能用后一层的成功替代前一层：

1. environment/package：依赖、commit、CUDA Toolkit 和编译器可见。
2. correctness：reference、runtime state machine、native extension 和 Triton 对齐。
3. non-instrumented benchmark：CUDA-event 或完整 Engine wall-clock 性能。
4. instrumented profiling：阶段归因和 Chrome trace，不作为 release latency。

所有公开数字必须绑定 commit、命令、GPU、PyTorch/CUDA、shape、dtype、warmup/repeat/trial 和结果摘要。当前 macOS Codex 工作区没有 torch/pytest/CUDA；GPU 证据只来自个人 RTX 5070 WSL 环境。

## 已验证参考环境

```text
OS: WSL2 Ubuntu 24.04
GPU: NVIDIA GeForce RTX 5070, 11.94 GiB
Python: 3.12.3
PyTorch: 2.11.0+cu128
PyTorch CUDA: 12.8
Triton: 3.6.0
CUDA Toolkit/NVCC: 12.8 / 12.8.93
CUDA_HOME: /usr/local/cuda-12.8
Ninja: 1.13.0
GCC/G++: 13.3.0
```

驱动、`nvidia-smi` 和完整环境历史见 `docs/environment.md`。这里记录的是已验证参考环境，不表示项目只能在这些精确版本运行；其他组合必须重新执行 correctness，不得直接继承性能结论。

正式 release 前需要在 clean WSL venv 生成并审核 `pip freeze`；当前不提交未在新环境验证的伪 lock file。

## Fresh clone 与安装

```bash
git clone https://github.com/cysong2025/flashdec.git
cd flashdec
git rev-parse --short HEAD

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"
```

extras：

- 默认依赖：torch、triton。
- `dev`：pytest。
- `cuda-extension`：Ninja，用于 lazy JIT C++/CUDA extension。

若通用 PyPI 解析出的 torch 不是目标 CUDA build，应先按目标环境安装匹配的 PyTorch，再使用：

```bash
python -m pip install -e ".[dev,cuda-extension]" --no-deps
```

不要在 WSL 安装 Linux NVIDIA display driver。WSL 使用 Windows host driver；本项目只需要匹配 PyTorch CUDA build 的 toolkit/compiler 来构建 extension。

## 环境记录

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python scripts/check_env.py
git status --short
git rev-parse --short HEAD
```

`MAX_JOBS=1` 仅限制首次 extension JIT build 的并行编译资源，不是 CUDA kernel 的 block/warp 参数。

## R0 分阶段编排器

手工命令仍保留在后续章节作为唯一计时边界说明；实际执行时推荐由 dependency-free 编排器保证顺序、tracked-clean/CUDA 预检、产物存在性和 Windows 导出。

先确认完整命令，不执行 GPU 工作：

```bash
python scripts/run_r0_validation.py \
  --phase all \
  --dry-run \
  --export-dir /mnt/c/Users/user/flashdec_results
```

分阶段正式执行：

```bash
python scripts/run_r0_validation.py \
  --phase local \
  --phase focused \
  --phase full

python scripts/run_r0_validation.py \
  --phase trials-quick \
  --phase profile-quick \
  --export-dir /mnt/c/Users/user/flashdec_results

python scripts/run_r0_validation.py \
  --phase trials-formal \
  --phase profile-formal \
  --export-dir /mnt/c/Users/user/flashdec_results
```

`all` 等价于从 `local` 到 `profile-formal` 的全部 evidence phase，故意不包含 `release`。正式 summary 在 Mac 审核、提交并由 WSL 重新拉取后，才单独运行：

```bash
python scripts/run_r0_validation.py --phase release
```

`release` 同时启用 `--require-clean --require-evidence`，但不会修改版本或创建 tag。编排器只允许 `benchmarks/results/`、`benchmarks/profiles/` 下的 untracked result artifacts，并要求每一步实际更新目标文件；不允许 formal evidence 基于 tracked source diff 或 untracked source。`--allow-dirty` 仅供 non-formal 开发实验，formal phase 会拒绝该选项。若 WSL 留有即将被 Git 覆盖的正式 summary，先移动到已忽略的 `benchmarks/results/local_backups/`，再拉取 Mac 提交的版本。

## CPU/reference 验证层

CPU/reference suite 不需要 NVIDIA GPU 或 CUDA Toolkit，但 Linux 环境仍需安装项目依赖：

Scheduler R1-A 没有 torch/pytest 依赖，可先在任意 Python 3.10+ 环境独立执行：

```bash
python -m unittest discover -s tests -p 'test_scheduler.py' -v
python -m unittest discover -s tests -p 'test_benchmark_helpers.py' -v
python -m unittest discover -s tests -p 'test_profile_decode_engine.py' -v
python -m unittest discover -s tests -p 'test_r0_validation.py' -v
```

完整 CPU/reference suite：

```bash
python -m pytest -vv \
  tests/test_benchmark_helpers.py \
  tests/test_perf_metrics.py \
  tests/test_decode_reference.py \
  tests/test_paged_cache.py \
  tests/test_rope_append.py \
  tests/test_scheduler.py \
  tests/test_workload.py \
  tests/test_workload_benchmark.py \
  tests/test_decode_engine_trial_summary.py \
  tests/test_profile_decode_engine.py \
  tests/test_r0_validation.py \
  tests/test_release_check.py
```

CUDA-only cases应明确显示 skip，而不是 import/build failure。记录 passed/skipped/failed 数量和耗时。

## RTX 5070 focused correctness

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1

python -m pytest -vv \
  tests/test_cuda_kv_append.py \
  tests/test_fused_rope_kv_append.py \
  tests/test_rope_append.py \
  tests/test_paged_cache.py \
  tests/test_paged_decode.py \
  tests/test_decode_engine.py \
  tests/test_workload.py \
  tests/test_workload_benchmark.py \
  tests/test_decode_engine_trial_summary.py \
  tests/test_profile_decode_engine.py \
  tests/test_public_api.py
```

首次运行可能触发两个 native extension 的 JIT build。首次 focused 时间包含编译，不得与后续 cached full regression 时间比较。

## Full regression

```bash
python -m pytest -vv
```

只有 focused 与 full 都通过，才进入 benchmark。pytest 总耗时只用于工程回归，不代表 kernel latency。

## Non-instrumented multi-trial workload

快速验证 runner/trial schema：

```bash
python benchmarks/run_decode_engine_workload.py \
  --quick \
  --trials 2 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials2_quick.csv
```

正式 release 数据：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/week12_decode_engine_workload_trials3.csv

python benchmarks/summarize_decode_engine_trials.py \
  --input benchmarks/results/week12_decode_engine_workload_trials3.csv \
  --output benchmarks/results/week12_decode_engine_workload_trials3_summary.md
```

正式矩阵必须是 3 workloads x 2 dtypes x 2 backends x 3 trials = 36 rows。CSV 和 summary 必须记录同一个 Git commit；validator 会拒绝缺行、重复行、commit 不一致、invariant failure、block accounting failure、torch/fused trajectory drift、seed 不连续或 backend order 未交替。

## Instrumented complete-step profiler

先跑最小代表场景：

```bash
python benchmarks/profile_decode_engine.py \
  --workload mixed_steady \
  --dtype float16 \
  --append-backends torch fused_cuda \
  --quick \
  --export-trace \
  --output-dir benchmarks/profiles/week12_decode_engine_quick \
  --summary-output benchmarks/results/week12_decode_engine_profile_quick_summary.md
```

Profiler ranges 会增加 CPU 开销；nested `engine_step` 与 append/decode device totals 也可能重叠。它们只用于解释性能组成，不能替换上一节 non-instrumented p50/p90/p99。

quick range/trace 通过后，生成 release 要求的正式 12-case attribution matrix。正式矩阵不导出大体积 Chrome trace：

```bash
python benchmarks/profile_decode_engine.py \
  --workload all \
  --dtype both \
  --append-backends torch fused_cuda \
  --output-dir benchmarks/profiles/week12_decode_engine \
  --summary-output benchmarks/results/week12_decode_engine_profile_summary.md
```

生成器会严格检查 3 workloads x 2 dtypes x 2 backends = 12 rows、统一 Git commit、CUDA event 非零，以及 `engine/preflight/append/decode` range count 与 successful/backpressure step 的对应关系。任一条件不满足时不接受该 summary 作为 release evidence。

## 结果文件与提交规则

默认忽略：

```text
benchmarks/results/*.csv
benchmarks/results/*.log
benchmarks/profiles/
```

公开仓库提交精简 Markdown summary，不提交大体积 Chrome trace。summary 至少记录 commit、设备、环境、命令、配置、validated 状态、核心表和负结果。

WSL 结果通过 Windows 目录中转：

```bash
mkdir -p /mnt/c/Users/user/flashdec_results
cp benchmarks/results/<result-files> /mnt/c/Users/user/flashdec_results/
```

Mac 再通过 Windows OpenSSH 拉取；代码同步始终使用 GitHub，不用手工复制代码文件。

## Release candidate 检查

提交 release 文档后执行：

```bash
python scripts/check_release.py --require-clean
```

该命令检查必需 artifact、`pyproject.toml`/`flashdec.__version__` 一致性和 clean Git worktree。当前版本仍为 `0.0.0`，因此此时不要求 tag。

完成所有 GPU/clean-install gate 后：

1. 将 `pyproject.toml` 和 `flashdec/__init__.py` 同步改为 `0.1.0`。
2. 将 Changelog 的 Unreleased 内容整理为 `## [0.1.0] - YYYY-MM-DD`。
3. 提交版本变更，确认 worktree clean。
4. 创建 annotated tag：`git tag -a v0.1.0 -m "FlashDec v0.1.0"`。
5. 执行 `python scripts/check_release.py --require-clean --require-evidence --require-tag`。
6. 推送 commit/tag，并在 GitHub release 中链接 reproducibility 和结果摘要。

不得在 GPU/clean-install gate 未完成时提前创建或推送 tag。

## Release gate status

| gate | 当前状态 | 完成证据 |
| --- | --- | --- |
| Kernel/runtime historical correctness | 已完成 | Week 1-11 weekly docs 与结果摘要 |
| 首轮 dynamic workload/invariant | 已完成 | `week12_decode_engine_workload_summary.md` |
| Multi-trial runner/validator code | 已实现，待真实数据 | 36-row RTX CSV + generated summary |
| Complete-step profiler code | 已实现严格 12-case/range validator，待上板 | validated profile summary + quick trace |
| Clean WSL editable install | 待执行 | 新 venv 命令、pip freeze、pytest/quick 输出 |
| Package version `0.1.0` | 未设置 | pyproject/package version equality |
| `v0.1.0` tag | 未创建 | `check_release.py --require-tag` |

当前只能称为 `v0.1.0 candidate`，不能称为已发布版本。
