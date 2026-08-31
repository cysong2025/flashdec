# FlashDec Reproducibility Guide

## 目的与证据边界

本文定义 FlashDec `0.0.0` 研究原型的证据复核流程。复现分为四层，不能用后一层的成功替代前一层：

1. environment/package：依赖、commit、CUDA Toolkit 和编译器可见。
2. correctness：reference、runtime state machine、native extension 和 Triton 对齐。
3. non-instrumented benchmark：CUDA-event 或完整 Engine wall-clock 性能。
4. instrumented profiling：阶段归因和 Chrome trace，不作为 non-instrumented latency 证据。

所有公开数字必须绑定 commit、命令、GPU、PyTorch/CUDA、shape、dtype、warmup/repeat/trial 和结果摘要。Dependency-free checks 可在无 GPU 环境执行；GPU correctness 与性能结论只绑定下述 RTX 5070 WSL 参考环境。

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

这里记录的是已验证参考环境，不表示项目只能在这些精确版本运行；其他组合必须重新执行 correctness，不得直接继承性能结论。约束文件只服务其注明的实验环境，不宣称通用锁定所有平台。

## 开发环境安装

以下命令用于建立开发环境。它不是对任意新机器的安装保证；版本不匹配时先参考[兼容性说明](compatibility.md)。

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
- `baseline`：固定 `flashinfer-python==0.6.15.post1`；必须配合 [`constraints/flashinfer-cu128.txt`](../constraints/flashinfer-cu128.txt) 并使用独立环境。

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

## 验证编排器

手工命令仍保留在后续章节作为计时边界说明；实际执行时可由 dependency-free 编排器保证顺序、tracked-clean/CUDA 预检与产物存在性。

```bash
export FLASHDEC_EXPORT_DIR="/path/outside/repository/flashdec_results"
```

先确认完整命令，不执行 GPU 工作：

```bash
python scripts/run_validation.py \
  --phase all \
  --dry-run \
  --export-dir "$FLASHDEC_EXPORT_DIR"
```

按验证层执行：

```bash
python scripts/run_validation.py \
  --phase local \
  --phase focused \
  --phase full

python scripts/run_validation.py \
  --phase trials-quick \
  --phase profile-quick \
  --export-dir "$FLASHDEC_EXPORT_DIR"

python scripts/run_validation.py \
  --phase trials-formal \
  --phase profile-formal \
  --export-dir "$FLASHDEC_EXPORT_DIR"
```

`all` 等价于从 `local` 到 `profile-formal` 的全部 evidence phase，不包含仓库一致性检查；后者单独运行：

```bash
python scripts/run_validation.py --phase release
```

`release` phase 同时启用 `--require-clean --require-evidence`，但不会修改版本或创建 tag。编排器只允许 `benchmarks/results/`、`benchmarks/profiles/` 下的 untracked result artifacts，并要求每一步实际更新目标文件；formal evidence 不能基于 tracked source diff 或 untracked source。`--allow-dirty` 仅供 non-formal 开发实验，formal phase 会拒绝该选项。

## CPU/reference 验证层

CPU/reference suite 不需要 NVIDIA GPU 或 CUDA Toolkit，但 Linux 环境仍需安装项目依赖：

Scheduler planner 与 benchmark/schema helpers 没有 torch/pytest 依赖，可先在任意 Python 3.10+ 环境独立执行：

```bash
python -m unittest discover -s tests -p 'test_scheduler.py' -v
python -m unittest discover -s tests -p 'test_benchmark_helpers.py' -v
python -m unittest discover -s tests -p 'test_profile_decode_engine.py' -v
python -m unittest discover -s tests -p 'test_validation_orchestrator.py' -v
python -m unittest discover -s tests -p 'test_scheduler_workload_summary.py' -v
python -m unittest discover -s tests -p 'test_multi_layer_workload_*.py' -v
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
  tests/test_scheduler_workload_summary.py \
  tests/test_multi_layer_transaction.py \
  tests/test_multi_layer_engine.py \
  tests/test_multi_layer_workload_benchmark.py \
  tests/test_multi_layer_workload_summary.py \
  tests/test_validation_orchestrator.py \
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
  tests/test_multi_layer_transaction.py \
  tests/test_multi_layer_engine.py \
  tests/test_multi_layer_workload_benchmark.py \
  tests/test_multi_layer_workload_summary.py \
  tests/test_public_api.py
```

首次运行可能触发两个 native extension 的 JIT build。首次 focused 时间包含编译，不得与后续 cached full regression 时间比较。

## Full regression

```bash
python -m pytest -vv
```

只有 focused 与 full 都通过，才进入 benchmark。pytest 总耗时只用于工程回归，不代表 kernel latency。

### 2026-08-30 R6-A 回归记录

代码提交 `87d8a34` 在 NVIDIA GeForce RTX 5070（SM 12.0）与 CUDA Toolkit/NVCC `12.8.93` 上完成一次 clean-checkout 回归。环境为 Python `3.12.3`、PyTorch `2.11.0+cu128`、Triton `3.6.0`，并设置 `FLASHINFER_CUDA_ARCH_LIST=12.0a`；`python -m pip check` 报告 `No broken requirements found`。

```bash
python scripts/run_validation.py --python "$(command -v python)" \
  --phase local --phase focused

python scripts/run_validation.py --python "$(command -v python)" \
  --phase full

python scripts/check_release.py \
  --require-clean --require-evidence --require-public
```

结果如下：

- local：48 个 dependency-free `unittest` 全部通过；
- focused：`254 passed, 20 subtests passed in 78.53s`；
- full：`501 passed, 100 subtests passed in 168.13s`；
- release check：clean worktree、公开许可证与 canonical evidence gate 均为 `PASS`。

本轮 focused 覆盖 transaction terminal-state 有界回收、detached handle provenance、scheduler decision 的 snapshot/config binding，以及 Engine 在任何 lifecycle/cache mutation 前的 authoritative snapshot 重建与 canonical replan。原始 `environment.log`、`local_focused.log`、`full.log` 和 `release_check.log` 保存在仓库外，不提交到 Git。

该记录复用已验证的隔离 cu128 开发环境，因此属于当前代码的 GPU correctness/release regression，不构成全新环境安装保证，也不产生新的性能结论。正式性能数字继续绑定各自 canonical summary 中记录的 evidence commit。

### 2026-08-30 R7 vLLM/Qwen 闭环记录

证据提交 `61836b6` 在同一 RTX 5070 上分别使用 vLLM/cu130 与核心 cu128 环境复核。vLLM 环境补充的 `pytest==9.1.1` 只作为测试工具安装，没有重新解析项目、Torch 或 CUDA dependencies；安装前后 `python -m pip check` 都为 `No broken requirements found`。

```bash
# vLLM/cu130 environment
python -m pytest -q -ra \
  tests/test_vllm_plugin.py \
  tests/test_vllm_backend.py \
  tests/test_vllm_attention_microbench_summary.py \
  tests/test_vllm_model_correctness_summary.py \
  tests/test_vllm_model_latency_summary.py \
  tests/test_vllm_serving_benchmark_summary.py

# core cu128 environment
python -m pytest -q -ra
python scripts/check_docs.py
python scripts/check_release.py \
  --require-clean --require-evidence --require-public
```

结果：

- vLLM/cu130 专项：`21 passed, 14 dependency deprecation warnings`；
- core cu128 full：`531 passed, 1 skipped, 100 subtests passed in 10.73s`；
- 唯一 skip：cu128 环境未安装 vLLM；对应 case 已由 cu130 专项实际执行；
- docs check：`PASS (52 files)`；
- clean-tree public release check：`PASS`。

原始 full、pip-check、docs-check 与 release-check logs 保存在仓库外的 R7 result directory。该回归证明 plugin/backend/summary 与核心 runtime 在两个固定环境中通过，不会把未通过的 model-latency 或 serving-throughput 性能门槛改写为通过。

## Non-instrumented multi-trial workload

快速验证 runner/trial schema：

```bash
python benchmarks/run_decode_engine_workload.py \
  --quick \
  --trials 2 \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_trials2_quick.csv
```

完整 multi-trial 数据：

```bash
python benchmarks/run_decode_engine_workload.py \
  --trials 3 \
  --dtype both \
  --output benchmarks/results/decode_engine_workload_trials3.csv

python benchmarks/summarize_decode_engine_trials.py \
  --input benchmarks/results/decode_engine_workload_trials3.csv \
  --output benchmarks/results/decode_engine_workload_trials3_summary.md
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
  --output-dir benchmarks/profiles/decode_engine_quick \
  --summary-output benchmarks/results/decode_engine_stage_profile_quick_summary.md
```

Profiler ranges 会增加 CPU 开销；nested `engine_step` 与 append/decode device totals 也可能重叠。它们只用于解释性能组成，不能替换上一节 non-instrumented p50/p90/p99。

quick range/trace 通过后，生成 12-case attribution matrix。完整矩阵不导出大体积 Chrome trace：

```bash
python benchmarks/profile_decode_engine.py \
  --workload all \
  --dtype both \
  --append-backends torch fused_cuda \
  --output-dir benchmarks/profiles/decode_engine \
  --summary-output benchmarks/results/decode_engine_stage_profile_summary.md
```

生成器会严格检查 3 workloads x 2 dtypes x 2 backends = 12 rows、统一 Git commit、CUDA event 非零，以及 `engine/preflight/append/decode` range count 与 successful/backpressure step 的对应关系。任一条件不满足时不接受该 summary 作为 canonical evidence。

## Scheduler 容量与进展证据

```bash
python benchmarks/run_scheduler_workload.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/scheduler_workload_trials3.csv

python benchmarks/summarize_scheduler_workload.py \
  --input benchmarks/results/scheduler_workload_trials3.csv \
  --output benchmarks/results/scheduler_capacity_progress_summary.md
```

完整矩阵必须是 2 cases x 2 dtypes x 3 policies x 3 trials = 36 rows。Canonical evidence 使用已经审核并提交的 Markdown summary；CSV 继续作为本地原始证据。结论是 lifetime commitment 的容量安全与进展保证，不是所有普通 workload 下无条件更快。

## Multi-layer transaction 证据

```bash
python benchmarks/run_multi_layer_engine.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/multi_layer_transaction_trials3.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/multi_layer_transaction_trials3.csv \
  --output benchmarks/results/multi_layer_transaction_summary.md
```

正式矩阵必须是 12 cases x 2 dtypes x 2 backends x 3 trials = 144 rows。Validator 会检查 pair trajectory、transaction、block accounting、rollback、profiler、seed 和 backend order。正式 latency 来自 non-instrumented complete-token 路径；profiler 字段只做 append/decode/launch 归因，p99 必须连同范围报告。

## Shared-prefix capacity confirmation

3-trial matrix 保留为 metadata-cache 优化前基线；8-trial confirmation 扩大样本，并让四种 hit-rate 顺序各轮转两次：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 8 \
  --output benchmarks/results/shared_prefix_workload_trials8.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/shared_prefix_workload_trials8.csv \
  --output benchmarks/results/shared_prefix_capacity_summary.md \
  --expected-trials 8 \
  --expected-dtypes float16 bfloat16
```

正式 confirmation 必须是 `4 hit rates x 2 dtypes x 8 trials = 64 rows`，seed 连续且四种顺序各出现两次。summary 必须继续验证 capacity commitment、physical block/byte、prefix lifecycle、materialized context、immutable contents 和 final cleanup。commit `fe72e27` 的结果表明所有非零 complete/scheduler/Engine p50 range 均跨 1，因此只接受 near-neutral/no stable direction 结论；容量/admission 结论独立成立。

## Trusted transaction 配对证据

该实验不比较两个不同 commit。runner 在同一进程、同一 commit 和同一 Cache/Engine API 上交替切换 checked/trusted raw validation，输入、context、transaction trajectory、parity 与 rollback 必须配对一致。

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case l4_b4_c128 \
  --dtype float16 \
  --trials 3 \
  --quick \
  --output benchmarks/results/trusted_transaction_l4_stress.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/trusted_transaction_l4_stress.csv \
  --output benchmarks/results/trusted_transaction_l4_stress_summary.md \
  --expected-trials 3 \
  --expected-cases l4_b4_c32 \
  --expected-dtypes float16

python benchmarks/run_fused_transaction_fast_path.py \
  --case all \
  --dtype both \
  --trials 5 \
  --output benchmarks/results/trusted_transaction_trials5.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/trusted_transaction_trials5.csv \
  --output benchmarks/results/trusted_transaction_summary.md \
  --expected-trials 5
```

先通过 l4 3-trial stress quick，再运行正式矩阵。正式矩阵必须是 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`。complete-token latency 使用 `synchronize + perf_counter + synchronize`，计时区间不创建 CUDA event；同一 trial 必须先完成两条 path 的 wall，再开始任一 attribution/rollback，不能让 profiler retry 插在 paired wall 中间。独立 profiler 使用 CPU-only WARMUP→active schedule；active CPU user annotation 的 inclusive host time必须逐 layer 正且有限，checked 每个 profiled layer恰有 5 次 `aten::item` 与 `_local_scalar_dense`、trusted 为 0。少记 range/scalar 属于 capture incompleteness并触发整 probe 重建，最多三次且写入 `profile_attempt_count`；多出 range/scalar 属于 active-work/fast-path 契约错误，必须立即失败而不能重试。CPU FunctionEvent correlated device time与 CUDA activity不在 strict schema；需要分段 GPU 时间时另做 CUDA Event/Nsight probe。Validator 还必须验证 block/transaction/Engine accounting、exact parity、rollback 和交替顺序。

commit `4018449` 在 RTX 5070、CUDA 12.8 上得到 160 rows、80 paired trials；全部 16 个 `dtype x case` 分组为 `trusted_faster` 且五轮 p50 最小值均大于 1。overall p50/p90/p99、TPS 和 append CPU/layer ratio 为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`；focused `73 passed, 23 subtests passed`，完整回归 `410 passed, 48 subtests passed`。7/16 分组的 p99 range 穿过 1，因此不得声明稳定尾延迟收益。Trusted path 保持默认；canonical evidence 见[五轮配对摘要](../benchmarks/results/trusted_transaction_summary.md)。

## Persistent metadata 负结果

Evidence commit `8047a9c` 在同一 trusted math 下执行 materialized/persistent 配对矩阵。focused `101 passed`、完整回归 `434 passed, 48 subtests passed`；matrix 为 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`，strict validator 检查 exact parity、block/transaction/Engine trajectory、rollback、metadata counters、CPU ranges 与 terminal zero-resident。overall p50/TPS/append CPU 为 `1.2493x/1.2392x/3.0366x`，但只有 13/16 分组五轮全部胜出，预注册 keep gate 失败，因此 materialized metadata 保持默认。Canonical evidence 见[五轮负结果摘要](../benchmarks/results/persistent_metadata_candidate_summary.md)。

该 candidate 的 runner/validator 只存在于 evidence commit。需要重验原始 CSV 时，应在独立 worktree checkout 对应 commit，不能把当前代码与旧 CSV 混用：

```bash
git worktree add ../flashdec-persistent-metadata-evidence 8047a9c
cd ../flashdec-persistent-metadata-evidence
python benchmarks/summarize_persistent_transaction_metadata.py \
  --input <persistent_transaction_metadata_trials5.csv> \
  --output <recomputed_summary.md> \
  --expected-trials 5
```

## Integrated scheduled multi-layer 证据

组合 workload 固定使用 trusted fused append、Triton decode 与 materialized metadata。先执行 targeted/focused correctness：

```bash
python -m pytest -q -ra \
  tests/test_integrated_workload.py \
  tests/test_integrated_workload_config.py \
  tests/test_integrated_workload_benchmark.py \
  tests/test_integrated_workload_summary.py \
  tests/test_multi_layer_transaction.py \
  tests/test_multi_layer_engine.py \
  tests/test_shared_prefix_blocks.py \
  tests/test_scheduler.py
```

然后执行 FP16 quick：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case l2_c32 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/integrated_runtime_lifecycle_quick.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_quick.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_quick_summary.md \
  --expected-trials 1 \
  --expected-cases l2_c32 \
  --expected-dtypes float16
```

quick 必须通过 reference/observed digest、rollback、reuse 与 cleanup gate。之后运行正式矩阵：

```bash
python benchmarks/run_integrated_scheduled_multi_layer.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/integrated_runtime_lifecycle_trials3.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/integrated_runtime_lifecycle_trials3.csv \
  --output benchmarks/results/integrated_runtime_lifecycle_summary.md \
  --expected-trials 3
```

正式 CSV 必须为 24 rows：2/4 layers、64/128 context、FP16/BF16、3 trials。四个 case 每轮轮转，seed 连续。随机 tensor 构建、prefix registration 和 terminal prefix eviction 不进入 logical-step wall；private multi-layer context writes、scheduler、transaction/decode 与 finish/cancel 在计时范围内。summary 只报告绝对 p50/p90/p99/TPS 与跨 trial range；没有预注册 backend ratio，也不能把 shared-prefix hit 解释成 latency speedup。

完整矩阵后运行 full regression 与 evidence integrity check，并把 stdout/stderr 与 CSV 一起保存：

```bash
python -m pytest -q -ra
python scripts/check_release.py --require-evidence
```

## FlashInfer 有限外部基线

对比只使用 `flashinfer-python==0.6.15.post1` 的官方 `BatchDecodeWithPagedKVCacheWrapper`、`backend="fa2"` 和 `use_tensor_cores=False/True` 两条路径。证据环境固定为 Python 3.12、Torch `2.11.0+cu128`、Triton `3.6.0`、CUDA Toolkit `12.8.1`，完整核心 pin 见 `constraints/flashinfer-cu128.txt`。必须使用独立 virtualenv，不能在已有环境中用无 constraints 的一条 `.[baseline]` 命令让 pip 自由升级 torch。

```bash
set -o pipefail

export BASELINE_VENV="$HOME/.virtualenvs/flashdec-baseline-$(date +%Y%m%d_%H%M%S)"
python3.12 -m venv "$BASELINE_VENV"
source "$BASELINE_VENV/bin/activate"

export RESULT_DIR="$HOME/flashdec_results/flashinfer_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
git status --short

python -m pip install --upgrade pip "setuptools>=68,<82" wheel \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --timeout 600 --retries 30 --progress-bar off \
  2>&1 | tee "$RESULT_DIR/install_build_tools.log"

python -m pip install \
  -c constraints/flashinfer-cu128.txt \
  "torch==2.11.0+cu128" "triton==3.6.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
  --timeout 600 --retries 30 --progress-bar off \
  2>&1 | tee "$RESULT_DIR/install_torch_cu128.log"

python -m pip install \
  -c constraints/flashinfer-cu128.txt \
  "flashinfer-python==0.6.15.post1" \
  "cuda-python==12.9.1" \
  "cuda-bindings==12.9.7" \
  "cuda-pathfinder==1.6.0" \
  "cuda-toolkit==12.8.1" \
  "ninja==1.13.0" pytest \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --timeout 600 --retries 30 --progress-bar off \
  2>&1 | tee "$RESULT_DIR/install_flashinfer_pinned.log"

python -m pip install --no-build-isolation --no-deps -e . \
  2>&1 | tee "$RESULT_DIR/install_flashdec_editable.log"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export FLASHINFER_CUDA_ARCH_LIST="12.0a"

python -m pip check \
  2>&1 | tee "$RESULT_DIR/pip_check.log"

"$CUDA_HOME/bin/nvcc" --version \
  2>&1 | tee "$RESULT_DIR/nvcc_version.log"

python -c "import os, torch, triton, flashinfer; from importlib.metadata import version; print('arch_list=', os.environ.get('FLASHINFER_CUDA_ARCH_LIST')); print('device=', torch.cuda.get_device_name()); print('capability=', torch.cuda.get_device_capability()); print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('triton=', triton.__version__); print('flashinfer=', version('flashinfer-python')); print('cuda_toolkit=', version('cuda-toolkit')); print('cuda_python=', version('cuda-python')); print('cuda_bindings=', version('cuda-bindings')); print('cuda_pathfinder=', version('cuda-pathfinder'))" \
  2>&1 | tee "$RESULT_DIR/flashinfer_sm120a_probe.log"

flashinfer show-config \
  2>&1 | tee "$RESULT_DIR/flashinfer_show_config.log"

python -m pip freeze --all > "$RESULT_DIR/pip_freeze.txt"
```

RTX 5070 的 compute capability 是 `(12, 0)`。在 CUDA 12.8 下，FlashInfer 必须在首次 import/JIT 前显式使用 `FLASHINFER_CUDA_ARCH_LIST=12.0a`；runner 会在 FlashInfer import 前 fail closed，strict summary 也会验证 CSV 记录的完整依赖版本、`CUDA_HOME`/realpath、NVCC release `12.8`/version `12.8.93` 和 arch list。传递安装中的 `nvidia-cuda-nvdisasm==13.3.73` 只是 CUTLASS DSL 工具依赖，不表示 PyTorch runtime 或 Toolkit 升级到 CUDA 13。

先运行 dependency-free 契约、既有 paged decode 回归和 optional CUDA/FlashInfer correctness：

```bash
python -m pytest -q -ra \
  tests/test_paged_decode.py \
  tests/test_flashinfer_baseline.py \
  tests/test_flashinfer_baseline_benchmark.py \
  tests/test_flashinfer_baseline_summary.py
```

quick 使用 uniform fixed-context medium/FP16，三个 backend 各生成一行：

```bash
python benchmarks/run_flashinfer_baseline.py \
  --case medium \
  --dtype float16 \
  --trials 1 \
  --quick \
  --warmup 2 \
  --repeat 10 \
  --require-clean \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_quick.csv"

python benchmarks/summarize_flashinfer_baseline.py \
  --input "$RESULT_DIR/flashinfer_paged_decode_baseline_quick.csv" \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_quick_summary.md" \
  --expected-trials 1 \
  --expected-warmup 2 \
  --expected-repeats 10 \
  --expected-cases medium_b16_ctx1024 \
  --expected-dtypes float16
```

quick summary 只有在 3 rows 的 `reference_validated`、`cross_backend_validated` 和 `validated_invariants` 全部为 `True`，且 strict matrix/pairing 校验通过时才可接受。然后运行正式矩阵：

```bash
python benchmarks/run_flashinfer_baseline.py \
  --case all \
  --dtype both \
  --trials 3 \
  --warmup 10 \
  --repeat 50 \
  --require-clean \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv"

python benchmarks/summarize_flashinfer_baseline.py \
  --input "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv" \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_summary.md" \
  --expected-trials 3 \
  --expected-warmup 10 \
  --expected-repeats 50
```

正式 CSV 必须恰好为 `4 cases x 2 dtypes x 3 backends x 3 trials = 72 rows`。small/medium/large/large_batch 都是 uniform fixed context，不解释为变长 workload 或 context 上界。三个 backend 共用 Q/K/V、page table、`seq_lens`、`sm_scale` 和 seed；FlashDec token-major `[page, head, token, dim]` 对应 FlashInfer `HND`。CUDA event 只计 `run`/kernel dispatch，排除 input construction、reference validation、FlashInfer plan/JIT 和 workspace/metadata 构建。

summary 用 `FlashDec p50 / external p50` 和 `external TPS / FlashDec TPS` 报告描述性比值；大于 1 表示对应 FlashInfer backend 占优，并另表展示绝对 p90/p99 的跨 trial median/range。`logical_workload_gbps_p50` 是每个 Q/K/V/output 元素只计一次的共同 workload proxy，不是任何 backend 的实测 DRAM bandwidth。runner 的 evidence 模式要求 clean worktree，记录完整命令和固定 cu128 环境；strict summary 验证 `12.0a`、quick `1/2/10` 或 formal `3/10/50` 采样强度、每个 FlashInfer wrapper 的 128 MiB workspace，以及 reference/cross-backend normalized tolerance ratio 不超过 1。该比较不设性能 pass/fail 或胜者门。Summary 产生后运行：

```bash
python -m pytest -q -ra
python scripts/check_release.py --require-clean --require-evidence
```

结果先写入仓库外的 `$RESULT_DIR`，避免未审核的 formal summary 改变被测 worktree。commit `d7d4feb` 按该流程在 RTX 5070 得到 post-schema focused `93 passed, 37 subtests passed`、3-row quick、72-row/3-trial formal、full `453 passed, 94 subtests passed` 与 clean-tree evidence check `PASS`。审核后的精简结果见 [canonical summary](../benchmarks/results/flashinfer_paged_decode_baseline_summary.md)；它绑定 `2026-07-26T15:28:08+08:00`、固定 cu128 环境、`12.0a`、clean commit 与完整 runner command。公平性与不可比边界见 [FlashInfer 基线设计](design_flashinfer_baseline.md)。

## vLLM Qwen2.5-3B 外部比较

R7 使用与核心 cu128/FlashInfer 环境隔离的 vLLM 环境。正式证据固定为 Python 3.12.3、`vLLM==0.25.1`、PyTorch `2.11.0+cu130`、Triton 3.6.0、PyTorch CUDA 13.0、RTX 5070 和本地 Qwen2.5-3B-Instruct BF16 weights。`vllm` extra 是版本声明，不建议在已有 Torch 环境中让 pip 自由解析整套 CUDA 依赖；优先复用已验证的 vLLM environment，再使用 `--no-deps` editable install。下面的 R7 runner/summarizer 命令绑定历史闭环提交 `61836b6`，应在该提交的独立 worktree/clone 中运行；当前 HEAD 已冻结 R8 protocol，不能重新生成历史 12-row R7 summary。

```bash
source /home/<user>/projects/QwenServe-12G/.venv/bin/activate
git -C /home/<user>/projects/flashdec worktree add \
  /home/<user>/projects/flashdec-r7-evidence 61836b6
cd /home/<user>/projects/flashdec-r7-evidence

python -m pip install --no-deps -e .
python -m pip check

export MODEL_DIR=/home/<user>/models/Qwen2.5-3B-Instruct
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/SHA256SUMS"
(cd "$MODEL_DIR" && sha256sum --check SHA256SUMS)

python -c "import torch, triton, vllm; print(torch.__version__, torch.version.cuda, triton.__version__, vllm.__version__); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

固定运行环境：

```bash
export VLLM_PLUGINS=flashdec
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export RESULT_DIR=/home/<user>/flashdec_results/r7_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"
git status --short
```

`VLLM_USE_FLASHINFER_SAMPLER=0` 只关闭该环境中独立的 FlashInfer sampling path；attention A/B 仍严格使用 `TRITON_ATTN` 与 `CUSTOM`。`VLLM_WSL2_ENABLE_PIN_MEMORY=1` 是 vLLM V2 runner 在 WSL 的固定设置。正式 runner 要求 clean worktree，原始 JSON/CSV/log 写入仓库外 `$RESULT_DIR`。

先验证 backend registry、fallback、split correctness 与 strict summaries：

```bash
python -m pytest -q -ra \
  tests/test_vllm_plugin.py \
  tests/test_vllm_backend.py \
  tests/test_vllm_attention_microbench_summary.py \
  tests/test_vllm_model_correctness_summary.py \
  tests/test_vllm_model_latency_summary.py \
  tests/test_vllm_serving_benchmark_summary.py
```

完整 attention、model correctness、fixed-batch model 和 online serving 命令见 [benchmark 命令](../benchmarks/README.md#vllm-qwen25-3b-外部比较)。正式协议包含：

- attention：5 cases × 2 backends × 5 trials = 50 rows，100/500 ms time-window sampling；
- model correctness：8 个固定 prompts、greedy、32 output tokens、两个独立 backend processes；
- model latency：2 cases × 2 backends × 3 processes = 12 rows，每进程 3 warmups/5 measurements；
- serving：128 prompts、concurrency 8、input4096/output128、8 warmups、3 个独立 server pairs，标准 `vllm bench serve`。

R7 的审核结果必须连同负结果一起解释：

| 层次 | 结果 |
| --- | --- |
| attention | external-kernel gate `PASS`；B8 ctx1024/2048 为 `0.8025x/0.7926x` |
| generation correctness | 第一 token 8/8 一致；split/non-split 诊断为 8/8 完整 rollout 一致 |
| fixed-batch model | 两个 case 都略快，但 target `0.9976x <= 0.995x` 失败 |
| online serving | median/p90 TPOT gate 通过；throughput `1.0019x >= 1.002x` 失败 |

历史提交 `61836b6` 上的 model-latency 与 serving summarizer 会先写出完整 summary，再因冻结门槛失败返回非零；这不是 runner 故障。不得通过删除 gate、四舍五入，或用当前 R8 summarizer 覆盖 summary 把它改为通过。canonical summaries 见 [R7 结果索引](../benchmarks/results/README.md#r7-vllm-qwen外部比较)，fast-path/fallback/数值边界见 [vLLM backend 设计](design_vllm_backend.md)。

### 2026-08-31 R8 长上下文 fixed-batch 闭环

R8 不覆盖 R7 的 model/serving 负结果，而是在 commit `3ba68e3` 上冻结一个新的长上下文 fixed-batch 目标。模型文件先按 `$MODEL_DIR/SHA256SUMS` 全量校验；正式 runner 为每个 case 启动 4 个独立 process pairs（共 8 对），并以 balanced AB/BA 顺序运行固定的 `TRITON_ATTN` 和 `CUSTOM`：

```bash
export VLLM_PLUGINS=flashdec
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
unset FLASHDEC_VLLM_NUM_SPLITS
export MODEL_DIR=/home/<user>/models/Qwen2.5-3B-Instruct
export RESULT_DIR=/home/<user>/flashdec_results/r8_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_DIR"
set -o pipefail

(cd "$MODEL_DIR" && sha256sum --check SHA256SUMS) \
  2>&1 | tee "$RESULT_DIR/model_sha256.log"

python benchmarks/run_vllm_model_latency.py \
  --model "$MODEL_DIR" \
  --output "$RESULT_DIR/model_latency.csv" \
  --case qwen_b8_i512_o2 \
  --case qwen_b8_i8192_o4096 \
  --trials 4 \
  --prime-iters 1 \
  --warmup-iters 1 \
  --num-iters 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --vllm-cache-base "$RESULT_DIR/vllm-cache" \
  --require-clean \
  2>&1 | tee "$RESULT_DIR/run.log"

python benchmarks/summarize_vllm_model_latency.py \
  "$RESULT_DIR/model_latency.csv" \
  --output "$RESULT_DIR/model_latency_summary.md" \
  2>&1 | tee "$RESULT_DIR/summary.log"
```

正式 result directory `r8_3ba68e3_formal_trials4` 的 strict summary 返回 0，结果如下：

| 层次 | paired latency ratio | 冻结门槛与结果 |
| --- | ---: | --- |
| B8 input512/output2 guard | `1.0029x [0.9890,1.0100]` | `<= 1.05x`，PASS |
| B8 input8192/output4096 target | `0.9542x [0.9530,0.9560]` | `<= 0.970x`，PASS |

target 对应 latency 降低 `4.58%`、output TPS 提升 `4.80%`。长上下文跨 backend 的最小共同前缀为 49 tokens、完整一致 7/8；每个 backend 的 warmup/measured 完整输出 hash 在各 trial 内稳定。8 个 `CUSTOM` workers 各自产生一个唯一 marker，全部通过 B8/Q16/KV2/D128/BF16/8-split、case/trial/dataset/commit binding 和 `cuda_graph_capture=true` 校验。

marker 只证明对应 decode CUDA Graph 在 capture 阶段包含成功的 FlashDec multi-split launch；它不是每次 measured replay 的 device-side 直接观测。worker/parent runner 读盘校验 marker 后把 canonical JSON、SHA-256、路径与绑定投影写入 CSV；formal summarizer 复核这些 CSV 字段及其一致性，native rows 则必须没有 marker。summarizer 本身不重新读取原始 marker 文件。

原始证据没有提交到 Git，保留在上述 result directory；关键 SHA-256 为：

| artifact | SHA-256 |
| --- | --- |
| `model_latency.csv` | `ae57b1788abb61847e1faa4ee1a6ab57de0fba309c2cb5317d660e4913d503e2` |
| `model_latency_summary.md` | `f511af02757b66cc75007768c5df7e9180ae31f3ed34853d00d00038e9354520` |
| `run.log` | `8c1956118f877f4f006c4fba50f9c91c814dcc283457c84ebc3ec59723a4b7f7` |
| `summary.log` | `1c8b5e95716fbd1a84528e8d39d6420d1d0b7b3a9023403c854edba2a4509bc6` |
| `evidence_manifest.sha256` | `cf7ea96e39133ff4bf12959877b177c31547d9eb6f17273e94ab35d19946fd57` |

canonical Markdown 见 [R8 长上下文模型摘要](../benchmarks/results/vllm_qwen_long_context_model_latency_summary.md)。这里的端到端范围是离线固定 B8 的 blocking `LLM.generate`，包含 model execution、scheduler、KV cache、sampling 和 Python API，排除 startup/model load、full-length JIT-prime 和 warmup。它不等同于 online serving，不能用于声称 TTFT、TPOT、并发请求吞吐或对 vLLM 默认/最快 backend 的收益；外部 baseline 明确固定为 `TRITON_ATTN`。

## 结果文件与提交规则

默认忽略：

```text
benchmarks/results/*.csv
benchmarks/results/*.log
benchmarks/profiles/
```

公开仓库提交精简 Markdown summary，不提交大体积 Chrome trace。summary 至少记录 commit、设备、环境、命令、配置、validated 状态、核心表和负结果。

原始结果应保存在仓库外的独立目录；代码同步使用 Git，不手工复制源码树。

## 仓库与证据一致性检查

提交受版本控制的证据前执行：

```bash
python scripts/check_release.py --require-clean
```

该命令检查必需 artifact、`pyproject.toml`/`flashdec.__version__` 一致性和 clean Git worktree。`--require-tag` 只用于确实存在 annotated version tag 的 commit；检查器本身不会修改版本或创建 tag。

## 已知安装与版本限制

- `0.0.0` 表示研究原型 API；兼容范围以[兼容性矩阵](compatibility.md)为准。
- 完整 GPU 回归与性能结果绑定上文 RTX 5070/CUDA 12.8 环境，不能外推到任意软件或硬件组合。
- Editable install 是开发路径，不等同于跨平台 wheel 或稳定安装接口。
- 研究结论的权威入口是[结果索引](../benchmarks/results/README.md)，不是测试计数或本地日志。
