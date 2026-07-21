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

## R1 Scheduler 正式证据

```bash
python benchmarks/run_scheduler_workload.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r1_scheduler_workload_trials3.csv

python benchmarks/summarize_scheduler_workload.py \
  --input benchmarks/results/r1_scheduler_workload_trials3.csv \
  --output benchmarks/results/r1_scheduler_workload_trials3_summary.md
```

正式矩阵必须是 2 cases x 2 dtypes x 3 policies x 3 trials = 36 rows。Release evidence 使用已经审核并提交的 Markdown summary；CSV 继续作为本地原始证据。R1 结论是 lifetime commitment 的容量安全与进展保证，不是所有普通 workload 下无条件更快。

## R2 Multi-layer 正式证据

```bash
python benchmarks/run_multi_layer_engine.py \
  --case all \
  --dtype both \
  --trials 3 \
  --output benchmarks/results/r2_multi_layer_engine_trials3.csv

python benchmarks/summarize_multi_layer_trials.py \
  --input benchmarks/results/r2_multi_layer_engine_trials3.csv \
  --output benchmarks/results/r2_multi_layer_engine_trials3_summary.md
```

正式矩阵必须是 12 cases x 2 dtypes x 2 backends x 3 trials = 144 rows。Validator 会检查 pair trajectory、transaction、block accounting、rollback、profiler、seed 和 backend order。正式 latency 来自 non-instrumented complete-token 路径；profiler 字段只做 append/decode/launch 归因，p99 必须连同范围报告。

## R3 Shared Prefix 最终 confirmation

R3-C 3-trial matrix 保留为 metadata-cache 优化前基线。R3-D correctness 通过后，用 8 trials 扩大样本并让四种 hit-rate 顺序各轮转两次：

```bash
python benchmarks/run_shared_prefix_workload.py \
  --hit-rate all \
  --dtype both \
  --trials 8 \
  --output benchmarks/results/r3_shared_prefix_workload_trials8.csv

python benchmarks/summarize_shared_prefix_trials.py \
  --input benchmarks/results/r3_shared_prefix_workload_trials8.csv \
  --output benchmarks/results/r3_shared_prefix_workload_trials8_summary.md \
  --expected-trials 8 \
  --expected-dtypes float16 bfloat16
```

正式 confirmation 必须是 `4 hit rates x 2 dtypes x 8 trials = 64 rows`，seed 连续且四种顺序各出现两次。summary 必须继续验证 capacity commitment、physical block/byte、prefix lifecycle、materialized context、immutable contents 和 final cleanup。commit `fe72e27` 的结果表明所有非零 complete/scheduler/Engine p50 range 均跨 1，因此只接受 near-neutral/no stable direction 结论；容量/admission 结论独立成立。

## R4-A Trusted Transaction 配对证据

R4-A 不比较两个不同 commit。runner 在同一进程、同一 commit 和同一 Cache/Engine API 上交替切换 checked/trusted raw validation，输入、context、transaction trajectory、parity 与 rollback 必须配对一致。

```bash
python benchmarks/run_fused_transaction_fast_path.py \
  --case l4_b4_c128 \
  --dtype float16 \
  --trials 3 \
  --quick \
  --output benchmarks/results/r4_fused_transaction_fast_path_l4_stress.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/r4_fused_transaction_fast_path_l4_stress.csv \
  --output benchmarks/results/r4_fused_transaction_fast_path_l4_stress_summary.md \
  --expected-trials 3 \
  --expected-cases l4_b4_c32 \
  --expected-dtypes float16

python benchmarks/run_fused_transaction_fast_path.py \
  --case all \
  --dtype both \
  --trials 5 \
  --output benchmarks/results/r4_fused_transaction_fast_path_trials5.csv

python benchmarks/summarize_fused_transaction_fast_path.py \
  --input benchmarks/results/r4_fused_transaction_fast_path_trials5.csv \
  --output benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md \
  --expected-trials 5
```

先通过 l4 3-trial stress quick，再运行正式矩阵。正式矩阵必须是 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`。complete-token latency 使用 `synchronize + perf_counter + synchronize`，计时区间不创建 CUDA event；同一 trial 必须先完成两条 path 的 wall，再开始任一 attribution/rollback，不能让 profiler retry 插在 paired wall 中间。独立 profiler 使用 CPU-only WARMUP→active schedule；active CPU user annotation 的 inclusive host time必须逐 layer 正且有限，checked 每个 profiled layer恰有 5 次 `aten::item` 与 `_local_scalar_dense`、trusted 为 0。少记 range/scalar 属于 capture incompleteness并触发整 probe 重建，最多三次且写入 `profile_attempt_count`；多出 range/scalar 属于 active-work/fast-path 契约错误，必须立即失败而不能重试。CPU FunctionEvent correlated device time与 CUDA activity不在 strict schema；需要分段 GPU 时间时另做 CUDA Event/Nsight probe。Validator 还必须验证 block/transaction/Engine accounting、exact parity、rollback 和交替顺序。

commit `4018449` 已在 RTX 5070、CUDA 12.8 完成正式证据：160 rows、80 paired trials，全部 16 个 `dtype x case` 分组为 `trusted_faster` 且五轮 p50 最小值均大于 1。overall p50/p90/p99、TPS 和 append CPU/layer ratio 为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`；focused `73 passed, 23 subtests passed`，完整回归 `410 passed, 48 subtests passed`。7/16 分组的 p99 range 穿过 1，因此不得声明稳定尾延迟收益。R4-A 已冻结，canonical release evidence 为[R4-A 五轮正式摘要](../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)。

## R4-B Persistent Metadata 正式负结果

R4-B evidence commit `8047a9c` 在同一 trusted math 下执行 materialized/persistent 配对矩阵。focused `101 passed`、完整回归 `434 passed, 48 subtests passed`；formal 为 `8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`，strict validator 检查 exact parity、block/transaction/Engine trajectory、rollback、metadata counters、CPU ranges 与 terminal zero-resident。overall p50/TPS/append CPU 为 `1.2493x/1.2392x/3.0366x`，但只有 13/16 分组五轮全部胜出，正式 keep gate 为 fail。主线因此恢复 R4-A/materialized 默认；canonical evidence 为[R4-B 五轮正式负结果](../benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)。

R4-B runner/validator 属于已回滚 candidate，不保留在当前生产树。需要重验原始 CSV 时，应在独立 worktree checkout evidence commit，不能把当前主线代码与旧 CSV 混用：

```bash
git worktree add ../flashdec-r4b-evidence 8047a9c
cd ../flashdec-r4b-evidence
python benchmarks/summarize_persistent_transaction_metadata.py \
  --input <r4_persistent_transaction_metadata_trials5.csv> \
  --output <recomputed_summary.md> \
  --expected-trials 5
```

## R4-C Integrated Scheduled Multi-layer 证据

R4-C 固定使用 R4-A trusted fused append、Triton decode 与 materialized metadata。先执行 targeted/focused correctness：

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
  --case l2_c64 \
  --dtype float16 \
  --trials 1 \
  --quick \
  --output benchmarks/results/r4_integrated_scheduled_multi_layer_quick.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/r4_integrated_scheduled_multi_layer_quick.csv \
  --output benchmarks/results/r4_integrated_scheduled_multi_layer_quick_summary.md \
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
  --output benchmarks/results/r4_integrated_scheduled_multi_layer_trials3.csv

python benchmarks/summarize_integrated_scheduled_multi_layer.py \
  --input benchmarks/results/r4_integrated_scheduled_multi_layer_trials3.csv \
  --output benchmarks/results/r4_integrated_scheduled_multi_layer_trials3_summary.md \
  --expected-trials 3
```

正式 CSV 必须为 24 rows：2/4 layers、64/128 context、FP16/BF16、3 trials。四个 case 每轮轮转，seed 连续。随机 tensor 构建、prefix registration 和 terminal prefix eviction 不进入 logical-step wall；private multi-layer context writes、scheduler、transaction/decode 与 finish/cancel 在计时范围内。summary 只报告绝对 p50/p90/p99/TPS 与跨 trial range；没有预注册 backend ratio，也不能把 shared-prefix hit 解释成 latency speedup。

正式矩阵后运行完整回归与 release evidence check，并把 stdout/stderr 与 CSV 一起保存：

```bash
python -m pytest -q -ra
python scripts/check_release.py --require-evidence
```

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

只有所有者明确启动 release 且完成所有 GPU/clean-install gate 后：

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
| Multi-trial runner/validator | 已完成 | commit `3708b87` 的 36-row RTX CSV + validated summary |
| Complete-step profiler | 已完成 | commit `3708b87` 的 validated 12-case summary + quick trace |
| R1 Scheduler v2 | 已完成 | commit `16de9d4` 的 36-row RTX policy matrix + validated summary |
| R2 Multi-layer transaction | 已完成 | commit `fa0f89a` 的 144-row RTX matrix + validated summary；证据提交 `67bee15` 的 `337 passed, 25 subtests passed in 5.82s` 无跳过完整回归 |
| R3 Shared Prefix correctness | 已完成 | commit `fe72e27` 的 targeted `1 passed`、focused `61 passed, 8 subtests passed` 与 full `361 passed, 25 subtests passed` |
| R3 Shared Prefix benchmark | 已完成 | commit `fe72e27` 的 8-trial/64-row FP16/BF16 RTX confirmation + strict paired/attribution summary |
| R3 metadata hot path | 已完成 | submission-time shared-block cache、lookup-count test 与 authoritative Cache cross-check；性能 near-neutral/no stable direction |
| R4-A trusted transaction | 已完成 | commit `4018449` 的 focused/full correctness、160-row/80-pair RTX 五轮矩阵与 [canonical strict summary](../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)；16/16 p50 分组稳定胜出，p99 保留 7/16 穿 1 的限制 |
| R4-B persistent metadata | 已评估并回滚 | commit `8047a9c` 的 correctness 与 [160-row/80-pair 正式负结果](../benchmarks/results/r4_persistent_transaction_metadata_trials5_summary.md)；overall p50 `1.2493x`，但仅 13/16 分组稳定胜出，未通过 keep gate |
| R4-C integrated workload | 实现就绪，等待 RTX | dependency-free reference/validator、multi-layer prompt transaction、observed/reference digest、rollback/reuse/cleanup tests 与 24-row runner；RTX quick/formal/full 待执行 |
| Clean WSL editable install | 暂停 | 仓库继续 private；收到 release 指令后保存新 venv、pip freeze、pytest/quick 输出 |
| Package version `0.1.0` | 未设置 | pyproject/package version equality |
| `v0.1.0` tag | 未创建 | `check_release.py --require-tag` |

当前是 private `0.0.0` development candidate，不能称为已发布版本。版本、公开与 tag 按所有者要求暂停。
