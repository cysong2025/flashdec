# FlashInfer Paged Decode Baseline 设计

## 1. 研究问题与边界

本设计选择一个版本固定、可公开安装的 FlashInfer paged decode 实现，与 FlashDec Triton paged decode 做有限 kernel-only 对比。目标不是证明某个系统全面更快，而是建立可审核的公开基线：输入语义、数值正确性、计时边界、版本和 shape 必须同时可追溯。

对比只覆盖三个 backend：

- FlashDec Triton：token-major paged decode，`block_size=32`、`num_warps=2`、`num_stages=None`。
- FlashInfer FA2：官方 `BatchDecodeWithPagedKVCacheWrapper`，`backend="fa2"`、`use_tensor_cores=False`。
- FlashInfer FA2 Tensor Core option：同一 decode wrapper 与 backend，设置公开的 `use_tensor_cores=True` dispatch 选项；它不是另一个包版本，也不宣称是独立命名的专用 decode kernel。

RTX 5070 canonical evidence 包含 72-row 正式 CSV、通过验证的 strict summary 与 focused/full correctness 记录。结论严格限制在共同 kernel-only 边界；不声明端到端 runtime 胜负或稳定生产尾延迟。

## 2. 依赖与 API 约束

Canonical GPU evidence 使用独立的 Linux/Python 3.12 virtualenv，并通过
`constraints/flashinfer-cu128.txt` 固定已经在 RTX 5070 验证可安装的核心栈：

```text
torch==2.11.0+cu128
triton==3.6.0
flashinfer-python==0.6.15.post1
cuda-toolkit==12.8.1
cuda-python==12.9.1
cuda-bindings==12.9.7
cuda-pathfinder==1.6.0
ninja==1.13.0
```

只使用该版本公开导出的 `BatchDecodeWithPagedKVCacheWrapper`，不复制 FlashInfer 内部 kernel，不调用非公开符号，不修改第三方源码。版本与支持环境以 [FlashInfer 安装文档](https://docs.flashinfer.ai/installation.html) 和 [PyPI 包记录](https://pypi.org/project/flashinfer-python/) 为准，wrapper 语义以 [BatchDecodeWithPagedKVCacheWrapper API](https://docs.flashinfer.ai/api/attention.html#flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper) 为准。`baseline` extra 继续固定 FlashInfer 包；`constraints/flashinfer-cu128.txt` 只约束正式 GPU 证据环境，不把 CUDA wheel 版本写入 FlashDec 的通用 package metadata。

必须先从 PyTorch cu128 index 安装 torch/triton，再在 constraints 下解析 FlashInfer，最后以 `--no-deps` 安装本项目：

```bash
python -m pip install \
  -c constraints/flashinfer-cu128.txt \
  "torch==2.11.0+cu128" "triton==3.6.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

python -m pip install \
  -c constraints/flashinfer-cu128.txt \
  "flashinfer-python==0.6.15.post1" \
  "cuda-python==12.9.1" \
  "cuda-bindings==12.9.7" \
  "cuda-pathfinder==1.6.0" \
  "cuda-toolkit==12.8.1" \
  "ninja==1.13.0" pytest \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

不得在已有 cu128 环境中无 constraints 地直接执行 `pip install -e ".[baseline]"`；FlashInfer 的未限定 torch 依赖可能让 resolver 升级到 Torch 2.13/CUDA 13。安装过程中出现的 `nvidia-cuda-nvdisasm==13.3.73` 是 CUTLASS DSL 的反汇编工具依赖，不代表 PyTorch runtime 或 Toolkit 已切换到 CUDA 13。

FlashInfer 的 `plan()`、JIT 编译与 workspace 初始化在计时前完成。CUDA-core 和 Tensor-core wrapper 各自持有 128 MiB workspace；一个 case/backend/dtype 的 wrapper 在 warmup 和正式 repeat 之间复用，不把每次 token 前的 plan 包装成 decode kernel 成本。

## 3. 共同输入与 layout 对齐

每个 case 只构建一组逻辑输入，三个 backend 共用：

- 同一 `Q`、K pages 和 V pages；
- 同一 physical page 编号与每个 request 的 page table；
- 同一组实际 `seq_lens`，每个 case 使用表中预注册的固定 context；
- 同一 `sm_scale = 1 / sqrt(head_dim)`；
- 同一 dtype、GQA head mapping、page size 和 seed。

FlashDec token-major page 的逻辑 shape 为：

```text
[num_pages, num_kv_heads, page_size, head_dim]
```

它直接对应 FlashInfer 的 `HND` page layout。FlashInfer wrapper 所需的 K/V 组合 view、CSR `indptr/indices` 和 `last_page_len` 只从同一 K/V pages、`block_tables` 与 `seq_lens` 确定性派生，不改变 page 顺序或 token 语义。这些 API 适配在计时前完成。

## 4. 固定 shape 与正式矩阵

四个 case 沿用 FlashDec 已有 paged decode 证据的命名和 shape：

| case | batch | Q heads | KV heads | head dim | context tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `small` | 1 | 32 | 8 | 128 | 128 |
| `medium` | 16 | 32 | 8 | 128 | 1024 |
| `large` | 16 | 32 | 8 | 128 | 8192 |
| `large_batch` | 64 | 32 | 8 | 128 | 4096 |

一个 case 内的所有 request 使用同一个预注册 context length；三个 backend 共用同一 `seq_lens` tensor。正式矩阵固定为：

```text
4 cases x 2 dtypes x 3 backends x 3 trials = 72 rows
dtypes: float16, bfloat16
```

每个 row 是一个 case/dtype/backend/trial 的独立结果。同一 case/dtype/trial 只构建一份输入对象，三个 backend 共用，CSV 记录相同 seed 和 `page_table_digest`。case、dtype 与 backend 顺序都按 trial 轮转，避免固定先后顺序恒定偏向某个组合。

## 5. 正确性与计时边界

所有 backend 的固定前 `min(2, batch)` 个 request 先与同一 PyTorch paged decode reference 比较，完整 batch 再与 FlashDec 输出做 cross-backend 比较。这使大 context 的 reference 开销受控，同时不放弃完整输出对齐。FP16/BF16 容差由 runner 固定并写入 CSV；runner 同时记录最大绝对误差和逐元素 `abs_error / (atol + rtol * abs(expected))` 的最大比值，strict summary 要求该比值不超过 1。任一 backend 不通过时，整个 case 不纳入可比性能证据。summary 必须校验 72-row 矩阵完整性、配对 `page_table_digest`、seed、版本、shape、dtype、backend 和 trial 唯一性。

性能只用 CUDA event 计时下列边界：

```text
FlashDec:              flashdec paged decode kernel dispatch
FlashInfer FA2:        wrapper.run(...) / kernel dispatch
FlashInfer FA2 + TC:   wrapper.run(...) / kernel dispatch
```

明确排除：

- Torch/FlashInfer import 和依赖安装；
- 首次 JIT/code generation 与 FlashInfer `plan()`；
- workspace、Q/K/V、page table 和 paged metadata 构建；
- FlashDec block table 到 FlashInfer CSR metadata 的 API 适配；
- PyTorch reference 计算和 correctness 比较；
- CUDA context 初始化、结果写盘与 summary。

runner 会在计时前完成 JIT/plan 和 warmup，在每个 event 区间内只发起对应 run/kernel dispatch。summary 报告绝对 p50/p90/p99 与吞吐，并固定用 `FlashDec p50 / external p50` 与 `external TPS / FlashDec TPS` 表达比值；两者大于 1 都表示对应 FlashInfer backend 占优。另一个 `logical_workload_gbps_p50` 只把每个 Q/K/V/output 元素计一次，用于 shape-normalized workload rate；它排除 metadata、cache 行为和实现特有的重复读取，不是实测 DRAM bandwidth。该基线不设预先选定的胜负门。

formal strict summary 默认且强制 `trials=3, warmup=10, repeats=50`；quick summary 必须显式声明 `trials=1, warmup=2, repeats=10`。runner 会把完整 argv 写入每一行，canonical summary 直接展示 runner command，避免只从外部文档猜测采样参数。

## 6. 公平性限制与不可比项

这是共同 paged decode 语义下的 kernel-only 对比，不是两个 runtime 或 serving 系统的端到端对比。以下内容不放在同一 speedup 表中：

- FlashInfer 的 plan/workspace/JIT 开销与 FlashDec lazy JIT 开销；
- FlashDec allocator、Shared Prefix、Scheduler、multi-layer transaction 与 DecodeEngine workload；
- prefill、RoPE/KV append、sampling、CUDA Graph 或完整 Transformer forward；
- 两个项目不同的安装成本、编译缓存和 API 适配复杂度；
- `use_tensor_cores=False/True` 之间的内部算法选择、workspace 需求或数值路径差异。

如果某个预注册 shape/dtype/backend 在固定版本上不受支持，不得静默删除该 row 以得到更完整的性能表；应保留错误与环境记录，并将矩阵标记为不完整。修订预注册矩阵时必须另行记录理由和证据版本。

## 7. RTX 5070 执行命令

下列命令使用独立结果目录，同时保留原始 log、CSV 和 summary：

```bash
set -o pipefail

export RESULT_DIR="$HOME/flashdec_results/flashinfer_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
git status --short

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

python -m pip freeze --all \
  > "$RESULT_DIR/pip_freeze.txt"
```

RTX 5070 是 SM 12.0。CUDA 12.8 下必须在任何 FlashInfer import/JIT 前显式设置 `FLASHINFER_CUDA_ARCH_LIST=12.0a`；不使用需要 CUDA 12.9 的 `12.0f`。runner 会在 import FlashInfer 前严格检查上述 distributions、`CUDA_HOME` 的绝对路径/realpath、可执行 `nvcc` 的 release `12.8`/version `12.8.93` 和 arch list，并把实际值写入每个 CSV row；strict summary 再拒绝缺字段或漂移。

focused correctness：

```bash
python -m pytest -q -ra \
  tests/test_paged_decode.py \
  tests/test_flashinfer_baseline.py \
  tests/test_flashinfer_baseline_benchmark.py \
  tests/test_flashinfer_baseline_summary.py \
  2>&1 | tee "$RESULT_DIR/focused.log"
```

quick smoke：

```bash
python benchmarks/run_flashinfer_baseline.py \
  --case medium \
  --dtype float16 \
  --trials 1 \
  --quick \
  --warmup 2 \
  --repeat 10 \
  --require-clean \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_quick.csv" \
  2>&1 | tee "$RESULT_DIR/quick.log"

python benchmarks/summarize_flashinfer_baseline.py \
  --input "$RESULT_DIR/flashinfer_paged_decode_baseline_quick.csv" \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_quick_summary.md" \
  --expected-trials 1 \
  --expected-warmup 2 \
  --expected-repeats 10 \
  --expected-cases medium_b16_ctx1024 \
  --expected-dtypes float16 \
  2>&1 | tee "$RESULT_DIR/quick_summary.log"
```

formal 72-row matrix：

```bash
python benchmarks/run_flashinfer_baseline.py \
  --case all \
  --dtype both \
  --trials 3 \
  --warmup 10 \
  --repeat 50 \
  --require-clean \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv" \
  2>&1 | tee "$RESULT_DIR/formal.log"

python benchmarks/summarize_flashinfer_baseline.py \
  --input "$RESULT_DIR/flashinfer_paged_decode_baseline_trials3.csv" \
  --output "$RESULT_DIR/flashinfer_paged_decode_baseline_summary.md" \
  --expected-trials 3 \
  --expected-warmup 10 \
  --expected-repeats 50 \
  2>&1 | tee "$RESULT_DIR/formal_summary.log"
```

最后运行完整回归与证据门：

```bash
python -m pytest -q -ra \
  2>&1 | tee "$RESULT_DIR/full.log"

python scripts/check_release.py --require-clean --require-evidence \
  2>&1 | tee "$RESULT_DIR/release_check.log"
```

## 8. Canonical evidence 与结果

Canonical evidence 使用以下验证契约：

1. focused/full correctness 无 failure；
2. quick 覆盖三个 backend，每个 row 的 `reference_validated`、`cross_backend_validated` 与 `validated_invariants` 全部为 `True`；
3. formal 恰好包含 72 rows，8 个 case/dtype 分组各含 3 backend x 3 trials；
4. 所有可比 row 的 seed、`page_table_digest`、shape、seq_lens、page metadata 与 `sm_scale` 对齐；
5. strict summary 通过矩阵、`3/10/50` 正式采样强度、固定 cu128 dependency stack、CUDA Toolkit realpath/NVCC `12.8.93`、`FLASHINFER_CUDA_ARCH_LIST=12.0a`、128 MiB workspace、clean-worktree、归一化 tolerance ratio、correctness 与正值/有限 latency 校验；
6. canonical summary 绑定带时区的 run timestamp、FlashDec commit、Python/PyTorch/Triton/PyTorch CUDA、CUDA package versions、`CUDA_HOME`、FlashInfer `0.6.15.post1`、arch list、GPU、runner command 和计时边界，并展示绝对 p50/p90/p99；
7. 结论同时记录胜出、持平、负结果与不可比项，不从单轮 p99 外推生产尾延迟。

RTX 5070 evidence 绑定 commit `d7d4feb`：post-schema focused 为 `93 passed, 37 subtests passed`，quick 为 3 rows，formal 为 72 rows/3 trials，完整回归为 `453 passed, 94 subtests passed`，clean-tree release check 为 `PASS`。strict summary 验证所有 row 的 reference/cross-backend parity、page-table pairing、固定环境和 timing invariant；canonical evidence 见 [FlashInfer baseline 正式摘要](../benchmarks/results/flashinfer_paged_decode_baseline_summary.md)。

正式矩阵中，FlashInfer FA2 CUDA-core 与 tensor-core 的 8 个 dtype/case p50 ratio 几何平均分别为 `1.2003x` 与 `1.2284x`，16/16 个 backend/dtype/case 三轮范围严格高于 1。该范围是三轮观察到的 `[min,max]`，不是置信区间；FP16 small 的两条比较共用同一 FlashDec baseline，并同时出现明显上界扩张。绝对 p99 的 7/16 组范围重叠且有两组 tensor-core 中位数方向反转，因此本证据不声明稳定尾延迟优势。以上结论只适用于本设计的共同 paged-decode kernel-only 边界，不比较 runtime 调度、KV ownership、transaction、planning/JIT 或服务吞吐。
