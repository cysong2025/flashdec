# FlashDec 当前状态与后续目标

本文记录当前完成边界和尚未启动的后续目标。详细阶段历史、实验数据和设计取舍分别归档在[项目演进](PROJECT_PLAN.md)、[性能报告](performance_report.md)和[路线图](ROADMAP.md)。

## 当前基线

FlashDec 已完成从 paged decode kernel 到单 GPU decode runtime 的主链路：

```text
Block-aware Scheduler
        -> DecodeEngine
        -> Multi-layer KV Token Transaction
        -> Fused RoPE/KV Append
        -> Triton Paged Decode Attention
```

已闭合的里程碑：

- kernel 配置冻结：token-major、`block_size=32`、`num_warps=2`、`num_stages=None`。
- Paged KV request lifecycle、block allocate/free/reuse、容量预检和 invariant validation。
- fused CUDA append、动态 DecodeEngine workload 和阶段 profiler 归因。
- R1 block-aware scheduler：容量安全、FIFO + aging、公平 runnable subset 和 stale decision 拒绝。
- R2 multi-layer token transaction：共享位置、单次 seq_len commit 和异常 rollback。
- RTX 5070 最终回归：`337 passed, 25 subtests passed`，无 skipped 或 failure。
- R2 正式矩阵：144 行全部通过严格校验，complete-token p50/p90/TPS 几何平均为 `1.2101x/1.3826x/1.2800x`。
- R3 shared prefix：A-D 已闭合；8-trial/64-row confirmation 继续确认 75% hit 节省 `68.8%`/`5.5 MiB` KV capacity，并把 bounded-pool admission 从 `9/16` 提高到 `16/16`。
- R3-D RTX 回归：targeted `1 passed`、focused `61 passed, 8 subtests passed`、full `361 passed, 25 subtests passed`。

当前版本仍为 `0.0.0`。这表示 release gate 尚未启动，不表示 R1-R3 功能未完成。

## 已完成目标：R3 Shared Prefix Blocks

状态：R3-A ownership、R3-B integration、R3-C benchmark 与 R3-D hot-path metadata cache 已全部闭合。commit `fe72e27` 的 correctness 和优化后 RTX confirmation 均已完成；仓库按所有者要求继续保持 private。

R3 研究重复 system prompt / 固定上下文的 immutable full-block 共享，目标是减少重复 KV physical blocks，同时保持 request lifecycle、transaction rollback 和容量预检的正确性。

分阶段目标：

- R3-A：已完成 prefix 注册、挂载、引用计数、inactive LRU、回收和 CPU/RTX 回归。
- R3-B：已完成 DecodeEngine/scheduler shared residency 与 request-private commitment 分离；focused 为 `56 passed, 14 subtests passed in 5.29s`，完整回归为 `352 passed, 25 subtests passed in 9.37s`。
- R3-C：0%/25%/50%/75% FP16/BF16 三轮正式矩阵共 24 行，matrix、capacity、block/byte、context、immutability、eviction 与 cleanup 全部通过；它作为优化前基线保留。
- R3-D：Engine 将 submission 时已验证的 shared block 数缓存在 request metadata，不再在每个 step 重复查询 prefix registry；Cache state/version invariant 保留。优化后 8 trials、64 行再次验证容量轨迹，75% hit 节省 `44/64` context blocks（`68.8%`，`5.5 MiB`），peak blocks 从 `80` 降至 `36`，bounded-pool admission 从 `9/16` 提高到 `16/16`。
- 最终性能边界：FP16/BF16 的 25%/50%/75% complete p50、scheduler p50 与 Engine p50 paired range 全部跨 1。中位数大多接近 1，但存在未复现的 Engine 整行慢点和尾部尖峰，因此既不声明稳定加速，也不声明稳定回退。

`5.5 MiB` 表示相对私有副本避免占用的 KV pool capacity。fixed-full-batch latency probe 在所有 hit rate 下仍预分配相同 80-block tensor，因此它不是 `torch.cuda.memory_allocated()` 的直接下降；若按 peak blocks right-size pool，才会转化为实际 pool allocation 缩减。

明确边界：

- 只共享调用方已经构建的 immutable full blocks；tail block 保持 request-private。
- 不实现 tokenizer、模型 prefill、sampling、HTTP server 或分布式执行。
- R3-B 只接受已经 resident 且覆盖完整 initial context 的 prefix；admission-time prefix eviction 暂不支持。

所有权与验收细节见[Shared Prefix Blocks 设计](design_shared_prefix_blocks.md)。

## 当前目标：R4 Trusted CUDA Transaction Fast Path

状态：R4-A 代码、配对 benchmark harness、dependency-free validator tests 与 RTX focused CUDA correctness（`40 passed in 2.34s`）已完成。第一次 quick 的 strict summary 暴露 profiler 同名 CPU/CUDA 分组覆盖问题；修复后的 quick、完整回归和正式 performance 尚待执行。仓库继续保持 private。

R2 profiler 显示 multi-layer fused path 的系统收益主要来自 append/launch，而 attention device time 基本不变。进一步审计发现，cache-owned transaction 每个 layer 仍通过公开 raw primitive 执行五次 CUDA index reduction + `.item()`：block id 上下界、offset 上下界和 position 非负。这些值已经由 Cache allocator 在 host 侧构造并证明范围，因此内部路径存在重复的 host/stream synchronization。

R4-A 的边界：

- `flashdec.fused_rope_kv_append()` 保留完整 device-value 检查，public safety 不变。
- `PagedKVCache.begin_token()` 以纯 host invariant 证明 allocator 位置，public transaction API 根据 id 回查该内部状态并使用 private trusted raw launch；DecodeEngine 继续只调用 Cache public API。
- trusted path 继续检查 shape、dtype、device、contiguity、RoPE 参数和 `int64` metadata。
- 本 slice 只删除 device reduction + `.item()`，不把仍存在的 transaction-view H2D materialization/copy 表述为完全无同步。
- 不同时修改 transaction buffer reuse、Triton kernel、Scheduler、shared-prefix metadata 或 CUDA Graph。

验证顺序：

1. public invalid-index、trusted/checked parity、detached-view tampering、Engine public-API routing 与 rollback tests。
2. RTX focused/full correctness。
3. 同 commit checked/trusted quick A/B；正式 wall 使用同步后的 `perf_counter`，CUDA event/profiler 独立归因。
4. 只有 p50 总体至少 `1.05x` 且目标 case 跨 trial 稳定，才进入 transaction metadata reuse；否则记录负结果并停止该优化线。
5. R4-A 冻结后，再实现统一 scheduled multi-layer workload，组合验证 R1/R2/R3。

Profiler quick 必须从 `profiler.events()` 原始事件中精确筛选 CPU user annotation，并验证每个真实 append/decode range；不能用同名 key 字典压平 CPU/CUDA 分组，也不能放宽 validator 接受零 host time。修复前 CSV 不具备可恢复的 CPU attribution，必须重新生成。

详细信任边界见[Multi-layer KV Transaction 设计](design_multi_layer_kv_transaction.md)。

## 暂停目标：private 维护与可选 v0.1.0 Release Gate

状态：R3 技术目标已经完成；按所有者要求暂不公开、不升级版本、不创建 tag。只有收到明确指令后才启动以下 release gate。

执行顺序：

1. 在全新 WSL virtualenv 中执行 editable install。
2. 运行 dependency-free 检查、CPU/reference suite、完整 RTX suite 和 release quick workload。
3. 保存环境、commit、命令与输出，确认公开数字可追溯。
4. 将 `pyproject.toml` 与 `flashdec.__version__` 同步升级为 `0.1.0`。
5. 更新 Changelog，创建并验证 `v0.1.0` tag。

版本升级和 tag 只能发生在 clean-install 证据通过之后。

## 公开发布设置

仓库当前继续保持 private。可见性和许可证不再阻塞 R3；只有准备公开时才重新确认：

- 是否将 GitHub repository 改为 public；
- 采用 MIT、Apache-2.0，或继续保留无开源授权状态；
- GitHub `quality` workflow 和公开链接是否正常。

R5 FlashInfer/vLLM 有限公开对比仍是选择性扩展，在 private R4 与 release gate 完成前不启动。

不在范围内：HTTP 服务、tokenizer、sampling、完整模型 forward、swap/offload、TP/PP 和多机执行。

## 日常质量检查

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence
python -m compileall -q flashdec tests benchmarks scripts
python -m pytest -q -ra
git diff --check
```

Clean install、正式 workload 和 tag 命令见[复现指南](reproducibility.md)。
