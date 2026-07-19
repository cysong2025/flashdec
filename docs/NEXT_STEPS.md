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

## 当前目标：R4-B Persistent Transaction Metadata

状态：R4-A 已在 commit `4018449` 完成并冻结。RTX focused/full correctness 为 `73 passed, 23 subtests passed` 与 `410 passed, 48 subtests passed`；五轮正式矩阵共 160 行、80 个 paired trials，complete-token p50/TPS 几何平均为 `1.7307x/1.7131x`，16/16 个 dtype/case 分组的五轮 p50 range 全部稳定胜出。append inclusive CPU 为 `2.3612x`，l2/l4 checked scalar extraction 为 `20/40`、trusted 为 0。7/16 个 p99 ranges仍跨 1，因此不声明稳定尾延迟或 CUDA device-kernel 加速。仓库继续保持 private。

R4-B 处理 R4-A 明确留下的下一层开销：当前 fused Engine 每 token 会在 begin、每层 launch 前后和 commit 重建 transaction view；每次重新创建 positions、physical block ids、offsets、effective seq lens 并复制 block table。l2/l4 当前分别形成 `2L+2 = 6/10` 套 view。目标是在 Cache host allocator proof 之后为 open transaction 构造一套 private device metadata，跨 layer 只读复用。

R4-B 的边界：

- public `KVTokenTransactionView` 与 `DecodeStepTransaction` 继续是 detached snapshot，绝不与 Cache-owned canonical tensors alias；调用方原地修改 snapshot 不能改变 launch、decode 或 rollback。
- persistent bundle 只在 transaction open 期间存活；commit、abort 或异常必须立即释放 device tensors。`_transactions` 可保留轻量 terminal tombstone，以继续报告 `already committed/aborted`，但不能积累 GPU metadata。
- materialized/persistent 两条 A/B 路径都使用已冻结的 R4-A trusted raw dispatch；不同时改变 CUDA/Triton kernel、output buffer、Scheduler、shared prefix 或 CUDA Graph。
- 用确定性 Cache counters 验证 metadata build/reuse/terminal cleanup，不把 Kineto device correlation重新引入 strict evidence。
- 正式 latency继续来自无 profiler/CUDA Event 的同步 wall；profiler若使用，只报告 CPU attribution，且两条路径的 item/local-scalar必须同为 0。

验证顺序：

1. 覆盖 l2/l4、tail/boundary、下一 token fresh metadata、public snapshot in-place tampering、layer failure rollback、commit/abort terminal cleanup和多 token无 retained-device-bundle增长。
2. 同 commit、相同输入/状态轨迹、交替 materialized/persistent 顺序做 quick stress；strict summary验证 parity、block/transaction/Engine trajectory与metadata counter公式。
3. quick通过后执行 FP16/BF16、8 cases、2 paths、5 trials的160-row正式矩阵。
4. keep门预注册为 overall complete-token p50 `>=1.05x`，且16/16分组五轮p50最小值严格大于1；否则记录负结果并恢复materialized默认，不继续同线微调。
5. R4-B冻结或回滚后，再进入R4-C integrated scheduled multi-layer workload。

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
