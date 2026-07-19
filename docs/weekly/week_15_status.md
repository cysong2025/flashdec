# Week 15 状态记录

## 本周主题

R4-A Trusted CUDA Transaction Fast Path 已在`d25107f`冻结；本周继续实现R4-B Persistent Transaction Metadata，并把当前边界推进到“core与配对证据工具就绪、等待RTX correctness/quick/formal”。

## 已实现

- `fused_rope_kv_append()` 继续验证 block id、block offset 与 position 的 CUDA 值域。
- private trusted raw primitive 复用同一结构检查与 native launch，只跳过五次 device-value reduction；它不导出到 `flashdec` public namespace。
- `PagedKVCache.begin_token()` 在 host 上验证 position、offset、physical block id 与 request block table 的 provenance，并在失败时沿用 reservation rollback。
- `write_token_layer_fused_cuda()` 根据 detached handle 的 transaction id/version/request ids 回查当前 open internal state；调用方修改 snapshot 中的位置 tensor 不会改变真实写入或 rollback ownership。
- R4-A冻结路径中`DecodeEngine`只调用Cache public transaction API；R4-B新增明确的package-private borrowed-metadata/Engine hooks，让math读取Cache private bundle，同时不把内部tensor暴露为public API。
- checked/trusted benchmark 使用同 commit、相同输入/状态轨迹和交替执行顺序；正式 complete-token wall 区间只包含 `synchronize + perf_counter + synchronize`，profiler 独立运行。
- R4-B 在open transaction发布前原子构建positions、physical block ids、block offsets、block tables和effective seq lens五tensorcanonical bundle；Engine math只借用private metadata，public Cache/Engine snapshot始终no-alias。
- commit/abort/error释放device bundle，terminal state只保留host tombstone；build/materialization/reuse/release/resident counters和invariant验证无历史GPU metadata积累。
- 新paired runner在同一commit中用benchmark-only hooks恢复legacy materialization boundary；materialized为`2L+2`个Cache views/token，persistent为1次materialization加`L`次reuse，两侧都使用R4-A trusted raw math。

## 明确未改变

- transaction begin/commit/abort、layer 顺序、single seq_len publish 与 block ownership。
- Triton decode、RoPE/KV native kernel math、Scheduler、shared-prefix residency 与已冻结 kernel 参数。
- R4-A正式证据只验证trusted validation边界，不能把其`4018449`结果当作R4-B跨commit基线；R4-B的controlled ablation必须在同一commit内完成。
- R4-B counter中的view只统计Cache transaction-view materialization，不表示全部Engine result tensor clone。

## 当前验证状态

Mac 工作区已执行以下本地 gate：

- 新 R4 benchmark/summary 与既有 multi-layer benchmark/summary dependency-free tests：修复后 `34 tests` 全部通过。
- `python3 -m compileall -q flashdec tests benchmarks scripts`：通过。
- `python3 scripts/check_docs.py`：`Documentation check: PASS (66 files)`。
- `python3 scripts/check_release.py --require-evidence`：private `0.0.0` tree gate 通过。
- `git diff --check` 与 runner/summary `--help`：通过。

RTX 5070 WSL 最终在 commit `4018449` 完成 focused CUDA suite：`73 passed, 23 subtests passed`；完整回归：`410 passed, 48 subtests passed`。两组均无 failure，覆盖 fused raw dispatch、multi-layer transaction、Engine、benchmark 与 strict summary。

第一次 FP16 quick 已写出 CSV，但 strict summary 拒绝 `profile_append_cpu_ms_per_layer=0`。审计确认运行时并非零 host cost：runner 先调用 `key_averages()`，随后用 `{event.key: event}` 再次压平，而 PyTorch 会按 device type 与 user annotation 分组；同名 CUDA group 覆盖 CPU user annotation 后，读取其 `cpu_time_total` 合法得到 0。第一版修复改读 unaggregated events 并精确选择 CPU user annotation，但仍错误假设同一个 CPU range 必须携带 device total；`.item()` 与 CUDA event 已改为按原始事件计数。validator 始终保持严格，修复前 CSV 作废。

commit `4ee5fab` 的第二次 quick 进一步发现：fresh capture 的第一个 `flashdec::paged_decode` CPU user annotation 有效 host time 为 `116.108 us`，但 correlated device total 为 0。commit `4e18f5d` 随后改用同名 CUDA user annotation 作为 device source；这使 quick 能写出 summary，但正式矩阵证明该 CUDA peer 并不是 PyTorch 保证的一一契约。

commit `4e18f5d` 的第三次 RTX 5070 FP16 quick 中，`l2_b4_c32` checked/trusted token p50 为 `2.403638/1.346150 ms`，trusted p50/TPS ratio 为 `1.7856x/1.8755x`；append CPU ratio 为 `2.3751x`，item/local-scalar 从 `20/20` 降至 `0/0`。完整 token p50 绝对减少 `1.057488 ms`（约 `44.0%`）。这些 wall/CPU/scalar 字段保留为 provisional 方向；旧 append/decode device 和 CUDA-event 字段依赖同名 CUDA user spans，撤回并等待重算。

commit `e88900a` 的正式矩阵在一个 l4 attribution probe 执行完 `2 steps x 4 layers`、得到 8 个 CPU append ranges 后，只捕获 7 个同名 CUDA user annotations。runner 在写 CSV 前按旧契约终止，因此没有正式性能数据。CPU `8/8` 证明 Engine 没有少执行 layer；问题位于 Kineto capture/证据定义。PyTorch 只保证 `record_function` CPU label，不能把同名 CUDA user annotation 当硬 peer。

commit `5d2f9c0` 的 l4 FP16 stress 在三次全新 probe 中都得到首个 decode CPU range 的正 host time（`61.332/71.163/69.992 us`）与零 correlated device time；runner 三次后 fail closed且没有写 CSV。WARMUP→active 与 retry都不能修复该稳定缺口，因此不再把 `device_time_total=0` 误分类为偶发 trace loss，也不增加 retry 次数或加入 prime。

新契约使用 CPU-only profiler `wait=0, warmup=1, active=1` schedule：warmup token完整执行后 abort，再进入 active evidence；CPU annotation 的 inclusive host提供 attribution，checked/trusted scalar count证明每层五次同步是否删除。range/scalar 少记时可用相同 seed和全新 probe最多重采集三次并记录 attempt count，多记或业务错误立即透传。append/decode device与 CUDA activity字段从 strict schema删除，未来需要时另建 CUDA Event/Nsight probe。同一 trial 的 checked/trusted 正式 wall都在 attribution前完成，retry不会污染 paired wall顺序。

旧 quick 只有单 dtype、单缩小 case、单 trial；p90/p99 来自极小样本，不能形成稳定尾延迟或全矩阵结论。Profiler inclusive CPU total只能归因 host sync，不能替代 non-instrumented wall。

commit `4018449` 的五轮正式矩阵已完成：`8 cases x 2 dtypes x 2 paths x 5 trials = 160 rows`、80 个 paired trials，全部 16 个 `dtype x case` 分组均为 `trusted_faster`，且各组 p50 五轮最小值都大于 1。overall p50/p90/p99、TPS 与 append CPU/layer ratio 为 `1.7307x/1.6751x/1.6944x/1.7131x/2.3612x`，通过预设 p50 gate。16 个分组中仍有 7 个 p99 range 穿过 1，因此不声明稳定尾延迟收益，也不把 CPU attribution 解释为 kernel device time。R4-A 已冻结；完整证据见[R4-A 五轮正式摘要](../../benchmarks/results/r4_fused_transaction_fast_path_trials5_summary.md)。

R4-B core、paired runner和strict summary已完成本地实现就绪检查：16个dependency-free tests、`py_compile`和diff check通过。Mac没有torch/pytest，因此没有运行Cache/Engine tensor correctness、CUDA pointer/no-alias、fused parity或完整pytest；RTX quick/formal也尚未执行。当前不能写“R4-B完成”“persistent更快”或任何p50/p99收益，尚未生成的formal summary也不进入release evidence gate。

## 下一步

1. 在RTX运行paged Cache、multi-layer transaction/Engine与R4-B benchmark/summary focused pytest，确认五tensorpointer稳定、public in-place tampering隔离、atomic snapshot、rollback和terminal cleanup。
2. focused通过后运行`l4_b4_c128`、FP16、3-trial quick和strict summary；任何parity、trajectory、counter或resident错误都先修复，不进入formal。
3. quick通过后运行FP16/BF16、8 cases、2 paths、5 trials的160-row正式矩阵，保留p50/p90/p99完整range。
4. 只在overall p50 `>=1.05x`且16/16分组paired p50五轮最小值严格大于1时保留persistent默认；否则归档负结果并恢复materialized，再进入R4-C。
