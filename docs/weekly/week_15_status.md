# Week 15 状态记录

## 本周主题

R4-A Trusted CUDA Transaction Fast Path：删除 Cache-owned multi-layer fused append 每层重复的 CUDA index reduction + `.item()`，同时保留公开 raw primitive 的防御性检查；正式证据通过后冻结 R4-A，当前转入 R4-B persistent transaction metadata。

## 已实现

- `fused_rope_kv_append()` 继续验证 block id、block offset 与 position 的 CUDA 值域。
- private trusted raw primitive 复用同一结构检查与 native launch，只跳过五次 device-value reduction；它不导出到 `flashdec` public namespace。
- `PagedKVCache.begin_token()` 在 host 上验证 position、offset、physical block id 与 request block table 的 provenance，并在失败时沿用 reservation rollback。
- `write_token_layer_fused_cuda()` 根据 detached handle 的 transaction id/version/request ids 回查当前 open internal state；调用方修改 snapshot 中的位置 tensor 不会改变真实写入或 rollback ownership。
- `DecodeEngine` 继续只调用 Cache public transaction API，不跨越到 private Cache method。
- checked/trusted benchmark 使用同 commit、相同输入/状态轨迹和交替执行顺序；正式 complete-token wall 区间只包含 `synchronize + perf_counter + synchronize`，profiler 独立运行。

## 明确未改变

- transaction begin/commit/abort、layer 顺序、single seq_len publish 与 block ownership。
- Triton decode、RoPE/KV native kernel math、Scheduler、shared-prefix residency 与已冻结 kernel 参数。
- 每层 `_transaction_view()` 仍会 materialize CUDA metadata 并复制 block table；R4-A 不能描述成完全 sync/copy-free，persistent metadata 属于独立 R4-B。

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

## 下一步

1. 在 `begin_token()` 的 host provenance 验证后一次性构建 Cache-owned positions、physical ids、offsets、block tables 与 effective seq lens device metadata。
2. 让 multi-layer Engine 跨 layer 复用 internal bundle；public transaction/Engine handle 始终 detached，不与内部 tensor alias。
3. commit/abort 后立即释放 internal GPU metadata，只保留 terminal host tombstone；补齐 OOM/failure atomic rollback、篡改隔离、pointer stability 与无历史显存积累测试。
4. 建立同 commit `materialized/persistent` A/B；两侧均使用 R4-A trusted raw math，并严格验证每 token materialization、reuse、release、parity、rollback 与 Engine trajectory。
5. quick correctness 通过后再运行 FP16/BF16、8 cases、五轮正式矩阵；保留 p90/p99 全范围，未形成稳定证据时不扩展到 CUDA Graph。
