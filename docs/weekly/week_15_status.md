# Week 15 状态记录

## 本周主题

R4 Trusted CUDA Transaction Fast Path：删除 Cache-owned multi-layer fused append 每层重复的 CUDA index reduction + `.item()`，同时保留公开 raw primitive 的防御性检查。

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

RTX 5070 WSL 已完成 commit `1169cb8` 的 focused CUDA suite：`40 passed in 2.34s`，覆盖 fused raw dispatch、multi-layer transaction 与 multi-layer Engine。完整回归仍待执行。

第一次 FP16 quick 已写出 CSV，但 strict summary 拒绝 `profile_append_cpu_ms_per_layer=0`。审计确认运行时并非零 host cost：runner 先调用 `key_averages()`，随后用 `{event.key: event}` 再次压平，而 PyTorch 会按 device type 与 user annotation 分组；同名 CUDA group 覆盖 CPU user annotation 后，读取其 `cpu_time_total` 合法得到 0。第一版修复改读 unaggregated events 并精确选择 CPU user annotation，但仍错误假设同一个 CPU range 必须携带 device total；`.item()` 与 CUDA event 已改为按原始事件计数。validator 始终保持严格，修复前 CSV 作废。

commit `4ee5fab` 的第二次 quick 进一步发现：`flashdec::paged_decode` CPU user annotation 的 host time 为 `116.108 us`，但该 CPU event 的 `device_time_total=0`。这是 Triton launch 的 device event 没有关联成 CPU annotation child，并非 decode 没有执行。最终取数契约改为：CPU user annotation 只提供 inclusive host time；同名 CUDA user range 独立提供 device time；两类原始 range 数都必须等于 `steps * layers`。二者不相加，零值仍失败。第二次运行在写 CSV 前终止，同样没有性能结论。

commit `4e18f5d` 的第三次 RTX 5070 FP16 quick 已通过严格 2-row summary。`l2_b4_c32` checked/trusted token p50 为 `2.403638/1.346150 ms`，trusted p50/TPS ratio 为 `1.7856x/1.8755x`；append CPU/device ratio 为 `2.3751x/2.4540x`，CUDA events 从 `166` 降至 `106`，item/local-scalar 从 `20/20` 降至 `0/0`，decode device ratio 为 `1.0062x`。完整 token p50 绝对减少 `1.057488 ms`（约 `44.0%`），与 append attribution 和 scalar extraction 消失方向一致；begin 接近中性，commit 略慢且绝对量很小，不是主要收益来源。

该 quick 只有单 dtype、单缩小 case、单 trial；p90/p99 来自极小样本，不能形成稳定尾延迟或全矩阵结论。Profiler CPU/device totals 来自独立 instrumented run，不能彼此相加，也不能替代 non-instrumented wall。当前 commit 的完整回归和五轮正式 A/B 完成前，不把 R4-A 标记为完成。正式门槛仍是 complete-token p50 总体至少 `1.05x`，且全部 16 个 `dtype x case` 分组的五轮 p50 `[min,max]` 都不穿过 1；未达到门槛就保留负结果并停止扩展到 CUDA Graph。

## 下一步

1. 在 quick 证据 commit 上运行 RTX 5070 完整 pytest，并保存 commit-bound 日志；不得用旧 focused 结果替代。
2. 运行 FP16/BF16、8 cases、checked/trusted、5 trials 的 160-row 正式矩阵，并用 strict summary 验证 80 个配对 trial。
3. 检查 overall p50 `>=1.05x`，并按所有 dtype/case 的五轮 `[min,max]` 审核是否穿过 1；保留 p90/p99 全范围。
4. 根据门槛决定进入 R4-B persistent metadata，或直接转入 R4-C integrated scheduled multi-layer correctness workload。
