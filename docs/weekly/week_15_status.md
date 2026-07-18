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

在修复后 quick、完整回归与正式 A/B 完成前，不记录 speedup，也不把 R4-A 标记为完成。正式门槛是 complete-token p50 总体至少 `1.05x`，且目标 l2/l4 case 跨 trial 不穿过 1；未达到门槛就保留负结果并停止扩展到 CUDA Graph。

## 下一步

1. 拉取 CPU/CUDA range pairing 修复，在 RTX 5070 重跑 focused 与完整 correctness。
2. 重新运行单 case FP16 quick A/B，并用 strict summary 校验成对 CPU/CUDA range 与 item/local-scalar 计数；不复用旧证据。
3. quick 通过后运行 FP16/BF16、l2/l4 五轮正式矩阵。
4. 根据门槛决定进入 R4-B persistent metadata，或直接转入 R4-C integrated scheduled multi-layer correctness workload。
