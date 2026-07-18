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

- 新 R4 benchmark/summary 与既有 multi-layer benchmark/summary dependency-free tests：`29 tests` 全部通过。
- `python3 -m compileall -q flashdec tests benchmarks scripts`：通过。
- `python3 scripts/check_docs.py`：`Documentation check: PASS (66 files)`。
- `python3 scripts/check_release.py --require-evidence`：private `0.0.0` tree gate 通过。
- `git diff --check` 与 runner/summary `--help`：通过。

Mac 没有项目 torch/pytest/CUDA 环境，因此真实 public safety、FP16/BF16 kernel parity、detached-view tampering、focused/full regression 与 paired performance 尚待 RTX 5070 WSL 执行。

在 RTX 证据完成前，不记录 speedup，也不把 R4-A 标记为完成。正式门槛是 complete-token p50 总体至少 `1.05x`，且目标 l2/l4 case 跨 trial 不穿过 1；未达到门槛就保留负结果并停止扩展到 CUDA Graph。

## 下一步

1. 在 RTX 5070 执行 targeted、focused 与完整 correctness。
2. 先运行单 case FP16 quick A/B 并用 strict summary 校验 profiler item/local-scalar 计数。
3. quick 通过后运行 FP16/BF16、l2/l4 五轮正式矩阵。
4. 根据门槛决定进入 R4-B persistent metadata，或直接转入 R4-C integrated scheduled multi-layer correctness workload。
