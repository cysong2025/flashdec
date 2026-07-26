# FlashDec 当前状态与后续目标

FlashDec 的 R1–R5 技术目标、正式 GPU 证据和 canonical summaries 已完成。本轮项目整理不增加功能、不修改冻结性能路径，也不执行新环境复现。统一交付入口见[交付状态](DELIVERY_STATUS.md)，阶段历史见[项目演进](PROJECT_PLAN.md)，完整结果见[结果索引](../benchmarks/results/README.md)。

## 当前基线

```text
Block-aware Scheduler
        -> DecodeEngine
        -> Multi-layer KV Token Transaction
        -> Fused RoPE/KV Append
        -> Triton Paged Decode Attention
```

| 阶段 | 交付状态 | 主要证据 |
| --- | --- | --- |
| R1 Scheduler | 完成并冻结 | 36-row policy matrix；boundary-deadlock 进展保证 |
| R2 Multi-layer | 完成并冻结 | 144-row matrix；commit/rollback 与 complete-token 证据 |
| R3 Shared Prefix | 完成并冻结 | 64-row confirmation；容量节省与 admission 提升 |
| R4 Trusted/Integrated | 完成并冻结 | R4-A 160-row 正结果；R4-B 正式负结果/回滚；R4-C 24-row 组合验证 |
| R5 FlashInfer baseline | 完成并冻结 | 72-row/3-trial 共同 paged-decode kernel-only 对比 |

当前仓库仍是 private `0.0.0` development candidate。R1–R5 的工程完成不等于 `v0.1.0` 已发布；clean install、版本升级、公开设置和 tag 是独立的 release gate。

## 项目整理结果

- README、范围、计划、路线图、兼容性、性能和复现文档使用同一当前状态。
- [交付状态](DELIVERY_STATUS.md)集中列出 R1–R5 能力、证据、限制和负结果。
- [结果索引](../benchmarks/results/README.md)区分 Git 跟踪的 canonical summaries 与本地忽略的 CSV、日志、quick/profile 产物。
- release checker 覆盖完整 R1 surface、交付入口、复现入口和 R1–R5 canonical evidence。
- 历史 weekly 记录与失败实验继续保留；它们用于追溯，不作为当前状态入口。

## 当前维护范围

在所有者明确启动新阶段前，只处理范围内的 correctness bug、回归、文档一致性和证据可追溯性。不自动开启新的 kernel sweep、R4-B 微调、vLLM serving 对比、HTTP 服务、完整模型 forward、sampling、swap/offload、TP/PP 或多机执行。

调用方可以向 R4-C 路径导入已经构建的多层 prompt/context K/V；FlashDec 不执行模型 prompt/prefill forward，也不负责 tokenizer、prefix 内容哈希或 K/V 构建。

## 已暂停：v0.1.0 Release Gate

按所有者要求，本轮不做新目录/新 virtualenv 复现。恢复 release 工作时按以下顺序执行：

1. 在全新 WSL virtualenv 中完成 editable install，并保存依赖与环境记录。
2. 运行 dependency-free checks、CPU/reference suite、RTX focused/full regression 和 release quick workload。
3. 审核所有输出与 commit 绑定，确认 worktree clean。
4. 将 `pyproject.toml` 与 `flashdec.__version__` 同步升级为 `0.1.0`。
5. 经所有者确认后再处理仓库公开设置、许可证、`v0.1.0` tag 和 release。

在上述 gate 完成前，不称项目为已发布 `v0.1.0`，也不提前创建或推送 tag。详细命令保留在[复现指南](reproducibility.md)。

## 日常质量检查

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence
python -m compileall -q flashdec tests benchmarks scripts
python -m pytest -q -ra
git diff --check
```
