# FlashDec 当前状态与后续目标

FlashDec 的 R1–R5 技术目标、正式 GPU 证据和 canonical summaries 已完成，并已整理为可公开审阅的 `0.0.0` research preview；该状态不增加运行时功能、不修改冻结性能路径，也不把历史环境重跑当作 fresh-environment 复现。统一交付入口见[交付状态](DELIVERY_STATUS.md)，阶段历史见[项目演进](PROJECT_PLAN.md)，完整结果见[结果索引](../benchmarks/results/README.md)。

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

当前仓库是 public `0.0.0` research preview。R1–R5 的工程完成和源码公开都不等于 `v0.1.0` 已发布；clean install、版本升级和 tag 是独立的稳定发布门。

## 项目整理结果

- README、范围、计划、路线图、兼容性、性能和复现文档使用同一当前状态。
- [交付状态](DELIVERY_STATUS.md)集中列出 R1–R5 能力、证据、限制和负结果。
- [结果索引](../benchmarks/results/README.md)区分 Git 跟踪的 canonical summaries 与本地忽略的 CSV、日志、quick/profile 产物。
- release checker 覆盖完整 R1 surface、交付入口、复现入口和 R1–R5 canonical evidence。
- 历史 weekly 记录与失败实验继续保留；它们用于追溯，不作为当前状态入口。

## 当前代码维护范围

`0.0.0` preview 维护期间，代码功能只处理范围内的 correctness bug、回归、文档一致性和证据可追溯性。不自动开启新的 kernel sweep、R4-B 微调、vLLM serving 对比、HTTP 服务、完整模型 forward、sampling、swap/offload、TP/PP 或多机执行。

调用方可以向 R4-C 路径导入已经构建的多层 prompt/context K/V；FlashDec 不执行模型 prompt/prefill forward，也不负责 tokenizer、prefix 内容哈希或 K/V 构建。

## 已完成：`0.0.0` Public-source Gate

1. 根许可证、package metadata、引用信息、贡献说明和发布检查保持一致。
2. 公开入口不含个人环境痕迹，并提供安全、支持、行为准则和引用入口。
3. 可审计图表概括 canonical evidence，同时保留负结果、ratio 方向和不可比边界。
4. 文档、结构、dependency-free tests、敏感信息和 GitHub 配置检查通过。
5. 仓库 visibility、分支保护、依赖与安全设置完成公开后复核。

公开状态仍是 pre-release `0.0.0` research preview，不提供全新环境安装保证。完整项目清单见[公开发布清单](PUBLIC_RELEASE_CHECKLIST.md)。

## 仍暂停：Fresh environment 与 `v0.1.0` Gate

本次源码公开不恢复新目录/新 virtualenv 复现。未来启动稳定发布工作时按以下顺序执行：

1. 在全新 WSL virtualenv 中完成 editable install，并保存依赖与环境记录。
2. 运行 dependency-free checks、CPU/reference suite、RTX focused/full regression 和 release quick workload。
3. 审核所有输出与 commit 绑定，确认 worktree clean。
4. 将 `pyproject.toml` 与 `flashdec.__version__` 同步升级为 `0.1.0`。
5. 创建并验证 `v0.1.0` tag 和 GitHub release。

在上述 gate 完成前，不称项目为已发布 `v0.1.0`，也不提前创建或推送 tag。详细命令保留在[复现指南](reproducibility.md)。

## 日常质量检查

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence --require-public
python -m compileall -q flashdec tests benchmarks scripts
python -m pytest -q -ra
git diff --check
```
