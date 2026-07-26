# Week 18 状态记录

## 本周主题

R1–R5 完成后的项目整理与交付审查：先统一当前状态、证据入口、仓库结构和 release 边界，随后启动 `0.0.0` public-readiness；不修改冻结的 kernel/runtime，也不执行新环境复现。

## 当前已完成

- 新增 [FlashDec 交付状态](../DELIVERY_STATUS.md)，集中列出 R1–R5 工程能力、主要实现、canonical evidence、负结果和限制。
- 阶段初始整理将 README、文档索引、项目计划、路线图、兼容性、性能报告和下一步统一为“研究交付完成、private `0.0.0`、release 暂停”；这是当时状态，后续 public-readiness 已于 2026-07-26 恢复。
- 修正 R3/R4-C 后仍残留的“未实现 prefix cache / multi-layer prompt prefill”歧义：当前支持调用方预构建 shared prefix 和调用方提供的多层 context K/V，但不执行模型 prefill forward 或内容构建/hash。
- canonical Markdown、忽略的 CSV/log/profile/local backups 和 release evidence gate 的用途已分开说明；本轮不删除本地实验原始文件。
- release artifact gate 补入统一交付状态和 R1 `scheduled_workload` 实现/测试覆盖，避免关键交付文件缺失时仍通过结构检查。
- GitHub landing page 使用有限 badge、GitHub light/dark 双主题 runtime SVG 与精简 R1–R5 交付矩阵；图中明确 rotated Q、Cache-owned paged K/V 和 FlashInfer kernel-only evidence 边界。新增 correctness/regression、scoped change/evidence issue forms 和 PR validation template，并将 `.github/` collaboration surface 纳入 docs/release checks。

## 阶段初始范围

- 不新增功能，不重开 kernel sweep，不改默认性能路径。
- 不执行 fresh clone/virtualenv、依赖重装、GPU benchmark 或新环境复现。
- 阶段初始整理不升级 `0.0.0`、不修改仓库可见性或许可证，也不创建 tag/release；随后只恢复 `0.0.0` 源码公开准备，稳定版本与 tag 仍不启动。
- 历史 weekly 和正式负结果保持原样，只修正当前入口中的事实冲突。

## 验证

本轮已在当前环境执行以下 dependency-free 检查：

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence
python -m compileall -q flashdec tests benchmarks scripts
git diff --check
```

并执行 `.github/workflows/quality.yml` 主 job 中列出的 19 个 dependency-free test modules；workflow 另有 Python 3.10 metadata/public-gate compatibility job。结果：

- documentation check：`PASS (77 files)`；
- release evidence check：`PASS`；
- 双主题 SVG XML、完整尺寸渲染、compileall 与 `git diff --check`：通过；
- GitHub issue/workflow YAML parse：通过；
- dependency-free pytest：`177 passed in 0.63s`。

GPU/full 证据继续使用 commit `d7d4feb` 的 `453 passed, 94 subtests passed`；本轮没有运行或声称新的 GPU 结果，也没有创建新环境。

## 下一步

2026-07-26 完成 public-source gate，仓库以 pre-release `0.0.0` research preview 公开；fresh-environment 复现、`v0.1.0` 版本与 tag 继续暂缓。
