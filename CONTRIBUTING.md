# Contributing to FlashDec

FlashDec 是一个以 correctness、状态所有权和可复现实验为优先级的研究型 runtime。Issue、bug report、文档修正和范围明确的实现建议都欢迎；较大的功能改动应先说明要回答的[研究问题](docs/research_questions.md)、语义边界和验证方案。

## Issue 与变更提案

请从 [GitHub issue selector](https://github.com/cysong2025/flashdec/issues/new/choose) 选择对应表单：

- correctness、状态 invariant 或回归使用 `Correctness / regression`；
- API、开发环境或复现问题使用 `Usage / environment question`；
- 范围明确的维护改动或经所有者批准的新研究使用 `Scoped change / evidence proposal`。

Issue 必须提供 commit、worktree、环境和最小复现；性能提案还必须写清 paired baseline、shape/dtype、seed、warmup/repeat/trial、计时边界和负结果处理。Issue 本身不授权版本、tag、可见性、许可证或 release 变更。

## 开发环境

推荐 Linux/WSL、Python 3.10+。GPU 路径需要 CUDA-compatible PyTorch、Triton，以及与 PyTorch CUDA build 匹配的 Toolkit。完整环境要求见[复现指南](docs/reproducibility.md)。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cuda-extension]"
```

## 提交前检查

Dependency-free 检查：

```bash
python scripts/check_docs.py
python scripts/check_release.py --require-evidence --require-public
python -m compileall -q flashdec tests benchmarks scripts
git diff --check
```

安装开发依赖后运行：

```bash
python -m pytest -q -ra
```

GPU 结果必须记录 device、PyTorch/CUDA、commit、shape、seed、warmup/repeat/trial 和计时边界。首次 JIT build 不应混入稳态延迟；profiler 字段只能用于归因。

## 变更原则

- 新 kernel 必须与 PyTorch reference 对齐，并覆盖错误输入。
- Cache 或 Scheduler 变更必须说明所有权、状态迁移和失败原子性。
- 性能改动同时保留绝对延迟、ratio 和负结果，不只报告最佳 case。
- 不把不同 shape、layout、硬件或计时范围的数字直接比较。
- 更新行为或公开结论时，同步更新相关设计、兼容性和复现文档。
- `0.0.0` 表示研究型 API；兼容性声明必须绑定[兼容性矩阵](docs/compatibility.md)与对应 correctness 证据。

## Contribution licensing and provenance

提交代码、文档、数据或实验材料前，请确认你有权贡献这些内容。提交 pull request 即表示你同意按仓库根目录 `LICENSE` 的相同条款授权该贡献。不得提交来源不明、许可证不兼容、受雇主或第三方保密义务约束的代码、数据或 benchmark 输出；确需引入第三方内容时，必须在 PR 中标明来源、许可证和兼容性依据。

参与项目即表示同意遵守[行为准则](CODE_OF_CONDUCT.md)。仓库的系统边界和非目标见[研究问题](docs/research_questions.md)与[总体设计](docs/design.md)，正式结果入口见[benchmark 结果索引](benchmarks/results/README.md)，安全报告与一般支持分别见[安全政策](SECURITY.md)和[支持说明](SUPPORT.md)。
