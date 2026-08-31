# Scripts

这里记录环境检查、证据校验和结果图生成脚本。

第一步先运行：

```bash
python scripts/check_env.py
```

这个脚本会检查：

- Python 版本。
- Python executable。
- PyTorch 是否安装。
- Triton 是否安装。
- Ninja 是否安装。
- CUDA 是否可用。
- GPU 型号。
- Git commit 与 worktree 是否 clean。
- 环境变量/PyTorch 检测到的 `CUDA_HOME`。
- `nvidia-smi`。
- `nvcc` path。
- `nvcc --version`。

文档链接检查：

```bash
python scripts/check_docs.py
```

该脚本扫描 README、CHANGELOG、CONTRIBUTING、`docs/`、`benchmarks/`、`scripts/` 和 `.github/` 中的 Markdown 链接及 HTML `src`/`srcset` 图片资源，拒绝缺失目标、逃逸出仓库的相对路径，以及与项目实现无关的个人评估措辞；外部 URL 不发起网络请求。`benchmarks/results/` 下由 Git 忽略的 `*_quick_summary.md`、`*_smoke.md` 和 `local_backups/` 不进入扫描。

README 架构图与公开实验概览图：

```bash
python scripts/generate_public_architecture.py
python scripts/generate_public_architecture.py --check
python scripts/generate_public_results_chart.py
python scripts/generate_public_results_chart.py --check
```

两个生成器都会确定性生成 GitHub 深色/浅色主题 SVG。性能图读取 `benchmarks/results/public_results_snapshot.json`，解析并验证 Qwen/vLLM、runtime 和外部 baseline 的 canonical summaries；该 JSON 是公开展示快照，不是原始 benchmark dataset，canonical Markdown 始终是权威来源。

仓库结构检查：

```bash
python scripts/check_release.py
```

许可证与正式证据检查：

```bash
python scripts/check_release.py --require-clean --require-evidence --require-public
```

`--require-public` 要求根目录 `LICENSE`、`pyproject.toml` license metadata、`CITATION.cff` 与 README license section 一致；`--require-clean` 确保结果绑定 clean commit。该检查不要求升级版本或创建 tag。

Annotated version tag 检查：

```bash
python scripts/check_release.py --require-clean --require-evidence --require-tag
```

结构检查同时要求 GitHub workflow/issue/PR collaboration surface、研究问题、结果/复现入口、runner/validator/tests，以及 `constraints/flashinfer-cu128.txt` 的固定 Torch/Triton/FlashInfer/CUDA Python packages/Ninja pins。`--require-evidence` 额外要求 canonical summaries、multi-trial、complete-step profile 与性能报告；正式 tag 必须启用。

分层验证与仓库外结果导出：

```bash
python scripts/run_validation.py --phase all --dry-run

python scripts/run_validation.py \
  --phase trials-formal \
  --phase profile-formal \
  --export-dir "/path/outside/repository/flashdec_results"
```

可选 phase：`local`、`focused`、`full`、`trials-quick`、`trials-formal`、`profile-quick`、`profile-formal`、`release` 和 `all`。`all` 运行全部 evidence phase，但故意不运行 `release`；正式 summary 同步、审核并提交后，再单独执行 `--phase release`，检查 clean tree 和全部 evidence 文件。

GPU phase 会检查 `CUDA_HOME`、NVCC、tracked diff 和 untracked source；formal phase 禁止 `--allow-dirty`。每个命令成功后还会检查预期 CSV/Markdown 是否真实生成或更新，所有选定 phase 成功后才执行 `--export-dir` 复制。
