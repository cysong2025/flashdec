# Scripts

这里放本地开发辅助脚本。

第一步先运行：

```bash
python scripts/check_env.py
```

然后把关键输出粘贴到：

```text
docs/environment.md
```

这个脚本会检查：

- Python 版本。
- Python executable。
- PyTorch 是否安装。
- Triton 是否安装。
- Ninja 是否安装。
- CUDA 是否可用。
- GPU 型号。
- 当前 Git commit 与 worktree 是否 clean。
- 环境变量/PyTorch 检测到的 `CUDA_HOME`。
- `nvidia-smi`。
- `nvcc` path。
- `nvcc --version`。

文档链接检查：

```bash
python scripts/check_docs.py
```

该脚本扫描 README、CHANGELOG、CONTRIBUTING、`docs/`、`benchmarks/`、`scripts/` 和 `.github/` 中的 canonical Markdown 链接，拒绝缺失目标、逃逸出仓库的相对路径，以及与项目实现无关的个人评估措辞；外部 URL 不发起网络请求。`benchmarks/results/` 下由 Git 忽略的 `*_quick_summary.md`、`*_smoke.md` 和 `local_backups/` 不属于交付文档，因此不进入扫描。

发布候选结构检查：

```bash
python scripts/check_release.py
```

正式 release commit/tag 阶段使用：

```bash
python scripts/check_release.py --require-clean --require-evidence --require-tag
```

结构检查同时要求 GitHub workflow/issue/PR collaboration surface、完整 R1 scheduled-workload surface、交付/结果/复现入口、R1–R5 runner/validator/tests，以及 `constraints/r5-cu128.txt` 的预注册 Torch/Triton/FlashInfer/CUDA Python packages/Ninja pins。`--require-evidence` 额外要求 R1–R5 canonical summaries、multi-trial、complete-step profile 与最终 performance summary；candidate 开发阶段可以不启用，正式 tag 必须启用。

R0 分阶段验证与 Windows 结果导出：

```bash
python scripts/run_r0_validation.py --phase all --dry-run

python scripts/run_r0_validation.py \
  --phase trials-formal \
  --phase profile-formal \
  --export-dir /mnt/c/Users/user/flashdec_results
```

可选 phase：`local`、`focused`、`full`、`trials-quick`、`trials-formal`、`profile-quick`、`profile-formal`、`release` 和 `all`。`all` 运行全部 evidence phase，但故意不运行 `release`；正式 summary 同步、审核并提交后，再单独执行 `--phase release`，检查 clean tree 和全部 evidence 文件。

GPU phase 会检查 `CUDA_HOME`、NVCC、tracked diff 和 untracked source；formal phase 禁止 `--allow-dirty`。每个命令成功后还会检查预期 CSV/Markdown 是否真实生成或更新，所有选定 phase 成功后才执行 `--export-dir` 复制。
