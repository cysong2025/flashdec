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

发布候选结构检查：

```bash
python scripts/check_release.py
```

正式 release commit/tag 阶段使用：

```bash
python scripts/check_release.py --require-clean --require-evidence --require-tag
```

`--require-evidence` 额外要求 multi-trial、complete-step profile 与最终 performance summary；candidate 开发阶段可以不启用，正式 tag 必须启用。

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
