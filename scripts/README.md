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
