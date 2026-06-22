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
- PyTorch 是否安装。
- Triton 是否安装。
- CUDA 是否可用。
- GPU 型号。
- `nvidia-smi`。
- `nvcc --version`。
