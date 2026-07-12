"""Print the local environment needed for FlashDec development."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"not available ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__version__", "unknown"))


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return "not found"
    return result.stdout.strip()


def main() -> None:
    print("FlashDec environment check")
    print("==========================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {_module_version('torch')}")
    print(f"Triton: {_module_version('triton')}")
    print(f"Ninja: {_module_version('ninja')}")
    print(f"Git commit: {_run(['git', 'rev-parse', '--short', 'HEAD'])}")
    git_status = _run(["git", "status", "--porcelain"])
    print(f"Git worktree clean: {git_status == ''}")
    print(f"CUDA_HOME env: {os.environ.get('CUDA_HOME')}")
    print(f"nvcc path: {shutil.which('nvcc')}")
    print()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"torch import failed: {type(exc).__name__}: {exc}")
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA: {getattr(torch.version, 'cuda', None)}")
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"PyTorch CUDA_HOME detection failed: {type(exc).__name__}: {exc}")
        else:
            print(f"PyTorch CUDA_HOME: {CUDA_HOME}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                memory_gib = props.total_memory / 1024**3
                capability = f"{props.major}.{props.minor}"
                print(
                    f"GPU {index}: {props.name}, "
                    f"{memory_gib:.2f} GiB, sm_{capability}"
                )
    print()

    print("nvidia-smi")
    print("----------")
    print(_run(["nvidia-smi"]))
    print()

    print("nvcc --version")
    print("--------------")
    print(_run(["nvcc", "--version"]))


if __name__ == "__main__":
    main()
