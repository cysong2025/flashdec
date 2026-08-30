#!/usr/bin/env python3
"""Run paired vLLM latency processes on frozen Qwen2.5-3B workloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import triton
import vllm


SCHEMA_VERSION = 1
MODEL_ID = "Qwen2.5-3B-Instruct"
BACKENDS = (
    ("vllm_triton_attn", "TRITON_ATTN"),
    ("flashdec", "CUSTOM"),
)


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    input_len: int
    output_len: int


CASES = (
    Case("qwen_b8_i128_o128", 8, 128, 128),
    Case("qwen_b8_i2048_o128", 8, 2048, 128),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Run only selected cases; may be repeated.",
    )
    return parser.parse_args()


def _validate_environment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.trials <= 0 or args.warmup_iters < 0 or args.num_iters <= 0:
        raise ValueError("trials/num-iters must be positive and warmup non-negative")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("set VLLM_USE_FLASHINFER_SAMPLER=0")
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1":
        raise RuntimeError("set VLLM_WSL2_ENABLE_PIN_MEMORY=1")
    if os.environ.get("VLLM_PLUGINS") != "flashdec":
        raise RuntimeError("set VLLM_PLUGINS=flashdec")
    if args.require_clean and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean needs a clean Git worktree")

    model_config = args.model / "config.json"
    model_manifest = args.model / "SHA256SUMS"
    if not model_config.is_file() or not model_manifest.is_file():
        raise FileNotFoundError("model config.json and SHA256SUMS are required")
    vllm_executable = Path(sys.executable).with_name("vllm")
    if not vllm_executable.is_file():
        raise FileNotFoundError(f"vLLM CLI not found: {vllm_executable}")
    return model_config, model_manifest, vllm_executable


def _command(
    *,
    executable: Path,
    model: Path,
    case: Case,
    backend: str,
    warmup_iters: int,
    num_iters: int,
    gpu_memory_utilization: float,
    output_json: Path,
) -> list[str]:
    return [
        str(executable),
        "bench",
        "latency",
        "--model",
        str(model),
        "--tokenizer",
        str(model),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--attention-backend",
        backend,
        "--input-len",
        str(case.input_len),
        "--output-len",
        str(case.output_len),
        "--batch-size",
        str(case.batch_size),
        "--n",
        "1",
        "--num-iters-warmup",
        str(warmup_iters),
        "--num-iters",
        str(num_iters),
        "--seed",
        "20260830",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--disable-detokenize",
        "--no-enforce-eager",
        "--output-json",
        str(output_json),
    ]


def main() -> None:
    args = _parse_args()
    model_config, model_manifest, executable = _validate_environment(args)
    selected = [case for case in CASES if not args.case or case.name in args.case]
    raw_dir = args.output.parent / f"{args.output.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = _git_value("rev-parse", "HEAD")
    worktree_clean = _git_value("status", "--porcelain") == ""
    model_path = args.model.resolve()
    common = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "git_commit": commit,
        "git_worktree_clean": worktree_clean,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_version": triton.__version__,
        "vllm_version": vllm.__version__,
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "model_config_sha256": _sha256(model_config),
        "model_manifest_sha256": _sha256(model_manifest),
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "max_model_len": 4096,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "compilation_mode": "default_inductor_cudagraph",
        "flashdec_num_splits": os.environ.get(
            "FLASHDEC_VLLM_NUM_SPLITS", "auto"
        ),
        "warmup_iters": args.warmup_iters,
        "num_iters": args.num_iters,
    }
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = "20260830"
    rows: list[dict[str, object]] = []

    for case in selected:
        for trial in range(1, args.trials + 1):
            ordered = BACKENDS if trial % 2 else tuple(reversed(BACKENDS))
            for run_order, (backend_name, backend_arg) in enumerate(ordered, 1):
                stem = f"{case.name}_trial{trial}_{backend_name}"
                output_json = raw_dir / f"{stem}.json"
                log_path = raw_dir / f"{stem}.log"
                command = _command(
                    executable=executable,
                    model=model_path,
                    case=case,
                    backend=backend_arg,
                    warmup_iters=args.warmup_iters,
                    num_iters=args.num_iters,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    output_json=output_json,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        env=child_env,
                        text=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"vLLM benchmark failed ({completed.returncode}); see {log_path}"
                    )
                result = json.loads(output_json.read_text(encoding="utf-8"))
                latencies = [float(value) for value in result["latencies"]]
                if len(latencies) != args.num_iters or any(
                    value <= 0 for value in latencies
                ):
                    raise ValueError(f"invalid latency samples in {output_json}")
                p50_s = float(result["percentiles"]["50"])
                p90_s = float(result["percentiles"]["90"])
                row = {
                    **common,
                    "case": case.name,
                    "batch_size": case.batch_size,
                    "input_len": case.input_len,
                    "output_len": case.output_len,
                    "backend": backend_name,
                    "backend_arg": backend_arg,
                    "trial": trial,
                    "run_order": run_order,
                    "avg_latency_ms": f"{float(result['avg_latency']) * 1000.0:.6f}",
                    "p50_latency_ms": f"{p50_s * 1000.0:.6f}",
                    "p90_latency_ms": f"{p90_s * 1000.0:.6f}",
                    "output_tokens_per_s": (
                        f"{case.batch_size * case.output_len / p50_s:.3f}"
                    ),
                    "raw_json": str(output_json),
                    "log": str(log_path),
                    "command": shlex.join(command),
                }
                rows.append(row)
                print(
                    case.name,
                    backend_name,
                    trial,
                    row["p50_latency_ms"],
                    row["output_tokens_per_s"],
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
