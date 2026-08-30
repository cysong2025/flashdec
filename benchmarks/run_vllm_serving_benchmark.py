#!/usr/bin/env python3
"""Run paired online-serving trials for vLLM Triton and FlashDec attention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
import triton
import vllm

from flashdec.benchmark import (
    vllm_cache_root_for_commit,
    write_vllm_cache_log_metadata,
)


SCHEMA_VERSION = 2
MODEL_ID = "Qwen2.5-3B-Instruct"
SERVED_MODEL_NAME = "qwen2.5-3b-instruct"
BACKENDS = (
    ("vllm_triton_attn", "TRITON_ATTN"),
    ("flashdec", "CUSTOM"),
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
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--num-warmups", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--port", type=int, default=8127)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument(
        "--vllm-cache-base",
        type=Path,
        help=(
            "Base directory for commit-scoped vLLM caches; defaults to "
            "FLASHDEC_VLLM_CACHE_BASE, then VLLM_CACHE_ROOT, then "
            "~/.cache/vllm-flashdec"
        ),
    )
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def _validate_environment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    positive = (
        args.trials,
        args.num_prompts,
        args.input_len,
        args.output_len,
        args.max_concurrency,
        args.port,
    )
    if any(value <= 0 for value in positive) or args.num_warmups < 0:
        raise ValueError("counts, lengths, concurrency, and port must be positive")
    if args.port > 65535:
        raise ValueError("port must be <= 65535")
    if args.max_concurrency > 8:
        raise ValueError("frozen server configuration supports concurrency <= 8")
    if args.input_len + args.output_len > 8192:
        raise ValueError("input_len + output_len must not exceed 8192")
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
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=0.25):
            raise RuntimeError(f"port {args.port} is already in use")
    except OSError:
        pass

    model_config = args.model / "config.json"
    model_manifest = args.model / "SHA256SUMS"
    if not model_config.is_file() or not model_manifest.is_file():
        raise FileNotFoundError("model config.json and SHA256SUMS are required")
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise FileNotFoundError(f"vLLM CLI not found: {executable}")
    return model_config, model_manifest, executable


def _server_command(
    executable: Path,
    model: Path,
    backend: str,
    port: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        str(executable),
        "serve",
        str(model),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--generation-config",
        "vllm",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--kv-cache-dtype",
        "bfloat16",
        "--seed",
        "20260830",
        "--attention-backend",
        backend,
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-enforce-eager",
        "--disable-log-stats",
    ]


def _benchmark_command(
    executable: Path,
    model: Path,
    args: argparse.Namespace,
    *,
    trial: int,
    backend_name: str,
    git_commit: str,
    vllm_cache_root: Path,
    output_json: Path,
) -> list[str]:
    return [
        str(executable),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        str(model),
        "--tokenizer",
        str(model),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--dataset-name",
        "random",
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        "inf",
        "--burstiness",
        "1",
        "--max-concurrency",
        str(args.max_concurrency),
        "--num-warmups",
        str(args.num_warmups),
        "--ready-check-timeout-sec",
        "30",
        "--seed",
        str(20260830 + trial),
        "--temperature",
        "0",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--metadata",
        f"git_commit={git_commit}",
        f"vllm_cache_root={vllm_cache_root}",
        f"attention_backend={backend_name}",
        f"trial={trial}",
        "--label",
        "r7_qwen_serving",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(output_json.parent),
        "--result-filename",
        output_json.name,
    ]


def _wait_ready(process: subprocess.Popen[str], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error = "not contacted"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM server exited early with code {returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready: {last_error}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=15)


def _run_pair_member(
    *,
    executable: Path,
    model: Path,
    args: argparse.Namespace,
    trial: int,
    run_order: int,
    backend_name: str,
    backend_arg: str,
    raw_dir: Path,
    common: dict[str, object],
    child_env: dict[str, str],
) -> dict[str, object]:
    stem = f"trial{trial}_{backend_name}"
    server_log_path = raw_dir / f"{stem}_server.log"
    benchmark_log_path = raw_dir / f"{stem}_benchmark.log"
    output_json = raw_dir / f"{stem}.json"
    server_command = _server_command(
        executable,
        model,
        backend_arg,
        args.port,
        args.gpu_memory_utilization,
    )
    benchmark_command = _benchmark_command(
        executable,
        model,
        args,
        trial=trial,
        backend_name=backend_name,
        git_commit=str(common["git_commit"]),
        vllm_cache_root=Path(str(common["vllm_cache_root"])),
        output_json=output_json,
    )

    with server_log_path.open("w", encoding="utf-8") as server_log:
        write_vllm_cache_log_metadata(
            server_log,
            commit=common["git_commit"],
            cache_root=common["vllm_cache_root"],
            command=shlex.join(server_command),
        )
        process = subprocess.Popen(
            server_command,
            env=child_env,
            text=True,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(process, args.port, timeout=240.0)
            with benchmark_log_path.open("w", encoding="utf-8") as benchmark_log:
                write_vllm_cache_log_metadata(
                    benchmark_log,
                    commit=common["git_commit"],
                    cache_root=common["vllm_cache_root"],
                    command=shlex.join(benchmark_command),
                )
                completed = subprocess.run(
                    benchmark_command,
                    env=child_env,
                    text=True,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"serving benchmark failed ({completed.returncode}); "
                    f"see {benchmark_log_path}"
                )
        finally:
            _stop_server(process)
    time.sleep(2.0)

    result = json.loads(output_json.read_text(encoding="utf-8"))
    if int(result["completed"]) != args.num_prompts or int(result["failed"]) != 0:
        raise ValueError(f"incomplete serving result: {output_json}")
    required_metrics = (
        "median_ttft_ms",
        "p90_ttft_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "median_itl_ms",
        "median_e2el_ms",
        "output_throughput",
    )
    if any(float(result[name]) <= 0 for name in required_metrics):
        raise ValueError(f"invalid serving metric in {output_json}")
    return {
        **common,
        "case": f"qwen_c{args.max_concurrency}_i{args.input_len}_o{args.output_len}",
        "backend": backend_name,
        "backend_arg": backend_arg,
        "trial": trial,
        "run_order": run_order,
        "completed": result["completed"],
        "failed": result["failed"],
        "median_ttft_ms": f"{float(result['median_ttft_ms']):.6f}",
        "p90_ttft_ms": f"{float(result['p90_ttft_ms']):.6f}",
        "median_tpot_ms": f"{float(result['median_tpot_ms']):.6f}",
        "p90_tpot_ms": f"{float(result['p90_tpot_ms']):.6f}",
        "median_itl_ms": f"{float(result['median_itl_ms']):.6f}",
        "p90_itl_ms": f"{float(result['p90_itl_ms']):.6f}",
        "median_e2el_ms": f"{float(result['median_e2el_ms']):.6f}",
        "p90_e2el_ms": f"{float(result['p90_e2el_ms']):.6f}",
        "request_throughput": f"{float(result['request_throughput']):.6f}",
        "output_throughput": f"{float(result['output_throughput']):.6f}",
        "total_token_throughput": f"{float(result['total_token_throughput']):.6f}",
        "raw_json": str(output_json),
        "server_log": str(server_log_path),
        "benchmark_log": str(benchmark_log_path),
        "server_command": shlex.join(server_command),
        "benchmark_command": shlex.join(benchmark_command),
    }


def main() -> None:
    args = _parse_args()
    model_config, model_manifest, executable = _validate_environment(args)
    raw_dir = args.output.parent / f"{args.output.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.model.resolve()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = _git_value("rev-parse", "HEAD")
    cache_root = vllm_cache_root_for_commit(
        commit,
        cache_base=args.vllm_cache_base,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "git_commit": commit,
        "git_worktree_clean": _git_value("status", "--porcelain") == "",
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
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "compilation_mode": "default_inductor_cudagraph",
        "vllm_cache_root": str(cache_root),
        "flashdec_num_splits": os.environ.get(
            "FLASHDEC_VLLM_NUM_SPLITS", "auto"
        ),
        "num_prompts": args.num_prompts,
        "num_warmups": args.num_warmups,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "max_concurrency": args.max_concurrency,
        "request_rate": "inf",
        "prefix_caching": False,
    }
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = "20260830"
    child_env["VLLM_CACHE_ROOT"] = str(cache_root)
    rows: list[dict[str, object]] = []
    print(f"VLLM_CACHE_ROOT={cache_root}", flush=True)

    for trial in range(1, args.trials + 1):
        ordered = BACKENDS if trial % 2 else tuple(reversed(BACKENDS))
        for run_order, (backend_name, backend_arg) in enumerate(ordered, 1):
            row = _run_pair_member(
                executable=executable,
                model=model_path,
                args=args,
                trial=trial,
                run_order=run_order,
                backend_name=backend_name,
                backend_arg=backend_arg,
                raw_dir=raw_dir,
                common=common,
                child_env=child_env,
            )
            rows.append(row)
            print(
                row["case"],
                backend_name,
                trial,
                row["median_tpot_ms"],
                row["output_throughput"],
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
