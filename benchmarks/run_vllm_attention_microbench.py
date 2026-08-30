#!/usr/bin/env python3
"""Benchmark FlashDec against vLLM Triton attention on Qwen2.5 decode shapes."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import triton

from flashdec.vllm_backend import FlashDecAttentionImpl
from vllm import __version__ as vllm_version
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)


SCHEMA_VERSION = 1
MODEL_ID = "Qwen2.5-3B-Instruct"
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 16


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    context_len: int


CASES = (
    Case("qwen_b1_ctx128", 1, 128),
    Case("qwen_b1_ctx1024", 1, 1024),
    Case("qwen_b4_ctx1024", 4, 1024),
    Case("qwen_b8_ctx1024", 8, 1024),
    Case("qwen_b8_ctx2048", 8, 2048),
)


class _LayerScales:
    def __init__(self, device: torch.device):
        one = torch.ones((), device=device, dtype=torch.float32)
        self._q_scale = one
        self._k_scale = one
        self._v_scale = one


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _make_case(case: Case, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(20260830 + case.batch_size + case.context_len)
    torch.cuda.manual_seed_all(20260830 + case.batch_size + case.context_len)
    logical_blocks = math.ceil(case.context_len / BLOCK_SIZE)
    num_blocks = case.batch_size * logical_blocks
    query = torch.randn(
        (case.batch_size, NUM_Q_HEADS, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    kv_cache = torch.randn(
        (num_blocks, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).view(case.batch_size, logical_blocks)
    seq_lens = torch.full(
        (case.batch_size,), case.context_len, device=device, dtype=torch.int32
    )
    query_start_loc = torch.arange(
        case.batch_size + 1, device=device, dtype=torch.int32
    )
    seq_threshold_3d = 64
    num_segments = 16
    metadata = TritonAttentionMetadata(
        num_actual_tokens=case.batch_size,
        max_query_len=1,
        query_start_loc=query_start_loc,
        max_seq_len=case.context_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.empty(
            (case.batch_size,), device=device, dtype=torch.int64
        ),
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=torch.empty(
            (seq_threshold_3d, NUM_Q_HEADS, num_segments, HEAD_DIM),
            device=device,
            dtype=torch.float32,
        ),
        softmax_segm_max=torch.empty(
            (seq_threshold_3d, NUM_Q_HEADS, num_segments),
            device=device,
            dtype=torch.float32,
        ),
        softmax_segm_expsum=torch.empty(
            (seq_threshold_3d, NUM_Q_HEADS, num_segments),
            device=device,
            dtype=torch.float32,
        ),
        causal=True,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    return query, kv_cache, metadata


def _make_impl(cls):
    return cls(
        NUM_Q_HEADS,
        HEAD_DIM,
        HEAD_DIM**-0.5,
        NUM_KV_HEADS,
        None,
        None,
        "bfloat16",
    )


def _run_once(impl, layer, query, kv_cache, metadata, output):
    empty = query.new_empty((0,))
    return impl.forward(
        layer,
        query,
        empty,
        empty,
        kv_cache,
        metadata,
        output,
    )


def _measure(fn, warmup_ms: int, repeat_ms: int) -> tuple[float, float, float]:
    # Time-window warmup is important for sub-0.1 ms kernels: a small count of
    # synchronized launches does not reliably raise the GPU from an idle
    # P-state. Triton's helper batches launches and returns event-based
    # quantiles after the requested warmup/measurement windows.
    p50, p90, p99 = triton.testing.do_bench(
        fn,
        warmup=float(warmup_ms),
        rep=float(repeat_ms),
        quantiles=[0.50, 0.90, 0.99],
    )
    return float(p50), float(p90), float(p99)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--warmup", type=int, default=100, help="Warmup window in milliseconds."
    )
    parser.add_argument(
        "--repeat", type=int, default=500, help="Measurement window in milliseconds."
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Run only selected cases; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.trials <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("trials/repeat must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    selected = [case for case in CASES if not args.case or case.name in args.case]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layer = _LayerScales(device)
    commit = _git_value("rev-parse", "HEAD")
    worktree_clean = _git_value("status", "--porcelain") == ""
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    device_name = torch.cuda.get_device_name(device)
    rows = []

    for case in selected:
        query, kv_cache, metadata = _make_case(case, dtype, device)
        native_impl = _make_impl(TritonAttentionImpl)
        flashdec_impl = _make_impl(FlashDecAttentionImpl)
        native_output = torch.empty_like(query)
        flashdec_output = torch.empty_like(query)

        _run_once(native_impl, layer, query, kv_cache, metadata, native_output)
        _run_once(flashdec_impl, layer, query, kv_cache, metadata, flashdec_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            flashdec_output, native_output, rtol=3e-2, atol=3e-2
        )

        backends = (
            (
                "vllm_triton_attn",
                lambda: _run_once(
                    native_impl, layer, query, kv_cache, metadata, native_output
                ),
            ),
            (
                "flashdec",
                lambda: _run_once(
                    flashdec_impl, layer, query, kv_cache, metadata, flashdec_output
                ),
            ),
        )
        for trial in range(1, args.trials + 1):
            ordered = backends if trial % 2 else tuple(reversed(backends))
            for backend, fn in ordered:
                p50, p90, p99 = _measure(fn, args.warmup, args.repeat)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "started_at": started_at,
                    "git_commit": commit,
                    "git_worktree_clean": worktree_clean,
                    "device": device_name,
                    "torch_version": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "triton_version": triton.__version__,
                    "vllm_version": vllm_version,
                    "model_id": MODEL_ID,
                    "dtype": str(dtype).removeprefix("torch."),
                    "case": case.name,
                    "batch_size": case.batch_size,
                    "context_len": case.context_len,
                    "num_q_heads": NUM_Q_HEADS,
                    "num_kv_heads": NUM_KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "block_size": BLOCK_SIZE,
                    "backend": backend,
                    "trial": trial,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "p50_ms": f"{p50:.6f}",
                    "p90_ms": f"{p90:.6f}",
                    "p99_ms": f"{p99:.6f}",
                    "sequences_per_s": f"{case.batch_size * 1000.0 / p50:.3f}",
                    "correctness": "PASS",
                }
                rows.append(row)
                print(
                    case.name,
                    backend,
                    trial,
                    row["p50_ms"],
                    row["sequences_per_s"],
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
