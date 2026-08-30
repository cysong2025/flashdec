#!/usr/bin/env python3
"""Benchmark fused FlashDec KV append + attention against the vLLM baseline.

The common timed operation is one current-token KV-cache append followed by
single-token decode attention.  The native path calls
``TritonAttentionImpl.do_kv_cache_update`` and then ``forward``; the FlashDec
path performs the same append inside its fused ``forward`` implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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


SCHEMA_VERSION = 2
MODEL_ID = "Qwen2.5-3B-Instruct"
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 16
INPUT_SEED_BASE = 20260830
COMPARISON_SCOPE = "current_token_kv_append_plus_single_token_decode_attention"
CACHE_STATE_POLICY = (
    "paired_deterministic_snapshot_reset_per_trial_idempotent_append"
)
TIMED_OPERATIONS = {
    "vllm_triton_attn": "do_kv_cache_update_then_forward",
    "flashdec": "fused_kv_append_forward",
}


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
        self.kv_sharing_target_layer_name = None


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _case_seed(case: Case) -> int:
    return INPUT_SEED_BASE + case.batch_size + case.context_len


def _append_slot_indices(case: Case) -> tuple[int, ...]:
    """Return physical slots for each request's current (last) token."""
    logical_blocks = math.ceil(case.context_len / BLOCK_SIZE)
    current_logical_block = (case.context_len - 1) // BLOCK_SIZE
    token_offset = (case.context_len - 1) % BLOCK_SIZE
    return tuple(
        (request * logical_blocks + current_logical_block) * BLOCK_SIZE
        + token_offset
        for request in range(case.batch_size)
    )


def _make_case(case: Case, dtype: torch.dtype, device: torch.device):
    seed = _case_seed(case)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logical_blocks = math.ceil(case.context_len / BLOCK_SIZE)
    num_blocks = case.batch_size * logical_blocks
    query = torch.randn(
        (case.batch_size, NUM_Q_HEADS, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        (case.batch_size, NUM_KV_HEADS, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        (case.batch_size, NUM_KV_HEADS, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    initial_kv_cache = torch.randn(
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
    append_slots = _append_slot_indices(case)
    slot_mapping = torch.tensor(append_slots, device=device, dtype=torch.int64)
    # The append destination is logically unallocated before this decode step.
    # Zeroing it makes the physical starting snapshot deterministic and also
    # ensures the correctness check proves that both paths really write K/V.
    for slot in append_slots:
        block_index, token_offset = divmod(slot, BLOCK_SIZE)
        initial_kv_cache[block_index, :, token_offset].zero_()
    seq_threshold_3d = 64
    num_segments = 16
    metadata = TritonAttentionMetadata(
        num_actual_tokens=case.batch_size,
        max_query_len=1,
        query_start_loc=query_start_loc,
        max_seq_len=case.context_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=torch.zeros(
            (seq_threshold_3d, NUM_Q_HEADS, num_segments, HEAD_DIM),
            device=device,
            dtype=torch.float32,
        ),
        softmax_segm_max=torch.zeros(
            (seq_threshold_3d, NUM_Q_HEADS, num_segments),
            device=device,
            dtype=torch.float32,
        ),
        softmax_segm_expsum=torch.zeros(
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
    return query, key, value, initial_kv_cache, metadata


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


def _run_native_once(
    impl, layer, query, key, value, kv_cache, metadata, output
):
    impl.do_kv_cache_update(
        layer,
        key,
        value,
        kv_cache,
        metadata.slot_mapping,
    )
    return impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )


def _run_flashdec_once(
    impl, layer, query, key, value, kv_cache, metadata, output
):
    return impl.forward(
        layer,
        query,
        key,
        value,
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
    print(f"Comparison scope: {COMPARISON_SCOPE}", flush=True)
    print(
        f"Timed paths: native={TIMED_OPERATIONS['vllm_triton_attn']}; "
        f"flashdec={TIMED_OPERATIONS['flashdec']}",
        flush=True,
    )
    print(f"Cache state policy: {CACHE_STATE_POLICY}", flush=True)

    for case in selected:
        query, key, value, initial_kv_cache, metadata = _make_case(
            case, dtype, device
        )
        native_impl = _make_impl(TritonAttentionImpl)
        flashdec_impl = _make_impl(FlashDecAttentionImpl)
        native_output = torch.empty_like(query)
        flashdec_output = torch.empty_like(query)
        native_kv_cache = initial_kv_cache.clone()
        flashdec_kv_cache = initial_kv_cache.clone()

        for trial in range(1, args.trials + 1):
            # Each paired trial starts from byte-identical, deterministic cache
            # snapshots.  The untimed correctness call writes the current K/V
            # once; subsequent warmup and measured writes are idempotent, so no
            # cache state drifts between timed iterations.
            native_kv_cache.copy_(initial_kv_cache)
            flashdec_kv_cache.copy_(initial_kv_cache)
            _run_native_once(
                native_impl,
                layer,
                query,
                key,
                value,
                native_kv_cache,
                metadata,
                native_output,
            )
            _run_flashdec_once(
                flashdec_impl,
                layer,
                query,
                key,
                value,
                flashdec_kv_cache,
                metadata,
                flashdec_output,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                flashdec_output, native_output, rtol=3e-2, atol=3e-2
            )
            torch.testing.assert_close(
                flashdec_kv_cache, native_kv_cache, rtol=0.0, atol=0.0
            )

            backends = (
                (
                    "vllm_triton_attn",
                    lambda: _run_native_once(
                        native_impl,
                        layer,
                        query,
                        key,
                        value,
                        native_kv_cache,
                        metadata,
                        native_output,
                    ),
                ),
                (
                    "flashdec",
                    lambda: _run_flashdec_once(
                        flashdec_impl,
                        layer,
                        query,
                        key,
                        value,
                        flashdec_kv_cache,
                        metadata,
                        flashdec_output,
                    ),
                ),
            )
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
                    "comparison_scope": COMPARISON_SCOPE,
                    "timed_operation": TIMED_OPERATIONS[backend],
                    "cache_state_policy": CACHE_STATE_POLICY,
                    "input_seed": _case_seed(case),
                    "flashdec_num_splits": os.environ.get(
                        "FLASHDEC_VLLM_NUM_SPLITS", "auto"
                    ),
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
