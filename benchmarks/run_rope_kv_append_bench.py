"""Benchmark torch, CUDA-append, and fused RoPE + paged KV append paths."""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, benchmark_case, write_csv


CASES = {
    "default": (16, 32, 8, 128, 1024),
    "large_batch": (64, 32, 8, 128, 1024),
    "long_context": (16, 32, 8, 128, 4096),
}
BACKENDS = ("torch", "cuda", "fused_cuda")


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _requested_dtypes(torch, dtype_name):
    names = ("float16", "bfloat16") if dtype_name == "both" else (dtype_name,)
    dtypes = []
    for name in names:
        dtype = _dtype_from_name(torch, name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 benchmark requested but this CUDA device does not support BF16")
        dtypes.append((name, dtype))
    return dtypes


def _assert_close(torch, actual, expected, dtype, fused=False):
    if fused:
        if dtype == torch.float32:
            tolerance = 2e-5
        elif dtype == torch.float16:
            tolerance = 3e-3
        else:
            tolerance = 2e-2
    else:
        tolerance = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 0.0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _make_cache(torch, shape, dtype, block_size, extra_steps):
    from flashdec.cache import PagedKVCache

    num_seqs, _, num_kv_heads, head_dim, initial_context = shape
    total_tokens = initial_context + extra_steps + 1
    max_blocks = num_seqs * ((total_tokens + block_size - 1) // block_size)
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )
    request_ids = list(range(num_seqs))
    prefill_k = torch.zeros((num_seqs, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    prefill_v = torch.zeros_like(prefill_k)
    for _ in range(initial_context):
        cache.append(0, request_ids, prefill_k, prefill_v)
    torch.cuda.synchronize()
    return cache, request_ids


def _step_inputs(torch, shape, dtype, seed):
    num_seqs, num_q_heads, num_kv_heads, head_dim, _ = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn((num_seqs, num_q_heads, head_dim), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((num_seqs, num_kv_heads, head_dim), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn((num_seqs, num_kv_heads, head_dim), device="cuda", dtype=dtype, generator=generator)
    return q, k, v


def _preload_backend(backend):
    if backend == "cuda":
        from flashdec import load_cuda_kv_append_extension

        load_cuda_kv_append_extension()
    elif backend == "fused_cuda":
        from flashdec import load_fused_rope_kv_append_extension

        load_fused_rope_kv_append_extension()


def _validate_case(torch, shape, dtype, block_size, seed):
    from flashdec.rope import rope_paged_kv_append

    q, k, v = _step_inputs(torch, shape, dtype, seed)
    reference_cache, request_ids = _make_cache(torch, shape, dtype, block_size, extra_steps=1)
    expected = rope_paged_kv_append(reference_cache, 0, request_ids, q, k, v, append_backend="torch")

    for backend in ("cuda", "fused_cuda"):
        native_cache, native_ids = _make_cache(torch, shape, dtype, block_size, extra_steps=1)
        actual = rope_paged_kv_append(native_cache, 0, native_ids, q, k, v, append_backend=backend)
        _assert_close(torch, actual.q, expected.q, dtype, fused=backend == "fused_cuda")
        torch.testing.assert_close(actual.positions, expected.positions)
        torch.testing.assert_close(actual.block_tables, expected.block_tables)
        torch.testing.assert_close(actual.seq_lens, expected.seq_lens)
        torch.cuda.synchronize()
        _assert_close(torch, native_cache.k_cache, reference_cache.k_cache, dtype, fused=backend == "fused_cuda")
        _assert_close(torch, native_cache.v_cache, reference_cache.v_cache, dtype, fused=backend == "fused_cuda")
        if native_cache.metrics() != reference_cache.metrics() or not native_cache.validate_invariants():
            raise RuntimeError(f"{backend} cache metadata diverged from torch reference")


def _metadata(torch, case_name, shape, dtype_name, backend, block_size, warmup, repeat):
    num_seqs, num_q_heads, num_kv_heads, head_dim, initial_context = shape
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "op": "rope_paged_kv_append",
        "impl": backend,
        "case": case_name,
        "dtype": dtype_name,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "initial_context": initial_context,
        "block_size": block_size,
        "warmup": warmup,
        "repeat": repeat,
        "validated": True,
        "timing_scope": "CUDA-event GPU work; excludes JIT build and prefill",
    }


def _with_speedup(result, torch_p50_ms):
    metadata = dict(result.metadata)
    metadata["speedup_vs_torch_p50"] = f"{torch_p50_ms / result.p50_ms:.4f}"
    return BenchmarkResult(
        name=result.name,
        mean_ms=result.mean_ms,
        p50_ms=result.p50_ms,
        p90_ms=result.p90_ms,
        min_ms=result.min_ms,
        max_ms=result.max_ms,
        repeats=result.repeats,
        metadata=metadata,
    )


def _run_case(torch, args, case_name, shape, dtype_name, dtype):
    from flashdec.rope import rope_paged_kv_append

    _preload_backend("cuda")
    _preload_backend("fused_cuda")
    _validate_case(torch, shape, dtype, args.block_size, args.seed)
    q, k, v = _step_inputs(torch, shape, dtype, args.seed)
    results = []

    for backend in BACKENDS:
        cache, request_ids = _make_cache(
            torch,
            shape,
            dtype,
            args.block_size,
            extra_steps=args.warmup + args.repeat,
        )
        result = benchmark_case(
            f"rope_paged_kv_append_{backend}",
            lambda: rope_paged_kv_append(cache, 0, request_ids, q, k, v, append_backend=backend),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata=_metadata(
                torch,
                case_name,
                shape,
                dtype_name,
                backend,
                args.block_size,
                args.warmup,
                args.repeat,
            ),
        )
        results.append(result)

    torch_result = results[0]
    return [_with_speedup(result, torch_result.p50_ms) for result in results]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--block-size", type=int, choices=[32], default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=251)
    parser.add_argument("--quick", action="store_true", help="Run only default with at most warmup=5, repeat=20.")
    parser.add_argument("--output", default="benchmarks/results/week11_rope_kv_append.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for RoPE + KV append benchmarks")
    if args.quick:
        args.warmup = min(args.warmup, 5)
        args.repeat = min(args.repeat, 20)
    case_names = [args.case] if args.case != "all" else (["default"] if args.quick else list(CASES))

    results = []
    for dtype_name, dtype in _requested_dtypes(torch, args.dtype):
        for case_name in case_names:
            results.extend(_run_case(torch, args, case_name, CASES[case_name], dtype_name, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
