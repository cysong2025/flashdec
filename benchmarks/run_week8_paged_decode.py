"""Run Week 8 paged decode attention performance experiments."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import BenchmarkResult, benchmark_case, write_csv
from flashdec.perf import estimate_paged_decode_bytes, paged_decode_metric_metadata


BATCH_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXT_SWEEP = [128, 256, 512, 1024, 2048, 4096, 8192]
QUICK_BATCH_SWEEP = [1, 16, 64]
QUICK_CONTEXT_SWEEP = [128, 1024, 4096]


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _dtype_name(dtype):
    if dtype.__str__().endswith("bfloat16"):
        return "bfloat16"
    if dtype.__str__().endswith("float16"):
        return "float16"
    return str(dtype).replace("torch.", "")


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _make_shape_matrix(args):
    batches = QUICK_BATCH_SWEEP if args.quick else BATCH_SWEEP
    contexts = QUICK_CONTEXT_SWEEP if args.quick else CONTEXT_SWEEP

    cases = []
    for batch in batches:
        shape = (batch, args.num_q_heads, args.num_kv_heads, args.head_dim, args.batch_context)
        cases.append((f"batch_b{batch}_ctx{args.batch_context}", "batch", shape))
    for context in contexts:
        shape = (args.context_batch, args.num_q_heads, args.num_kv_heads, args.head_dim, context)
        cases.append((f"context_b{args.context_batch}_ctx{context}", "context", shape))

    deduped = []
    seen = set()
    for case_name, sweep, shape in cases:
        if shape in seen:
            continue
        seen.add(shape)
        deduped.append((case_name, sweep, shape))
    return deduped


def _common_metadata(torch, args, dtype_name, impl, case_name, sweep, shape, seq_lens, block_tables, cache):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    device = torch.cuda.current_device()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "benchmark": "week8_paged_decode",
        "experiment": getattr(args, "experiment", "num_warps"),
        "case": case_name,
        "sweep": sweep,
        "dtype": dtype_name,
        "impl": impl,
        "op": "paged_decode_attention",
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "min_seq_len": int(seq_lens.min().item()),
        "max_actual_seq_len": int(seq_lens.max().item()),
        "block_size": args.block_size,
        "max_blocks_per_seq": block_tables.shape[1],
        "used_blocks": cache.num_used_blocks,
        "validated": str(not args.skip_validate),
    }


def _with_metadata(result, extra_metadata):
    metadata = dict(result.metadata)
    metadata.update(extra_metadata)
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


def _with_speedup(result, ref_mean_ms):
    return _with_metadata(result, {"speedup_vs_ref": f"{ref_mean_ms / result.mean_ms:.4f}"})


def _with_perf_metrics(result, estimate):
    return _with_metadata(
        result,
        paged_decode_metric_metadata(estimate, mean_ms=result.mean_ms, p50_ms=result.p50_ms),
    )


def _make_inputs(torch, args, shape, dtype, case_index):
    from flashdec.cache import PagedKVCache

    torch.manual_seed(args.seed + case_index)
    torch.cuda.manual_seed_all(args.seed + case_index)

    num_seqs, _, num_kv_heads, head_dim, max_seq_len = shape
    request_ids = list(range(num_seqs))
    low = max(1, max_seq_len // 2)
    seq_lens = torch.randint(low, max_seq_len + 1, (num_seqs,), device="cuda", dtype=torch.int32)
    seq_lens_list = [int(value) for value in seq_lens.detach().cpu().tolist()]
    max_blocks = sum((seq_len + args.block_size - 1) // args.block_size for seq_len in seq_lens_list)
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=args.block_size,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )

    token_k = torch.randn((num_seqs, max_seq_len, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    token_v = torch.randn_like(token_k)
    for step in range(max_seq_len):
        active_rows = [row for row, seq_len in enumerate(seq_lens_list) if step < seq_len]
        if not active_rows:
            continue
        active_ids = [request_ids[row] for row in active_rows]
        cache.append(0, active_ids, token_k[active_rows, step], token_v[active_rows, step])

    num_q_heads = shape[1]
    q = torch.randn((num_seqs, num_q_heads, head_dim), device="cuda", dtype=dtype)
    block_tables = cache.block_tables(request_ids)
    seq_lens = cache.seq_lens_tensor(request_ids)
    return q, cache.k_cache[0], cache.v_cache[0], block_tables, seq_lens, cache


def _validate(torch, q, k_cache, v_cache, block_tables, seq_lens, block_size, num_warps):
    from flashdec.kernels.paged_decode import paged_decode_attention
    from flashdec.paged_reference import paged_decode_attention_ref

    actual = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_size=block_size,
        num_warps=num_warps,
    )
    expected = paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens)
    if q.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _make_estimate(shape, dtype_name, seq_lens, block_tables, block_size):
    seq_lens_list = [int(value) for value in seq_lens.detach().cpu().tolist()]
    num_seqs, num_q_heads, _, head_dim, _ = shape
    return estimate_paged_decode_bytes(
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        seq_lens=seq_lens_list,
        max_blocks_per_seq=block_tables.shape[1],
        block_size=block_size,
        dtype=dtype_name,
        block_table_entry_bytes=block_tables.element_size(),
        seq_len_entry_bytes=seq_lens.element_size(),
    )


def run_shape(torch, args, case_index, case_name, sweep, shape, dtype):
    from flashdec.kernels.paged_decode import paged_decode_attention
    from flashdec.paged_reference import paged_decode_attention_ref

    q, k_cache, v_cache, block_tables, seq_lens, cache = _make_inputs(torch, args, shape, dtype, case_index)
    dtype_name = _dtype_name(dtype)
    estimate = _make_estimate(shape, dtype_name, seq_lens, block_tables, args.block_size)
    results = []

    ref_mean_ms = None
    if args.mode == "all":
        ref_result = benchmark_case(
            "paged_decode_attention_ref",
            lambda: paged_decode_attention_ref(q, k_cache, v_cache, block_tables, seq_lens),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata=_common_metadata(torch, args, dtype_name, "torch_ref", case_name, sweep, shape, seq_lens, block_tables, cache),
        )
        ref_mean_ms = ref_result.mean_ms
        results.append(_with_speedup(_with_perf_metrics(ref_result, estimate), ref_result.mean_ms))

    for num_warps in args.num_warps:
        if not args.skip_validate:
            _validate(torch, q, k_cache, v_cache, block_tables, seq_lens, args.block_size, num_warps)
        triton_result = benchmark_case(
            "triton_paged_decode_attention",
            lambda num_warps=num_warps: paged_decode_attention(
                q,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                block_size=args.block_size,
                num_warps=num_warps,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            metadata={
                **_common_metadata(torch, args, dtype_name, "triton", case_name, sweep, shape, seq_lens, block_tables, cache),
                "num_warps": num_warps,
            },
        )
        triton_result = _with_perf_metrics(triton_result, estimate)
        if ref_mean_ms is not None:
            triton_result = _with_speedup(triton_result, ref_mean_ms)
        results.append(triton_result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="both")
    parser.add_argument("--mode", choices=["all", "triton"], default="triton")
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--batch-context", type=int, default=1024)
    parser.add_argument("--context-batch", type=int, default=16)
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--num-warps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--quick", action="store_true", help="Use a smaller shape matrix for fast smoke benchmarks.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip reference checks before timing each Triton config.")
    parser.add_argument("--output", default="benchmarks/results/week8_paged_decode_warps.csv")
    args = parser.parse_args()
    if args.num_q_heads % args.num_kv_heads != 0:
        parser.error("num_q_heads must be divisible by num_kv_heads")
    if any(not _is_power_of_two(value) for value in args.num_warps):
        parser.error("num_warps values must be positive powers of two")
    return args


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for paged decode benchmarks")

    if args.dtype == "both":
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        else:
            print("Skipping bfloat16 because torch.cuda.is_bf16_supported() is false.")
    else:
        dtypes = [_dtype_from_name(torch, args.dtype)]
        if dtypes[0] == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("bfloat16 was requested, but this CUDA device does not report BF16 support")

    results = []
    cases = _make_shape_matrix(args)
    for dtype in dtypes:
        for case_index, (case_name, sweep, shape) in enumerate(cases):
            results.extend(run_shape(torch, args, case_index, case_name, sweep, shape, dtype))

    write_csv(results, args.output)
    for result in results:
        print(result.as_row())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
