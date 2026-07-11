"""Profile paged decode attention with PyTorch profiler."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import benchmark_case
from flashdec.perf import estimate_paged_decode_bytes, paged_decode_metric_metadata


CASES = {
    "small": ("small_b1_ctx128", (1, 32, 8, 128, 128)),
    "medium": ("medium_b16_ctx1024", (16, 32, 8, 128, 1024)),
    "large": ("large_b16_ctx8192", (16, 32, 8, 128, 8192)),
    "large_batch": ("large_batch_b64_ctx4096", (64, 32, 8, 128, 4096)),
}


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


def _dtype_names(name):
    if name == "both":
        return ["float16", "bfloat16"]
    return [name]


def _selected_cases(case):
    if case == "all":
        return list(CASES.values())
    return [CASES[case]]


def _impls(name):
    if name == "both":
        return ["triton", "ref"]
    return [name]


def _parse_num_stages(value):
    if value == "default":
        return None
    try:
        num_stages = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("num_stages must be default or an integer from 1 to 4") from exc
    if num_stages not in (1, 2, 3, 4):
        raise argparse.ArgumentTypeError("num_stages must be default or an integer from 1 to 4")
    return num_stages


def _num_stages_label(num_stages):
    return "default" if num_stages is None else str(num_stages)


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
    k_cache = cache.k_cache[0]
    v_cache = cache.v_cache[0]
    if args.kv_layout == "dim_major":
        k_cache = k_cache.permute(0, 1, 3, 2).contiguous()
        v_cache = v_cache.permute(0, 1, 3, 2).contiguous()
    return q, k_cache, v_cache, block_tables, seq_lens, cache


def _estimate(shape, dtype_name, seq_lens, block_tables, block_size):
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


def _validate(
    torch,
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    block_size,
    num_warps,
    kv_layout,
    num_stages=None,
):
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
        num_stages=num_stages,
        kv_layout=kv_layout,
    )
    expected = paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        kv_layout=kv_layout,
    )
    if q.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _profile_callable(torch, args, label, fn):
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=args.profile_memory,
        acc_events=True,
    ) as prof:
        for _ in range(args.repeat):
            with record_function(label):
                fn()
        torch.cuda.synchronize()
    return prof


def _metadata(torch, args, case_name, shape, dtype_name, impl, seq_lens, block_tables, cache):
    num_seqs, num_q_heads, num_kv_heads, head_dim, max_seq_len = shape
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "case": case_name,
        "impl": impl,
        "dtype": dtype_name,
        "num_seqs": num_seqs,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "min_seq_len": int(seq_lens.min().item()),
        "max_actual_seq_len": int(seq_lens.max().item()),
        "block_size": args.block_size,
        "kv_layout": args.kv_layout,
        "max_blocks_per_seq": block_tables.shape[1],
        "used_blocks": cache.num_used_blocks,
        "num_warps": args.num_warps,
        "num_stages": _num_stages_label(args.num_stages),
        "warmup": args.warmup,
        "repeat": args.repeat,
    }


def _format_metadata(metadata):
    return "\n".join(f"{key}: {value}" for key, value in metadata.items())


def _write_profile_text(path, metadata, latency_row, metric_metadata, table, trace_path=None):
    lines = [
        "# Paged Decode Profile",
        "",
        "## Metadata",
        "",
        _format_metadata(metadata),
        "",
        "## CUDA Event Latency",
        "",
        _format_metadata(latency_row),
        "",
        "## Estimated Traffic Metrics",
        "",
        _format_metadata(metric_metadata),
        "",
        "## PyTorch Profiler",
        "",
        table,
        "",
    ]
    if trace_path is not None:
        lines.extend(["## Chrome Trace", "", str(trace_path), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _profile_one_impl(torch, args, case_name, shape, dtype_name, tensors, cache, impl, summary_rows):
    from flashdec.kernels.paged_decode import paged_decode_attention
    from flashdec.paged_reference import paged_decode_attention_ref

    q, k_cache, v_cache, block_tables, seq_lens = tensors
    if impl == "triton":
        fn = lambda: paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            block_size=args.block_size,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            kv_layout=args.kv_layout,
        )
    elif impl == "ref":
        fn = lambda: paged_decode_attention_ref(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            kv_layout=args.kv_layout,
        )
    else:
        raise ValueError(f"unsupported impl: {impl}")

    estimate = _estimate(shape, dtype_name, seq_lens, block_tables, args.block_size)
    metadata = _metadata(torch, args, case_name, shape, dtype_name, impl, seq_lens, block_tables, cache)
    latency = benchmark_case(
        f"paged_decode_{impl}",
        fn,
        warmup=args.warmup,
        repeat=args.repeat,
        metadata=metadata,
    )
    metric_metadata = paged_decode_metric_metadata(estimate, mean_ms=latency.mean_ms, p50_ms=latency.p50_ms)
    label = f"paged_decode/{impl}/{case_name}/{dtype_name}/{args.kv_layout}"
    prof = None
    if args.skip_torch_profiler:
        table = "Skipped by --skip-torch-profiler. Use external profilers such as ncu/nsys for this run."
    else:
        prof = _profile_callable(torch, args, label, fn)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)

    slug = (
        f"{case_name}_{dtype_name}_{args.kv_layout}_{impl}"
        f"_b{args.block_size}_w{args.num_warps}_s{_num_stages_label(args.num_stages)}"
    )
    output_path = Path(args.output_dir) / f"{slug}.txt"
    trace_path = None
    if args.export_trace:
        if prof is None:
            raise ValueError("--export-trace requires PyTorch profiler; remove --skip-torch-profiler")
        trace_path = Path(args.output_dir) / f"{slug}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    latency_row = latency.as_row()
    _write_profile_text(output_path, metadata, latency_row, metric_metadata, table, trace_path=trace_path)
    print(table)
    print(f"Wrote {output_path}")
    if trace_path is not None:
        print(f"Wrote {trace_path}")

    summary_rows.append(
        {
            "case": case_name,
            "shape": "x".join(str(value) for value in shape),
            "impl": impl,
            "dtype": dtype_name,
            "kv_layout": args.kv_layout,
            "block_size": args.block_size,
            "num_warps": args.num_warps,
            "num_stages": _num_stages_label(args.num_stages),
            "p50_ms": latency_row["p50_ms"],
            "p90_ms": latency_row["p90_ms"],
            "mean_ms": latency_row["mean_ms"],
            "effective_total_gbps_p50": metric_metadata["effective_total_gbps_p50"],
            "device": metadata["device"],
            "torch": metadata["torch"],
            "cuda": metadata["cuda"],
            "profile": str(output_path),
        }
    )


def _write_summary(path, rows):
    lines = [
        "# Week 9 Paged Decode Profiling Summary",
        "",
        "Shape order: `num_seqs x num_q_heads x num_kv_heads x head_dim x max_seq_len`.",
        "",
        "| case | shape | impl | dtype | kv_layout | block_size | num_warps | num_stages | p50_ms | p90_ms | mean_ms | effective_total_gbps_p50 | device | torch | cuda | profile |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {shape} | {impl} | {dtype} | {kv_layout} | {block_size} | {num_warps} | {num_stages} | {p50_ms} | {p90_ms} | {mean_ms} | {effective_total_gbps_p50} | {device} | {torch} | {cuda} | {profile} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Profile text files live under `benchmarks/profiles/` and are intentionally not committed by default.",
            "- Treat effective bandwidth as a logical estimate; use Nsight Compute for hardware memory throughput.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    parser.add_argument("--impl", choices=["triton", "ref", "both"], default="triton")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="float16")
    parser.add_argument(
        "--kv-layout",
        choices=["token_major", "dim_major"],
        default="token_major",
    )
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=32)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument(
        "--num-stages",
        type=_parse_num_stages,
        default=None,
        metavar="STAGE",
        help="Triton stages: default, 1, 2, 3, or 4.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-torch-profiler", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/profiles/week9_paged_decode")
    parser.add_argument("--summary-output", default="benchmarks/results/week9_summary.md")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for paged decode profiling")

    summary_rows = []
    for dtype_name in _dtype_names(args.dtype):
        dtype = _dtype_from_name(torch, dtype_name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            if args.dtype == "bfloat16":
                raise SystemExit(
                    "bfloat16 was requested, but this CUDA device does not report BF16 support"
                )
            print("Skipping bfloat16 because torch.cuda.is_bf16_supported() is false.")
            continue

        for case_index, (case_name, shape) in enumerate(_selected_cases(args.case)):
            q, k_cache, v_cache, block_tables, seq_lens, cache = _make_inputs(
                torch,
                args,
                shape,
                dtype,
                case_index,
            )
            if not args.skip_validate:
                _validate(
                    torch,
                    q,
                    k_cache,
                    v_cache,
                    block_tables,
                    seq_lens,
                    args.block_size,
                    args.num_warps,
                    args.kv_layout,
                    args.num_stages,
                )
            tensors = (q, k_cache, v_cache, block_tables, seq_lens)
            for impl in _impls(args.impl):
                _profile_one_impl(
                    torch,
                    args,
                    case_name,
                    shape,
                    _dtype_name(dtype),
                    tensors,
                    cache,
                    impl,
                    summary_rows,
                )

    summary_path = Path(args.summary_output)
    _write_summary(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
