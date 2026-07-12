"""Profile complete DecodeEngine workloads with explicit runtime stage ranges."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_decode_engine_workload import (
    BLOCK_SIZE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    WORKLOADS,
    _preload_native_backends,
    _quick_spec,
    _requested_dtypes,
)
from flashdec.engine import (
    DecodeEngine,
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
    PROFILE_RANGE_PREFLIGHT,
)
from flashdec.benchmark import git_commit
from flashdec.workload import run_synthetic_workload


PROFILE_RANGE_SUBMIT = "flashdec::request_submit"
PROFILE_RANGE_ADMIT = "flashdec::request_admit"
PROFILE_RANGE_ENGINE_STEP = "flashdec::engine_step"
PROFILE_RANGE_FINISH = "flashdec::request_finish"
PROFILE_RANGE_CANCEL = "flashdec::request_cancel"

PROFILE_RANGES = (
    PROFILE_RANGE_SUBMIT,
    PROFILE_RANGE_ADMIT,
    PROFILE_RANGE_ENGINE_STEP,
    PROFILE_RANGE_PREFLIGHT,
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
    PROFILE_RANGE_FINISH,
    PROFILE_RANGE_CANCEL,
)


class ProfileValidationError(ValueError):
    """Raised when profiler rows cannot support stage-attribution evidence."""


class _ProfiledDecodeEngine(DecodeEngine):
    """DecodeEngine with lifecycle and complete-step record_function ranges."""

    def _record(self, name):
        import torch

        return torch.profiler.record_function(name)

    def add_request(self, request_id):
        with self._record(PROFILE_RANGE_SUBMIT):
            return super().add_request(request_id)

    def admit(self, request_ids=None):
        with self._record(PROFILE_RANGE_ADMIT):
            return super().admit(request_ids)

    def step(self, *args, **kwargs):
        with self._record(PROFILE_RANGE_ENGINE_STEP):
            return super().step(*args, **kwargs)

    def finish_request(self, request_id):
        with self._record(PROFILE_RANGE_FINISH):
            return super().finish_request(request_id)

    def cancel_request(self, request_id):
        with self._record(PROFILE_RANGE_CANCEL):
            return super().cancel_request(request_id)


def _dtype_names(dtype_name):
    return ["float16", "bfloat16"] if dtype_name == "both" else [dtype_name]


def _selected_workloads(workload_name):
    if workload_name == "all":
        return list(WORKLOADS.values())
    return [WORKLOADS[workload_name]]


def _make_engine(torch, dtype, max_blocks, append_backend, profiled):
    from flashdec.cache import PagedKVCache

    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        max_blocks=max_blocks,
        dtype=dtype,
        device="cuda",
    )
    engine_class = _ProfiledDecodeEngine if profiled else DecodeEngine
    return engine_class(
        cache,
        append_backend=append_backend,
        decode_backend="triton",
        num_warps=2,
        profile_ranges=profiled,
    )


def _event_time_us(event, primary, fallback):
    value = getattr(event, primary, None)
    if value is None:
        value = getattr(event, fallback, 0.0)
    return float(value or 0.0)


def _stage_rows(key_averages):
    """Extract stable stage records from torch profiler key averages."""
    events = {event.key: event for event in key_averages}
    rows = []
    for name in PROFILE_RANGES:
        event = events.get(name)
        if event is None:
            rows.append(
                {
                    "range": name,
                    "count": 0,
                    "cpu_total_ms": 0.0,
                    "cpu_self_ms": 0.0,
                    "device_total_ms": 0.0,
                    "device_self_ms": 0.0,
                }
            )
            continue
        rows.append(
            {
                "range": name,
                "count": int(event.count),
                "cpu_total_ms": _event_time_us(event, "cpu_time_total", "cpu_time_total")
                / 1_000.0,
                "cpu_self_ms": _event_time_us(
                    event,
                    "self_cpu_time_total",
                    "self_cpu_time_total",
                )
                / 1_000.0,
                "device_total_ms": _event_time_us(
                    event,
                    "device_time_total",
                    "cuda_time_total",
                )
                / 1_000.0,
                "device_self_ms": _event_time_us(
                    event,
                    "self_device_time_total",
                    "self_cuda_time_total",
                )
                / 1_000.0,
            }
        )
    return rows


def _stage_map(stage_rows):
    return {row["range"]: row for row in stage_rows}


def _device_event_count(key_averages):
    """Count profiler events whose recorded device type is CUDA."""
    return sum(
        int(event.count)
        for event in key_averages
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    )


def _write_profile_text(
    path,
    metadata,
    result,
    stage_rows,
    profiler_table,
    cuda_event_count,
    trace_path=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DecodeEngine profile metadata"]
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    lines.extend(
        [
            "",
            "# Instrumented workload result",
            f"mean_ms: {result.mean_ms:.6f}",
            f"p50_ms: {result.p50_ms:.6f}",
            f"p90_ms: {result.p90_ms:.6f}",
            f"p99_ms: {result.p99_ms:.6f}",
            f"tokens_per_second: {result.tokens_per_second:.3f}",
            f"successful_steps: {result.successful_steps}",
            f"backpressure_steps: {result.backpressure_steps}",
            f"profiler_cuda_event_count: {cuda_event_count}",
            "",
            "# Explicit profiler ranges",
            "range,count,cpu_total_ms,cpu_self_ms,device_total_ms,device_self_ms",
        ]
    )
    for row in stage_rows:
        lines.append(
            "{range},{count},{cpu_total_ms:.6f},{cpu_self_ms:.6f},"
            "{device_total_ms:.6f},{device_self_ms:.6f}".format(**row)
        )
    lines.extend(["", "# PyTorch profiler table", profiler_table])
    if trace_path is not None:
        lines.extend(["", f"chrome_trace: {trace_path}"])
    path.write_text("\n".join(lines) + "\n")


def _write_summary(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("profile summary rows must be non-empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DecodeEngine Stage Profiling Summary",
        "",
        "Instrumented wall-clock values are not release benchmark numbers. "
        "Nested CPU/device profiler totals are attribution evidence and must not be added blindly.",
        f"Git commit: `{rows[0]['git_commit']}`.",
        "",
        "| workload | dtype | append | steps | successful | backpressure | CUDA events | ranges step/preflight/append/decode | wall p50 ms | wall p99 ms | engine CPU ms | engine device ms | append device ms | decode device ms | profile | trace |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {dtype} | {append_backend} | {steps} | {successful_steps} | "
            "{backpressure_steps} | {cuda_event_count} | "
            "{engine_count}/{preflight_count}/{append_count}/{decode_count} | "
            "{p50_ms:.6f} | {p99_ms:.6f} | "
            "{engine_cpu_ms:.6f} | {engine_device_ms:.6f} | "
            "{append_device_ms:.6f} | {decode_device_ms:.6f} | {profile} | {trace} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `engine_step` is an inclusive range containing preflight, append, and decode.",
            "- `rope_kv_append` includes Python allocator/metadata work plus the selected append backend.",
            "- Matrix completeness, Git commit, CUDA event presence, and named-range counts were validated before this summary was written.",
            "- Q/K/V generation and prompt prefill are captured by the global profiler but intentionally outside named Engine ranges.",
            "- Final performance decisions remain based on non-instrumented multi-trial workload CSVs.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _profile_int(row, field, *, minimum=0):
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileValidationError(f"{field} must be an integer") from exc
    if value < minimum:
        raise ProfileValidationError(f"{field} must be >= {minimum}")
    return value


def _profile_positive_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ProfileValidationError(f"{field} must be positive and finite")
    return value


def validate_profile_rows(rows, expected_workloads, expected_dtypes, expected_backends):
    """Validate profile matrix completeness and named-range execution counts."""
    rows = list(rows)
    expected_workloads = tuple(expected_workloads)
    expected_dtypes = tuple(expected_dtypes)
    expected_backends = tuple(expected_backends)
    for name, values in (
        ("expected_workloads", expected_workloads),
        ("expected_dtypes", expected_dtypes),
        ("expected_backends", expected_backends),
    ):
        if not values or len(set(values)) != len(values):
            raise ProfileValidationError(f"{name} must be non-empty and unique")

    expected_keys = {
        (workload, dtype, backend)
        for workload in expected_workloads
        for dtype in expected_dtypes
        for backend in expected_backends
    }
    indexed = {}
    commits = set()
    for row in rows:
        try:
            key = (row["workload"], row["dtype"], row["append_backend"])
            commit = str(row["git_commit"]).strip()
        except KeyError as exc:
            raise ProfileValidationError(f"missing profile field: {exc.args[0]}") from exc
        if key in indexed:
            raise ProfileValidationError(f"duplicate profile row: {key}")
        if not commit:
            raise ProfileValidationError("git_commit must be non-empty")
        indexed[key] = row
        commits.add(commit)

        steps = _profile_int(row, "steps", minimum=1)
        successful = _profile_int(row, "successful_steps", minimum=1)
        backpressure = _profile_int(row, "backpressure_steps")
        engine_count = _profile_int(row, "engine_count", minimum=1)
        preflight_count = _profile_int(row, "preflight_count", minimum=1)
        append_count = _profile_int(row, "append_count", minimum=1)
        decode_count = _profile_int(row, "decode_count", minimum=1)
        _profile_int(row, "cuda_event_count", minimum=1)
        _profile_positive_float(row, "p50_ms")
        _profile_positive_float(row, "p99_ms")

        if engine_count != steps:
            raise ProfileValidationError(f"engine range count must equal steps: {key}")
        if preflight_count != engine_count:
            raise ProfileValidationError(f"preflight range count must equal engine count: {key}")
        if successful + backpressure != engine_count:
            raise ProfileValidationError(f"successful + backpressure must equal engine count: {key}")
        if append_count != successful or decode_count != successful:
            raise ProfileValidationError(
                f"append/decode range counts must equal successful steps: {key}"
            )
    actual_keys = set(indexed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ProfileValidationError(
            f"profile matrix is incomplete; missing={missing}, unexpected={unexpected}"
        )
    if len(commits) != 1:
        raise ProfileValidationError(
            f"profile rows use inconsistent git commits: {sorted(commits)}"
        )
    return rows


def _metadata(torch, spec, dtype_name, append_backend, seed, quick, commit):
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "workload": spec.config.name,
        "dtype": dtype_name,
        "append_backend": append_backend,
        "decode_backend": "triton",
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_commit": commit,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": 2,
        "steps": spec.config.steps,
        "max_active": spec.config.max_active,
        "max_blocks": spec.max_blocks,
        "seed": seed,
        "quick": quick,
        "timing_scope": "instrumented dynamic workload; profiler ranges add overhead",
    }


def _prewarm(torch, spec, dtype, append_backend, warmup_steps, seed):
    if warmup_steps <= 0:
        return
    engine = _make_engine(torch, dtype, spec.max_blocks, append_backend, profiled=False)
    config = replace(spec.config, steps=warmup_steps)
    run_synthetic_workload(
        engine,
        config,
        num_q_heads=NUM_Q_HEADS,
        warmup_steps=0,
        seed=seed,
    )
    torch.cuda.synchronize()


def _profile_case(torch, args, spec, dtype_name, dtype, append_backend):
    _prewarm(torch, spec, dtype, append_backend, args.warmup_steps, args.seed)
    engine = _make_engine(torch, dtype, spec.max_blocks, append_backend, profiled=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        result = run_synthetic_workload(
            engine,
            spec.config,
            num_q_heads=NUM_Q_HEADS,
            warmup_steps=0,
            seed=args.seed,
        )
    torch.cuda.synchronize()

    key_averages = prof.key_averages()
    stage_rows = _stage_rows(key_averages)
    stages = _stage_map(stage_rows)
    cuda_event_count = _device_event_count(key_averages)
    profiler_table = key_averages.table(sort_by="cuda_time_total", row_limit=args.row_limit)
    slug = f"{spec.config.name}_{dtype_name}_{append_backend}"
    profile_path = Path(args.output_dir) / f"{slug}.txt"
    trace_path = None
    if args.export_trace:
        trace_path = Path(args.output_dir) / f"{slug}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    metadata = _metadata(
        torch,
        spec,
        dtype_name,
        append_backend,
        args.seed,
        args.quick,
        args.git_commit,
    )
    _write_profile_text(
        profile_path,
        metadata,
        result,
        stage_rows,
        profiler_table,
        cuda_event_count,
        trace_path=trace_path,
    )
    engine_stage = stages[PROFILE_RANGE_ENGINE_STEP]
    append_stage = stages[PROFILE_RANGE_APPEND]
    decode_stage = stages[PROFILE_RANGE_DECODE]
    return {
        "workload": spec.config.name,
        "dtype": dtype_name,
        "append_backend": append_backend,
        "steps": spec.config.steps,
        "successful_steps": result.successful_steps,
        "backpressure_steps": result.backpressure_steps,
        "cuda_event_count": cuda_event_count,
        "engine_count": engine_stage["count"],
        "preflight_count": stages[PROFILE_RANGE_PREFLIGHT]["count"],
        "append_count": append_stage["count"],
        "decode_count": decode_stage["count"],
        "p50_ms": result.p50_ms,
        "p99_ms": result.p99_ms,
        "engine_cpu_ms": engine_stage["cpu_total_ms"],
        "engine_device_ms": engine_stage["device_total_ms"],
        "append_device_ms": append_stage["device_total_ms"],
        "decode_device_ms": decode_stage["device_total_ms"],
        "profile": str(profile_path),
        "trace": str(trace_path) if trace_path is not None else "-",
        "git_commit": args.git_commit,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["all", *WORKLOADS], default="mixed_steady")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "both"], default="float16")
    parser.add_argument(
        "--append-backends",
        nargs="+",
        choices=["torch", "cuda", "fused_cuda"],
        default=["fused_cuda"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/profiles/week12_decode_engine")
    parser.add_argument(
        "--summary-output",
        default="benchmarks/results/week12_decode_engine_profile_summary.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup_steps < 0 or args.row_limit <= 0:
        raise SystemExit("warmup-steps must be non-negative and row-limit must be positive")
    if len(set(args.append_backends)) != len(args.append_backends):
        raise SystemExit("append-backends must be unique")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DecodeEngine profiling")
    _preload_native_backends(args.append_backends)
    args.git_commit = git_commit(PROJECT_ROOT)
    specs = _selected_workloads(args.workload)
    if args.quick:
        specs = [_quick_spec(spec) for spec in specs]

    summary_rows = []
    dtype_pairs = _requested_dtypes(torch, args.dtype)
    for dtype_name, dtype in dtype_pairs:
        for spec in specs:
            for append_backend in args.append_backends:
                row = _profile_case(
                    torch,
                    args,
                    spec,
                    dtype_name,
                    dtype,
                    append_backend,
                )
                summary_rows.append(row)
                print(f"Wrote {row['profile']}")
                if row["trace"] != "-":
                    print(f"Wrote {row['trace']}")

    try:
        validate_profile_rows(
            summary_rows,
            expected_workloads=[spec.config.name for spec in specs],
            expected_dtypes=[name for name, _ in dtype_pairs],
            expected_backends=args.append_backends,
        )
    except ProfileValidationError as exc:
        raise SystemExit(f"invalid DecodeEngine profile evidence: {exc}") from exc
    _write_summary(args.summary_output, summary_rows)
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
