"""Run the R5 FlashDec vs FlashInfer paged-decode baseline matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib import metadata as importlib_metadata
import math
import os
import platform
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashdec.benchmark import cuda_event_timer, git_commit, percentile
from flashdec.perf import dtype_nbytes


EXPECTED_FLASHINFER_VERSION = "0.6.15.post1"
EXPECTED_PYTHON_MAJOR_MINOR = "3.12"
EXPECTED_TORCH_VERSION = "2.11.0+cu128"
EXPECTED_TRITON_VERSION = "3.6.0"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_CUDA_TOOLKIT_VERSION = "12.8.1"
EXPECTED_CUDA_PYTHON_VERSION = "12.9.1"
EXPECTED_CUDA_BINDINGS_VERSION = "12.9.7"
EXPECTED_CUDA_PATHFINDER_VERSION = "1.6.0"
EXPECTED_NINJA_VERSION = "1.13.0"
EXPECTED_CUDA_HOME_BASENAME = "cuda-12.8"
EXPECTED_NVCC_RELEASE = "12.8"
EXPECTED_NVCC_VERSION = "12.8.93"
FLASHINFER_CUDA_ARCH_ENV = "FLASHINFER_CUDA_ARCH_LIST"
EXPECTED_FLASHINFER_CUDA_ARCH_LIST = "12.0a"
FLASHINFER_BACKEND = "fa2"
FLASHINFER_WORKSPACE_MIB = 128
BLOCK_SIZE = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
FLASHDEC_KV_LAYOUT = "token_major"
FLASHINFER_KV_LAYOUT = "HND"
TIMING_SCOPE = "cuda_event_run_only_plan_jit_inputs_reference_excluded"
FORMAL_TRIALS = 3
FORMAL_WARMUP = 10
FORMAL_REPEATS = 50
QUICK_TRIALS = 1
QUICK_WARMUP = 2
QUICK_REPEATS = 10

BACKENDS = (
    "flashdec_triton",
    "flashinfer_fa2_cuda_core",
    "flashinfer_fa2_tensor_core",
)
DTYPES = ("float16", "bfloat16")


@dataclass(frozen=True)
class BaselineCase:
    """One pre-registered common paged-decode geometry."""

    name: str
    num_seqs: int
    context_tokens: int


CASES = {
    "small": BaselineCase("small_b1_ctx128", 1, 128),
    "medium": BaselineCase("medium_b16_ctx1024", 16, 1024),
    "large": BaselineCase("large_b16_ctx8192", 16, 8192),
    "large_batch": BaselineCase("large_batch_b64_ctx4096", 64, 4096),
}
DEFAULT_CASES = tuple(case.name for case in CASES.values())


def _rotate(values, offset):
    values = tuple(values)
    if not values:
        raise ValueError("values must be non-empty")
    offset %= len(values)
    return values[offset:] + values[:offset]


def _backend_order(trial):
    if trial <= 0:
        raise ValueError("trial must be positive")
    return _rotate(BACKENDS, trial - 1)


def _case_order(cases, trial):
    if trial <= 0:
        raise ValueError("trial must be positive")
    return _rotate(cases, trial - 1)


def _dtype_order(dtypes, trial):
    if trial <= 0:
        raise ValueError("trial must be positive")
    return _rotate(dtypes, trial - 1)


def _selected_cases(name):
    if name == "all":
        return tuple(CASES.values())
    return (CASES[name],)


def _selected_dtypes(name):
    if name == "both":
        return DTYPES
    return (name,)


def _dtype_from_name(torch, name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _flashinfer_version(flashinfer):
    try:
        return importlib_metadata.version("flashinfer-python")
    except importlib_metadata.PackageNotFoundError:
        version = getattr(flashinfer, "__version__", None)
        if version:
            return str(version)
        raise RuntimeError("cannot determine flashinfer-python version")


def _installed_version(distribution, version_getter=None):
    getter = (
        importlib_metadata.version if version_getter is None else version_getter
    )
    try:
        return str(getter(distribution))
    except importlib_metadata.PackageNotFoundError:
        return "<not installed>"


def _parse_nvcc_version(output):
    match = re.search(
        r"release\s+(\d+\.\d+),\s+V(\d+\.\d+\.\d+)",
        output,
    )
    if match is None:
        raise RuntimeError("cannot parse CUDA release/version from nvcc --version")
    return match.group(1), match.group(2)


def _probe_cuda_toolkit(cuda_home):
    path = Path(cuda_home)
    if not path.is_absolute():
        raise RuntimeError("CUDA_HOME must be an absolute path")
    try:
        realpath = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"CUDA_HOME cannot be resolved: {cuda_home!r}") from exc
    nvcc = realpath / "bin" / "nvcc"
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise RuntimeError(f"CUDA_HOME does not contain executable nvcc: {nvcc}")
    result = subprocess.run(
        [str(nvcc), "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc --version failed with exit code {result.returncode}")
    release, version = _parse_nvcc_version(result.stdout)
    return {
        "cuda_home_realpath": str(realpath),
        "nvcc_release": release,
        "nvcc_version": version,
    }


def _validate_r5_environment(
    torch,
    *,
    environ=None,
    version_getter=None,
    python_version=None,
    cuda_probe=None,
):
    """Return the frozen R5 environment or reject it before FlashInfer import/JIT."""

    environ = os.environ if environ is None else environ
    python_version = (
        platform.python_version() if python_version is None else python_version
    )
    cuda_home = str(environ.get("CUDA_HOME", "")).strip()
    if cuda_home:
        probe = _probe_cuda_toolkit if cuda_probe is None else cuda_probe
        toolkit = probe(cuda_home)
    else:
        toolkit = {
            "cuda_home_realpath": "",
            "nvcc_release": "",
            "nvcc_version": "",
        }
    actual = {
        "python": str(python_version),
        "torch": str(torch.__version__),
        "triton": _installed_version("triton", version_getter),
        "cuda": str(torch.version.cuda),
        "cuda_toolkit": _installed_version("cuda-toolkit", version_getter),
        "cuda_python": _installed_version("cuda-python", version_getter),
        "cuda_bindings": _installed_version("cuda-bindings", version_getter),
        "cuda_pathfinder": _installed_version("cuda-pathfinder", version_getter),
        "ninja": _installed_version("ninja", version_getter),
        "flashinfer_version": _installed_version(
            "flashinfer-python", version_getter
        ),
        "cuda_home": cuda_home,
        **toolkit,
        "flashinfer_cuda_arch_list": str(
            environ.get(FLASHINFER_CUDA_ARCH_ENV, "")
        ).strip(),
    }
    expected = {
        "torch": EXPECTED_TORCH_VERSION,
        "triton": EXPECTED_TRITON_VERSION,
        "cuda": EXPECTED_TORCH_CUDA_VERSION,
        "cuda_toolkit": EXPECTED_CUDA_TOOLKIT_VERSION,
        "cuda_python": EXPECTED_CUDA_PYTHON_VERSION,
        "cuda_bindings": EXPECTED_CUDA_BINDINGS_VERSION,
        "cuda_pathfinder": EXPECTED_CUDA_PATHFINDER_VERSION,
        "ninja": EXPECTED_NINJA_VERSION,
        "nvcc_release": EXPECTED_NVCC_RELEASE,
        "nvcc_version": EXPECTED_NVCC_VERSION,
        "flashinfer_version": EXPECTED_FLASHINFER_VERSION,
        "flashinfer_cuda_arch_list": EXPECTED_FLASHINFER_CUDA_ARCH_LIST,
    }
    mismatches = [
        f"{field}: expected {value!r}, got {actual[field]!r}"
        for field, value in expected.items()
        if actual[field] != value
    ]
    if not actual["python"].startswith(f"{EXPECTED_PYTHON_MAJOR_MINOR}."):
        mismatches.append(
            "python: expected "
            f"{EXPECTED_PYTHON_MAJOR_MINOR}.x, got {actual['python']!r}"
        )
    if not cuda_home:
        mismatches.append("CUDA_HOME: expected an explicit CUDA 12.8 toolkit path")
    elif Path(cuda_home).name != EXPECTED_CUDA_HOME_BASENAME:
        mismatches.append(
            "CUDA_HOME: expected a path ending in "
            f"{EXPECTED_CUDA_HOME_BASENAME!r}, got {cuda_home!r}"
        )
    realpath = actual["cuda_home_realpath"]
    if not realpath:
        mismatches.append("CUDA_HOME realpath must be recorded")
    elif Path(realpath).name != EXPECTED_CUDA_HOME_BASENAME:
        mismatches.append(
            "CUDA_HOME realpath: expected a path ending in "
            f"{EXPECTED_CUDA_HOME_BASENAME!r}, got {realpath!r}"
        )
    if mismatches:
        raise RuntimeError("R5 environment mismatch; " + "; ".join(mismatches))
    return actual


def _git_worktree_clean(root):
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot determine Git worktree state")
    return not result.stdout.strip()


def _page_table_digest(page_indices):
    values = page_indices.detach().cpu().tolist()
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _logical_workload_bytes(*, num_seqs, context_tokens, dtype_name):
    """Count each logical Q/K/V/output element once for every backend.

    This intentionally excludes page metadata and implementation-specific
    rereads.  It is a common workload-size proxy, not an estimate of physical
    DRAM traffic for either implementation.
    """
    element_bytes = dtype_nbytes(dtype_name)
    q_and_output = (
        2 * num_seqs * NUM_Q_HEADS * HEAD_DIM * element_bytes
    )
    k_and_v = (
        2
        * num_seqs
        * context_tokens
        * NUM_KV_HEADS
        * HEAD_DIM
        * element_bytes
    )
    return q_and_output + k_and_v


def _make_inputs(torch, case, dtype, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pages_per_seq = math.ceil(case.context_tokens / BLOCK_SIZE)
    num_pages = case.num_seqs * pages_per_seq
    q = torch.randn(
        (case.num_seqs, NUM_Q_HEADS, HEAD_DIM),
        dtype=dtype,
        device="cuda",
    )
    k_cache = torch.randn(
        (num_pages, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
        dtype=dtype,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    page_indices = torch.randperm(num_pages, device="cuda").to(torch.int32)
    block_tables = page_indices.view(case.num_seqs, pages_per_seq).contiguous()
    seq_lens = torch.full(
        (case.num_seqs,),
        case.context_tokens,
        dtype=torch.int32,
        device="cuda",
    )
    page_indptr = torch.arange(
        0,
        num_pages + 1,
        pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    last_page_length = case.context_tokens % BLOCK_SIZE or BLOCK_SIZE
    last_page_len = torch.full(
        (case.num_seqs,),
        last_page_length,
        dtype=torch.int32,
        device="cuda",
    )
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "page_indptr": page_indptr,
        "page_indices": page_indices,
        "last_page_len": last_page_len,
        "page_table_digest": _page_table_digest(page_indices),
        "pages_per_seq": pages_per_seq,
        "num_pages": num_pages,
    }


def _make_flashinfer_wrapper(
    torch,
    flashinfer,
    inputs,
    *,
    dtype,
    use_tensor_cores,
    workspace_mib,
    backend,
):
    workspace = torch.zeros(
        workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        FLASHINFER_KV_LAYOUT,
        use_tensor_cores=use_tensor_cores,
        backend=backend,
    )
    wrapper.plan(
        inputs["page_indptr"],
        inputs["page_indices"],
        inputs["last_page_len"],
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        sm_scale=HEAD_DIM**-0.5,
    )
    return wrapper, workspace


def _max_abs_error(torch, actual, expected):
    return float(torch.max(torch.abs(actual.float() - expected.float())).item())


def _max_tolerance_ratio(torch, actual, expected, *, rtol, atol):
    error = torch.abs(actual.float() - expected.float())
    allowed = atol + rtol * torch.abs(expected.float())
    return float(torch.max(error / allowed).item())


def _validate_outputs(torch, inputs, callables, dtype_name):
    from flashdec.paged_reference import paged_decode_attention_ref

    outputs = {backend: fn() for backend, fn in callables.items()}
    torch.cuda.synchronize()
    sample_size = min(2, inputs["q"].shape[0])
    expected = paged_decode_attention_ref(
        inputs["q"][:sample_size],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["block_tables"][:sample_size],
        inputs["seq_lens"][:sample_size],
        kv_layout=FLASHDEC_KV_LAYOUT,
    )
    if dtype_name == "bfloat16":
        rtol = atol = 3e-2
    else:
        rtol = atol = 2e-2

    reference_errors = {}
    cross_errors = {}
    reference_tolerance_ratios = {}
    cross_tolerance_ratios = {}
    flashdec_output = outputs["flashdec_triton"]
    for backend, output in outputs.items():
        torch.testing.assert_close(
            output[:sample_size],
            expected,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            output,
            flashdec_output,
            rtol=rtol,
            atol=atol,
        )
        reference_errors[backend] = _max_abs_error(
            torch, output[:sample_size], expected
        )
        cross_errors[backend] = _max_abs_error(torch, output, flashdec_output)
        reference_tolerance_ratios[backend] = _max_tolerance_ratio(
            torch,
            output[:sample_size],
            expected,
            rtol=rtol,
            atol=atol,
        )
        cross_tolerance_ratios[backend] = _max_tolerance_ratio(
            torch,
            output,
            flashdec_output,
            rtol=rtol,
            atol=atol,
        )
    return {
        "reference_sample_size": sample_size,
        "rtol": rtol,
        "atol": atol,
        "reference_errors": reference_errors,
        "cross_errors": cross_errors,
        "reference_tolerance_ratios": reference_tolerance_ratios,
        "cross_tolerance_ratios": cross_tolerance_ratios,
    }


def _latency_row(latencies):
    return {
        "mean_ms": f"{statistics.fmean(latencies):.6f}",
        "p50_ms": f"{percentile(latencies, 50):.6f}",
        "p90_ms": f"{percentile(latencies, 90):.6f}",
        "p99_ms": f"{percentile(latencies, 99):.6f}",
        "min_ms": f"{min(latencies):.6f}",
        "max_ms": f"{max(latencies):.6f}",
    }


def _write_rows(rows, path):
    if not rows:
        raise ValueError("rows must be non-empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_case(
    torch,
    flashinfer,
    args,
    *,
    case,
    dtype_name,
    trial,
    case_order,
    dtype_order,
    backend_order,
    commit,
    worktree_clean,
    run_started_at,
    run_command,
    actual_flashinfer_version,
    environment,
):
    from flashdec.kernels.paged_decode import paged_decode_attention

    dtype = _dtype_from_name(torch, dtype_name)
    seed = args.seed + trial - 1
    inputs = _make_inputs(torch, case, dtype, seed)
    core_wrapper, core_workspace = _make_flashinfer_wrapper(
        torch,
        flashinfer,
        inputs,
        dtype=dtype,
        use_tensor_cores=False,
        workspace_mib=args.workspace_mib,
        backend=args.flashinfer_backend,
    )
    tensor_wrapper, tensor_workspace = _make_flashinfer_wrapper(
        torch,
        flashinfer,
        inputs,
        dtype=dtype,
        use_tensor_cores=True,
        workspace_mib=args.workspace_mib,
        backend=args.flashinfer_backend,
    )
    kv_cache = (inputs["k_cache"], inputs["v_cache"])
    callables = {
        "flashdec_triton": lambda: paged_decode_attention(
            inputs["q"],
            inputs["k_cache"],
            inputs["v_cache"],
            inputs["block_tables"],
            inputs["seq_lens"],
            block_size=BLOCK_SIZE,
            num_warps=2,
            num_stages=None,
            kv_layout=FLASHDEC_KV_LAYOUT,
        ),
        "flashinfer_fa2_cuda_core": lambda: core_wrapper.run(
            inputs["q"], kv_cache
        ),
        "flashinfer_fa2_tensor_core": lambda: tensor_wrapper.run(
            inputs["q"], kv_cache
        ),
    }
    validation = _validate_outputs(torch, inputs, callables, dtype_name)
    logical_workload_bytes = _logical_workload_bytes(
        num_seqs=case.num_seqs,
        context_tokens=case.context_tokens,
        dtype_name=dtype_name,
    )

    rows = []
    for backend in backend_order:
        latencies = cuda_event_timer(
            callables[backend],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        latency = _latency_row(latencies)
        p50_ms = float(latency["p50_ms"])
        use_tensor_cores = {
            "flashdec_triton": "not_applicable",
            "flashinfer_fa2_cuda_core": "False",
            "flashinfer_fa2_tensor_core": "True",
        }[backend]
        row = {
            "name": "r5_flashinfer_paged_decode",
            "op": "paged_decode_attention",
            "date": run_started_at,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "python": environment["python"],
            "torch": environment["torch"],
            "triton": environment["triton"],
            "cuda": environment["cuda"],
            "cuda_toolkit": environment["cuda_toolkit"],
            "cuda_python": environment["cuda_python"],
            "cuda_bindings": environment["cuda_bindings"],
            "cuda_pathfinder": environment["cuda_pathfinder"],
            "ninja": environment["ninja"],
            "cuda_home": environment["cuda_home"],
            "cuda_home_realpath": environment["cuda_home_realpath"],
            "nvcc_release": environment["nvcc_release"],
            "nvcc_version": environment["nvcc_version"],
            "flashinfer_cuda_arch_list": environment[
                "flashinfer_cuda_arch_list"
            ],
            "git_commit": commit,
            "git_worktree_clean": str(worktree_clean),
            "command": run_command,
            "flashinfer_version": actual_flashinfer_version,
            "expected_flashinfer_version": args.expected_flashinfer_version,
            "flashinfer_workspace_mib": args.workspace_mib,
            "case": case.name,
            "dtype": dtype_name,
            "backend": backend,
            "flashinfer_backend": args.flashinfer_backend,
            "flashinfer_use_tensor_cores": use_tensor_cores,
            "num_seqs": case.num_seqs,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "context_tokens": case.context_tokens,
            "min_seq_len": case.context_tokens,
            "max_seq_len": case.context_tokens,
            "block_size": BLOCK_SIZE,
            "pages_per_seq": inputs["pages_per_seq"],
            "num_pages": inputs["num_pages"],
            "flashdec_kv_layout": FLASHDEC_KV_LAYOUT,
            "flashinfer_kv_layout": FLASHINFER_KV_LAYOUT,
            "pos_encoding_mode": "NONE",
            "sm_scale": f"{HEAD_DIM**-0.5:.12f}",
            "trial": trial,
            "trial_count": args.trials,
            "backend_order": "->".join(backend_order),
            "case_order": "->".join(item.name for item in case_order),
            "dtype_order": "->".join(dtype_order),
            "base_seed": args.seed,
            "seed": seed,
            "warmup": args.warmup,
            "repeats": args.repeat,
            "timing_scope": TIMING_SCOPE,
            "page_table_digest": inputs["page_table_digest"],
            "reference_sample_size": validation["reference_sample_size"],
            "reference_validated": "True",
            "cross_backend_validated": "True",
            "max_abs_error_vs_reference": (
                f"{validation['reference_errors'][backend]:.8f}"
            ),
            "max_abs_error_vs_flashdec": (
                f"{validation['cross_errors'][backend]:.8f}"
            ),
            "max_tolerance_ratio_vs_reference": (
                f"{validation['reference_tolerance_ratios'][backend]:.8f}"
            ),
            "max_tolerance_ratio_vs_flashdec": (
                f"{validation['cross_tolerance_ratios'][backend]:.8f}"
            ),
            "rtol": validation["rtol"],
            "atol": validation["atol"],
            **latency,
            "decode_tokens_per_second": f"{case.num_seqs * 1000.0 / p50_ms:.3f}",
            "logical_workload_bytes": logical_workload_bytes,
            "logical_workload_gbps_p50": (
                f"{logical_workload_bytes / (p50_ms * 1_000_000.0):.4f}"
            ),
            "validated_invariants": "True",
        }
        rows.append(row)

    # Keep workspaces alive until every timed call has completed.
    _ = (core_workspace, tensor_workspace)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument(
        "--dtype", choices=[*DTYPES, "both"], default="both"
    )
    parser.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    parser.add_argument("--warmup", type=int, default=FORMAL_WARMUP)
    parser.add_argument("--repeat", type=int, default=FORMAL_REPEATS)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--expected-flashinfer-version",
        default=EXPECTED_FLASHINFER_VERSION,
    )
    parser.add_argument(
        "--flashinfer-backend",
        choices=[FLASHINFER_BACKEND],
        default=FLASHINFER_BACKEND,
    )
    parser.add_argument("--workspace-mib", type=int, default=FLASHINFER_WORKSPACE_MIB)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap trials/warmup/repeat at 1/2/10 for a smoke run.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Refuse to benchmark when the Git worktree is dirty.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/r5_flashinfer_paged_decode_trials3.csv",
    )
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("trials must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("repeat must be positive")
    if args.workspace_mib <= 0:
        parser.error("workspace-mib must be positive")
    if args.quick:
        args.trials = min(args.trials, QUICK_TRIALS)
        args.warmup = min(args.warmup, QUICK_WARMUP)
        args.repeat = min(args.repeat, QUICK_REPEATS)
    return args


def main(argv=None):
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(cli_argv)
    run_command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *cli_argv]
    )

    worktree_clean = _git_worktree_clean(PROJECT_ROOT)
    if args.require_clean and not worktree_clean:
        raise SystemExit(
            "R5 evidence requires a clean Git worktree; commit or stash source changes"
        )

    import torch

    try:
        environment = _validate_r5_environment(torch)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the R5 FlashInfer baseline")
    if args.dtype in ("bfloat16", "both") and not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected matrix requires CUDA BF16 support")

    import flashinfer

    actual_flashinfer_version = _flashinfer_version(flashinfer)
    if actual_flashinfer_version != args.expected_flashinfer_version:
        raise SystemExit(
            "flashinfer-python version mismatch: "
            f"expected {args.expected_flashinfer_version}, got {actual_flashinfer_version}"
        )

    selected_cases = _selected_cases(args.case)
    selected_dtypes = _selected_dtypes(args.dtype)
    commit = git_commit(PROJECT_ROOT)
    run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for trial in range(1, args.trials + 1):
        case_order = _case_order(selected_cases, trial)
        dtype_order = _dtype_order(selected_dtypes, trial)
        backend_order = _backend_order(trial)
        for dtype_name in dtype_order:
            for case in case_order:
                rows.extend(
                    _run_case(
                        torch,
                        flashinfer,
                        args,
                        case=case,
                        dtype_name=dtype_name,
                        trial=trial,
                        case_order=case_order,
                        dtype_order=dtype_order,
                        backend_order=backend_order,
                        commit=commit,
                        worktree_clean=worktree_clean,
                        run_started_at=run_started_at,
                        run_command=run_command,
                        actual_flashinfer_version=actual_flashinfer_version,
                        environment=environment,
                    )
                )
                torch.cuda.empty_cache()

    _write_rows(rows, args.output)
    for row in rows:
        print(
            row["trial"],
            row["dtype"],
            row["case"],
            row["backend"],
            row["p50_ms"],
            row["decode_tokens_per_second"],
        )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
