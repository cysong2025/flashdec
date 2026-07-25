"""Dependency-free coverage for the R5 FlashInfer baseline runner."""

import contextlib
import io
import unittest
from unittest import mock

from benchmarks.run_flashinfer_baseline import (
    BACKENDS,
    CASES,
    DEFAULT_CASES,
    DTYPES,
    EXPECTED_CUDA_BINDINGS_VERSION,
    EXPECTED_CUDA_PATHFINDER_VERSION,
    EXPECTED_CUDA_PYTHON_VERSION,
    EXPECTED_CUDA_TOOLKIT_VERSION,
    EXPECTED_FLASHINFER_VERSION,
    EXPECTED_FLASHINFER_CUDA_ARCH_LIST,
    EXPECTED_NINJA_VERSION,
    EXPECTED_NVCC_RELEASE,
    EXPECTED_NVCC_VERSION,
    EXPECTED_PYTHON_MAJOR_MINOR,
    EXPECTED_TORCH_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRITON_VERSION,
    FLASHDEC_KV_LAYOUT,
    FLASHINFER_KV_LAYOUT,
    FORMAL_REPEATS,
    FORMAL_TRIALS,
    FORMAL_WARMUP,
    QUICK_REPEATS,
    QUICK_TRIALS,
    QUICK_WARMUP,
    _backend_order,
    _case_order,
    _dtype_order,
    _logical_workload_bytes,
    _page_table_digest,
    _parse_nvcc_version,
    _selected_cases,
    _selected_dtypes,
    _validate_r5_environment,
    main,
    parse_args,
)


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeTorch:
    __version__ = EXPECTED_TORCH_VERSION

    class version:
        cuda = EXPECTED_TORCH_CUDA_VERSION


def _canonical_environment():
    environ = {
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "FLASHINFER_CUDA_ARCH_LIST": EXPECTED_FLASHINFER_CUDA_ARCH_LIST,
    }
    versions = {
        "triton": EXPECTED_TRITON_VERSION,
        "cuda-toolkit": EXPECTED_CUDA_TOOLKIT_VERSION,
        "cuda-python": EXPECTED_CUDA_PYTHON_VERSION,
        "cuda-bindings": EXPECTED_CUDA_BINDINGS_VERSION,
        "cuda-pathfinder": EXPECTED_CUDA_PATHFINDER_VERSION,
        "ninja": EXPECTED_NINJA_VERSION,
        "flashinfer-python": EXPECTED_FLASHINFER_VERSION,
    }
    toolkit = {
        "cuda_home_realpath": "/usr/local/cuda-12.8",
        "nvcc_release": EXPECTED_NVCC_RELEASE,
        "nvcc_version": EXPECTED_NVCC_VERSION,
    }
    return environ, versions, toolkit


class FlashInferBaselineBenchmarkTests(unittest.TestCase):
    def test_formal_matrix_uses_frozen_public_shapes(self):
        self.assertEqual(
            [(case.name, case.num_seqs, case.context_tokens) for case in CASES.values()],
            [
                ("small_b1_ctx128", 1, 128),
                ("medium_b16_ctx1024", 16, 1024),
                ("large_b16_ctx8192", 16, 8192),
                ("large_batch_b64_ctx4096", 64, 4096),
            ],
        )
        self.assertEqual(DEFAULT_CASES, tuple(case.name for case in CASES.values()))
        self.assertEqual(DTYPES, ("float16", "bfloat16"))
        self.assertEqual(
            BACKENDS,
            (
                "flashdec_triton",
                "flashinfer_fa2_cuda_core",
                "flashinfer_fa2_tensor_core",
            ),
        )

    def test_layout_and_dependency_version_are_pre_registered(self):
        self.assertEqual(EXPECTED_FLASHINFER_VERSION, "0.6.15.post1")
        self.assertEqual(EXPECTED_TORCH_VERSION, "2.11.0+cu128")
        self.assertEqual(EXPECTED_TRITON_VERSION, "3.6.0")
        self.assertEqual(EXPECTED_TORCH_CUDA_VERSION, "12.8")
        self.assertEqual(EXPECTED_CUDA_TOOLKIT_VERSION, "12.8.1")
        self.assertEqual(EXPECTED_CUDA_PYTHON_VERSION, "12.9.1")
        self.assertEqual(EXPECTED_CUDA_BINDINGS_VERSION, "12.9.7")
        self.assertEqual(EXPECTED_CUDA_PATHFINDER_VERSION, "1.6.0")
        self.assertEqual(EXPECTED_NINJA_VERSION, "1.13.0")
        self.assertEqual(EXPECTED_NVCC_RELEASE, "12.8")
        self.assertEqual(EXPECTED_NVCC_VERSION, "12.8.93")
        self.assertEqual(EXPECTED_FLASHINFER_CUDA_ARCH_LIST, "12.0a")
        self.assertEqual(EXPECTED_PYTHON_MAJOR_MINOR, "3.12")
        self.assertEqual(FLASHDEC_KV_LAYOUT, "token_major")
        self.assertEqual(FLASHINFER_KV_LAYOUT, "HND")

    def test_canonical_r5_environment_is_returned_for_csv_evidence(self):
        environ, versions, toolkit = _canonical_environment()
        actual = _validate_r5_environment(
            _FakeTorch,
            environ=environ,
            version_getter=versions.__getitem__,
            python_version="3.12.3",
            cuda_probe=lambda _path: toolkit,
        )
        self.assertEqual(actual["torch"], "2.11.0+cu128")
        self.assertEqual(actual["cuda_toolkit"], "12.8.1")
        self.assertEqual(actual["cuda_home"], "/usr/local/cuda-12.8")
        self.assertEqual(actual["cuda_home_realpath"], "/usr/local/cuda-12.8")
        self.assertEqual(actual["nvcc_version"], "12.8.93")
        self.assertEqual(actual["flashinfer_cuda_arch_list"], "12.0a")

    def test_nvcc_version_parser_requires_release_and_full_version(self):
        output = "Cuda compilation tools, release 12.8, V12.8.93"
        self.assertEqual(_parse_nvcc_version(output), ("12.8", "12.8.93"))
        with self.assertRaisesRegex(RuntimeError, "cannot parse"):
            _parse_nvcc_version("Cuda compilation tools, unknown")

    def test_r5_environment_rejects_version_arch_or_toolkit_path_drift(self):
        mutations = (
            ("environment", "FLASHINFER_CUDA_ARCH_LIST", "", "arch_list"),
            ("environment", "FLASHINFER_CUDA_ARCH_LIST", "12.0", "arch_list"),
            ("environment", "CUDA_HOME", "/usr/local/cuda-13.0", "CUDA_HOME"),
            ("versions", "triton", "3.7.1", "triton"),
            ("versions", "cuda-toolkit", "13.0.3.0", "cuda_toolkit"),
            ("toolkit", "nvcc_version", "12.9.86", "nvcc_version"),
            (
                "toolkit",
                "cuda_home_realpath",
                "/usr/local/cuda-13.0",
                "realpath",
            ),
        )
        for target, field, value, message in mutations:
            with self.subTest(field=field, value=value):
                environ, versions, toolkit = _canonical_environment()
                selected = {
                    "environment": environ,
                    "versions": versions,
                    "toolkit": toolkit,
                }[target]
                selected[field] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    _validate_r5_environment(
                        _FakeTorch,
                        environ=environ,
                        version_getter=versions.__getitem__,
                        python_version="3.12.3",
                        cuda_probe=lambda _path: toolkit,
                    )

        environ, versions, toolkit = _canonical_environment()
        with self.assertRaisesRegex(RuntimeError, "python"):
            _validate_r5_environment(
                _FakeTorch,
                environ=environ,
                version_getter=versions.__getitem__,
                python_version="3.13.0",
                cuda_probe=lambda _path: toolkit,
            )

    def test_trial_orders_rotate_without_mutating_inputs(self):
        cases = tuple(CASES.values())
        dtypes = tuple(DTYPES)
        self.assertEqual(_backend_order(1), BACKENDS)
        self.assertEqual(_backend_order(2), BACKENDS[1:] + BACKENDS[:1])
        self.assertEqual(_backend_order(3), BACKENDS[2:] + BACKENDS[:2])
        self.assertEqual(_case_order(cases, 2), cases[1:] + cases[:1])
        self.assertEqual(_dtype_order(dtypes, 2), dtypes[1:] + dtypes[:1])
        self.assertEqual(tuple(CASES.values()), cases)
        self.assertEqual(tuple(DTYPES), dtypes)

    def test_selection_helpers_preserve_registered_order(self):
        self.assertEqual(_selected_cases("all"), tuple(CASES.values()))
        self.assertEqual(_selected_cases("medium"), (CASES["medium"],))
        self.assertEqual(_selected_dtypes("both"), DTYPES)
        self.assertEqual(_selected_dtypes("float16"), ("float16",))

    def test_page_table_digest_is_order_sensitive(self):
        first = _page_table_digest(_FakeTensor([2, 0, 1]))
        second = _page_table_digest(_FakeTensor([2, 0, 1]))
        reordered = _page_table_digest(_FakeTensor([0, 1, 2]))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, reordered)

    def test_logical_workload_bytes_are_backend_independent_payload(self):
        case = CASES["small"]
        expected = 2 * (
            case.num_seqs * 32 * 128
            + case.num_seqs * case.context_tokens * 8 * 128
        ) * 2
        for dtype_name in ("float16", "bfloat16"):
            self.assertEqual(
                _logical_workload_bytes(
                    num_seqs=case.num_seqs,
                    context_tokens=case.context_tokens,
                    dtype_name=dtype_name,
                ),
                expected,
            )

    def test_cli_defaults_match_formal_contract(self):
        args = parse_args([])
        self.assertEqual(args.case, "all")
        self.assertEqual(args.dtype, "both")
        self.assertEqual(args.trials, FORMAL_TRIALS)
        self.assertEqual(args.warmup, FORMAL_WARMUP)
        self.assertEqual(args.repeat, FORMAL_REPEATS)
        self.assertEqual(args.expected_flashinfer_version, "0.6.15.post1")
        self.assertEqual(args.flashinfer_backend, "fa2")
        self.assertFalse(args.quick)
        self.assertFalse(args.require_clean)

    def test_quick_caps_benchmark_counts(self):
        args = parse_args(
            ["--quick", "--trials", "9", "--warmup", "8", "--repeat", "100"]
        )
        self.assertEqual(
            (args.trials, args.warmup, args.repeat),
            (QUICK_TRIALS, QUICK_WARMUP, QUICK_REPEATS),
        )

    def test_clean_evidence_flag_is_explicit(self):
        self.assertTrue(parse_args(["--require-clean"]).require_clean)

    @mock.patch(
        "benchmarks.run_flashinfer_baseline._git_worktree_clean",
        return_value=False,
    )
    def test_require_clean_fails_before_gpu_imports(self, _clean):
        with self.assertRaisesRegex(SystemExit, "clean Git worktree"):
            main(["--require-clean"])

    def test_cli_rejects_non_positive_counts(self):
        for argv in (["--trials", "0"], ["--repeat", "0"], ["--workspace-mib", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
