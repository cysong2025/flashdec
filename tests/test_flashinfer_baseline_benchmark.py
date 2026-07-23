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
    EXPECTED_FLASHINFER_VERSION,
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
    _selected_cases,
    _selected_dtypes,
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
        self.assertEqual(FLASHDEC_KV_LAYOUT, "token_major")
        self.assertEqual(FLASHINFER_KV_LAYOUT, "HND")

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
