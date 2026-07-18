"""Dependency-free configuration tests for the fused transaction A/B runner."""

from types import SimpleNamespace
import unittest

from benchmarks.run_fused_transaction_fast_path import (
    CASES,
    FastPathCase,
    PROFILE_TIMING_SCOPE,
    TRANSACTION_PATHS,
    WALL_TIMING_SCOPE,
    _max_blocks,
    _profiler_kwargs,
    _quick_case,
    _raw_cpu_operator_count,
    _raw_cuda_event_count,
    _raw_user_range_totals,
    _run_tokens,
    _selected_cases,
    _selected_fused_append,
    _summarize_row,
    _transaction_path_context,
    _trial_path_order,
    _with_speedup,
)
from benchmarks.summarize_fused_transaction_fast_path import REQUIRED_FIELDS


class _FakeCuda:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1

    def Event(self, *args, **kwargs):  # pragma: no cover - failure guard
        raise AssertionError("pure wall timing must not construct CUDA events")


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def begin_step(self, request_ids):
        self.calls.append(("begin", tuple(request_ids)))
        return object()

    def step_layer(self, transaction, layer_idx, q, k, v):
        self.calls.append(("layer", layer_idx, q, k, v))

    def commit_step(self, transaction):
        self.calls.append(("commit",))


class FusedTransactionFastPathBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _profile_event(
        key,
        device_type,
        *,
        is_user_annotation=False,
        cpu_time_total=0.0,
        device_time_total=0.0,
        self_cpu_time_total=0.0,
    ):
        return SimpleNamespace(
            key=key,
            device_type=device_type,
            is_user_annotation=is_user_annotation,
            cpu_time_total=cpu_time_total,
            device_time_total=device_time_total,
            self_cpu_time_total=self_cpu_time_total,
        )

    def test_matrix_is_bounded_to_l2_l4_representative_shapes(self):
        self.assertEqual(len(CASES), 8)
        self.assertEqual(
            {
                (case.num_layers, case.batch_size, case.context_tokens)
                for case in CASES.values()
            },
            {
                (layers, batch, context)
                for layers in (2, 4)
                for batch in (4, 16)
                for context in (128, 1024)
            },
        )
        self.assertFalse(any(name.startswith("l1_") for name in CASES))

    def test_path_order_alternates_without_mutating_input(self):
        paths = ["checked", "trusted"]
        self.assertEqual(_trial_path_order(paths, 0), ["checked", "trusted"])
        self.assertEqual(_trial_path_order(paths, 1), ["trusted", "checked"])
        self.assertEqual(paths, ["checked", "trusted"])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _trial_path_order(paths, -1)

    def test_checked_and_trusted_select_distinct_raw_functions(self):
        checked = object()
        trusted = object()
        module = SimpleNamespace(
            fused_rope_kv_append=checked,
            _fused_rope_kv_append_trusted=trusted,
        )
        self.assertIs(_selected_fused_append(module, "checked"), checked)
        self.assertIs(_selected_fused_append(module, "trusted"), trusted)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _selected_fused_append(module, "unknown")

    def test_transaction_path_context_routes_and_restores_module_function(self):
        import flashdec._fused_rope_kv_append as fused_module

        trusted = fused_module._fused_rope_kv_append_trusted
        checked = fused_module.fused_rope_kv_append
        with _transaction_path_context("checked"):
            self.assertIs(fused_module._fused_rope_kv_append_trusted, checked)
        self.assertIs(fused_module._fused_rope_kv_append_trusted, trusted)

        with _transaction_path_context("trusted"):
            self.assertIs(fused_module._fused_rope_kv_append_trusted, trusted)
        self.assertIs(fused_module._fused_rope_kv_append_trusted, trusted)

    def test_quick_case_preserves_layer_batch_and_reduces_context(self):
        case = CASES["l4_b16_c1024"]
        quick = _quick_case(case)
        self.assertEqual(quick.name, "l4_b16_c64")
        self.assertEqual(quick.num_layers, 4)
        self.assertEqual(quick.batch_size, 16)
        self.assertEqual(quick.context_tokens, 64)
        self.assertEqual(_selected_cases("l4_b16_c1024", True), [quick])

    def test_max_blocks_covers_measurement_and_next_boundary(self):
        self.assertEqual(_max_blocks(CASES["l2_b4_c128"], 20), 20)

    def test_wall_runner_synchronizes_without_cuda_events(self):
        cuda = _FakeCuda()
        torch = SimpleNamespace(cuda=cuda)
        engine = _FakeEngine()
        token = (("q0", "k0", "v0"), ("q1", "k1", "v1"))
        timings = _run_tokens(torch, engine, (1, 2), (token, token))

        self.assertEqual(cuda.synchronize_count, 4)
        self.assertEqual(len(timings["wall_ms"]), 2)
        self.assertEqual(len(timings["begin_host_ms"]), 2)
        self.assertEqual(len(timings["commit_host_ms"]), 2)
        self.assertEqual(
            [call[0] for call in engine.calls],
            ["begin", "layer", "layer", "commit"] * 2,
        )
        self.assertIn("no CUDA events", WALL_TIMING_SCOPE)
        self.assertEqual(
            PROFILE_TIMING_SCOPE, "separate instrumented attribution probe"
        )

    def test_profiler_accumulates_events_across_internal_cycles(self):
        torch = SimpleNamespace(
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")
            )
        )
        self.assertEqual(
            _profiler_kwargs(torch),
            {"activities": ["cpu", "cuda"], "acc_events": True},
        )
        self.assertEqual(TRANSACTION_PATHS, ("checked", "trusted"))

    def test_raw_range_uses_cpu_annotation_inclusive_time_in_any_order(self):
        cpu = self._profile_event(
            "append",
            "DeviceType.CPU",
            is_user_annotation=True,
            cpu_time_total=400.0,
            device_time_total=250.0,
            self_cpu_time_total=0.0,
        )
        cuda = self._profile_event(
            "append",
            "DeviceType.CUDA",
            cpu_time_total=0.0,
            device_time_total=250.0,
        )
        for events in ((cpu, cuda), (cuda, cpu)):
            with self.subTest(order=events):
                totals = _raw_user_range_totals(events, "append", 1)
                self.assertEqual(totals["count"], 1)
                self.assertEqual(totals["cpu_time_us"], 400.0)
                self.assertEqual(totals["device_time_us"], 250.0)

    def test_raw_range_sums_multiple_cpu_annotations(self):
        events = (
            self._profile_event(
                "append",
                "cpu",
                is_user_annotation=True,
                cpu_time_total=100.0,
                device_time_total=40.0,
            ),
            self._profile_event(
                "append",
                "cpu",
                is_user_annotation=True,
                cpu_time_total=120.0,
                device_time_total=60.0,
            ),
        )
        totals = _raw_user_range_totals(events, "append", 2)
        self.assertEqual(totals["count"], 2)
        self.assertEqual(totals["cpu_time_us"], 220.0)
        self.assertEqual(totals["device_time_us"], 100.0)

    def test_raw_range_rejects_missing_annotation_or_nonpositive_time(self):
        non_annotation = self._profile_event(
            "append",
            "cpu",
            cpu_time_total=100.0,
            device_time_total=50.0,
        )
        with self.assertRaisesRegex(RuntimeError, "raw CPU user annotations"):
            _raw_user_range_totals((non_annotation,), "append", 1)

        zero_cpu = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=0.0,
            device_time_total=50.0,
        )
        with self.assertRaisesRegex(RuntimeError, "inclusive CPU time"):
            _raw_user_range_totals((zero_cpu,), "append", 1)

        positive = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=100.0,
            device_time_total=50.0,
        )
        invalid_device = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=100.0,
            device_time_total=float("nan"),
        )
        with self.assertRaisesRegex(RuntimeError, "inclusive device time"):
            _raw_user_range_totals(
                (positive, invalid_device), "append", 2
            )

    def test_raw_operator_and_cuda_counts_do_not_collapse_device_groups(self):
        events = (
            self._profile_event("aten::item", "cpu"),
            self._profile_event("aten::item", "cpu"),
            self._profile_event("aten::item", "cuda"),
            self._profile_event("kernel", "cuda"),
        )
        self.assertEqual(_raw_cpu_operator_count(events, "aten::item"), 2)
        self.assertEqual(_raw_cuda_event_count(events), 2)

    def test_runner_row_schema_matches_strict_summary_schema(self):
        case = FastPathCase("l2_b4_c128", 2, 4, 128)
        repeats = 20
        seq_len = case.context_tokens + repeats
        blocks_per_request = 5
        max_blocks = case.batch_size * blocks_per_request

        class Cache:
            dtype = "float16"

            def request_state(self, _request_id):
                return {"seq_len": seq_len}

            def request_block_ids(self, _request_id):
                return tuple(range(blocks_per_request))

        cache_metrics = {
            "max_blocks": max_blocks,
            "used_blocks": max_blocks,
            "free_blocks": 0,
            "allocation_count": max_blocks,
            "fresh_allocation_count": max_blocks,
            "reuse_count": 0,
            "capacity_failure_count": 0,
            "transaction_begin_count": seq_len,
            "transaction_commit_count": seq_len,
            "transaction_abort_count": 0,
            "transaction_layer_write_count": seq_len * case.num_layers,
        }

        class Engine:
            cache = Cache()

            def metrics(self):
                return {
                    "cache": cache_metrics,
                    "completed_step_count": repeats,
                    "appended_token_count": repeats * case.batch_size,
                }

            def validate_invariants(self):
                return True

        torch = SimpleNamespace(
            __version__="2.11.0+cu128",
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                get_device_name=lambda _device: "NVIDIA GeForce RTX 5070",
            ),
        )
        args = SimpleNamespace(warmup=3, trials=3)
        profile = {
            "profile_steps": 2,
            "profile_token_count": 2,
            "profile_append_count": 4,
            "profile_decode_count": 4,
            "profile_cuda_event_count": 80,
            "profile_append_cpu_ms_per_layer": 0.4,
            "profile_append_device_ms_per_layer": 0.4,
            "profile_decode_device_ms_per_layer": 0.8,
            "profile_item_count": 20,
            "profile_local_scalar_dense_count": 20,
        }
        rollback = {
            "rollback_repeats": 2,
            "rollback_p50_ms": 1.5,
            "rollback_blocks": 8,
            "rollback_validated": True,
        }
        parity = {
            "parity_steps": 2,
            "parity_output_equal": True,
            "parity_cache_equal": True,
            "parity_state_equal": True,
            "parity_validated": True,
        }
        timings = {
            "wall_ms": [4.0] * repeats,
            "begin_host_ms": [0.5] * repeats,
            "commit_host_ms": [0.25] * repeats,
        }
        result = _summarize_row(
            torch,
            args,
            case,
            "float16",
            "float16",
            "checked",
            1,
            701,
            ("checked", "trusted"),
            timings,
            profile,
            rollback,
            parity,
            Engine(),
            "abc1234",
            "run-abc1234",
        )
        row = _with_speedup(result, result).as_row()
        self.assertEqual(set(row), REQUIRED_FIELDS)


if __name__ == "__main__":
    unittest.main()
