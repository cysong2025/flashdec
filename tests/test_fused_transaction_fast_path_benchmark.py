"""Dependency-free configuration tests for the fused transaction A/B runner."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.run_fused_transaction_fast_path import (
    CASES,
    FastPathCase,
    IncompleteProfilerTrace,
    MAX_PROFILE_ATTEMPTS,
    PROFILE_TIMING_SCOPE,
    TRANSACTION_PATHS,
    WALL_TIMING_SCOPE,
    _max_blocks,
    _profiler_kwargs,
    _profile_probe,
    _profile_probe_once,
    _profile_scalar_counts,
    _quick_case,
    _raw_cpu_operator_count,
    _raw_profile_range_totals,
    _run_profile_token,
    _run_paired_trial,
    _run_tokens,
    _selected_cases,
    _selected_fused_append,
    _summarize_row,
    _transaction_path_context,
    _trial_path_order,
    _validate_profile_warmup_abort,
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

    def abort_step(self, transaction):
        self.calls.append(("abort",))


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
            PROFILE_TIMING_SCOPE,
            "separate CPU-only profiler attribution probe; device attribution excluded",
        )

    def test_profiler_accumulates_events_across_internal_cycles(self):
        schedule_calls = []
        torch = SimpleNamespace(
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu"),
                schedule=lambda **kwargs: schedule_calls.append(kwargs)
                or "warmup-active-schedule",
            )
        )
        self.assertEqual(
            _profiler_kwargs(torch),
            {
                "activities": ["cpu"],
                "schedule": "warmup-active-schedule",
                "acc_events": True,
            },
        )
        self.assertEqual(
            schedule_calls,
            [{"wait": 0, "warmup": 1, "active": 1, "repeat": 1}],
        )
        self.assertEqual(TRANSACTION_PATHS, ("checked", "trusted"))

    def test_raw_range_uses_only_cpu_annotation_inclusive_time(self):
        cpu = self._profile_event(
            "append",
            "DeviceType.CPU",
            is_user_annotation=True,
            cpu_time_total=400.0,
            device_time_total=0.0,
            self_cpu_time_total=0.0,
        )
        cuda = self._profile_event(
            "append",
            "DeviceType.CUDA",
            is_user_annotation=True,
            cpu_time_total=0.0,
            device_time_total=250.0,
        )
        decoy = self._profile_event(
            "append",
            "DeviceType.CUDA",
            is_user_annotation=False,
            device_time_total=999.0,
        )
        for events in ((cpu, cuda, decoy), (decoy, cuda, cpu), (cpu,)):
            with self.subTest(order=events):
                totals = _raw_profile_range_totals(events, "append", 1)
                self.assertEqual(totals["count"], 1)
                self.assertEqual(totals["cpu_time_us"], 400.0)
                self.assertNotIn("device_time_us", totals)

    def test_raw_range_sums_cpu_annotations(self):
        events = (
            self._profile_event(
                "append",
                "cpu",
                is_user_annotation=True,
                cpu_time_total=100.0,
                device_time_total=60.0,
            ),
            self._profile_event(
                "append",
                "cpu",
                is_user_annotation=True,
                cpu_time_total=120.0,
                device_time_total=40.0,
            ),
        )
        totals = _raw_profile_range_totals(events, "append", 2)
        self.assertEqual(totals["count"], 2)
        self.assertEqual(totals["cpu_time_us"], 220.0)

    def test_raw_range_rejects_missing_extra_or_nonpositive_cpu_time(self):
        non_annotation = self._profile_event(
            "append",
            "cpu",
            cpu_time_total=100.0,
            device_time_total=50.0,
        )
        with self.assertRaisesRegex(IncompleteProfilerTrace, "found CPU=0"):
            _raw_profile_range_totals((non_annotation,), "append", 1)

        zero_cpu = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=0.0,
            device_time_total=0.0,
        )
        with self.assertRaisesRegex(
            IncompleteProfilerTrace, "inclusive CPU time"
        ):
            _raw_profile_range_totals((zero_cpu,), "append", 1)

        cpu = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=100.0,
            device_time_total=float("nan"),
        )
        totals = _raw_profile_range_totals((cpu,), "append", 1)
        self.assertEqual(totals["cpu_time_us"], 100.0)

        extra_cpu = self._profile_event(
            "append",
            "cpu",
            is_user_annotation=True,
            cpu_time_total=100.0,
            device_time_total=50.0,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            _raw_profile_range_totals((extra_cpu, extra_cpu), "append", 1)

    def test_raw_range_accepts_eight_cpu_ranges_despite_seven_cuda_peers(self):
        cpu_ranges = tuple(
            self._profile_event(
                "append",
                "cpu",
                is_user_annotation=True,
                cpu_time_total=350.0 + index,
                device_time_total=0.0 if index == 0 else 3.5,
            )
            for index in range(8)
        )
        cuda_ranges = tuple(
            self._profile_event(
                "append",
                "cuda",
                is_user_annotation=True,
                device_time_total=300.0 + index,
            )
            for index in range(7)
        )
        totals = _raw_profile_range_totals(
            cpu_ranges + cuda_ranges,
            "append",
            8,
        )
        self.assertEqual(totals["count"], 8)
        self.assertEqual(totals["cpu_time_us"], 2828.0)

    def test_raw_operator_count_does_not_collapse_device_groups(self):
        events = (
            self._profile_event("aten::item", "cpu"),
            self._profile_event("aten::item", "cpu"),
            self._profile_event("aten::item", "cuda"),
            self._profile_event("kernel", "cuda"),
            self._profile_event(
                "ProfilerStep#1",
                "cuda",
                is_user_annotation=True,
            ),
        )
        self.assertEqual(_raw_cpu_operator_count(events, "aten::item"), 2)

    def test_profile_scalar_counts_are_part_of_capture_completeness(self):
        checked_events = tuple(
            self._profile_event("aten::item", "cpu") for _ in range(10)
        ) + tuple(
            self._profile_event("aten::_local_scalar_dense", "cpu")
            for _ in range(10)
        )
        self.assertEqual(
            _profile_scalar_counts(checked_events, "checked", 2),
            {
                "profile_item_count": 10,
                "profile_local_scalar_dense_count": 10,
            },
        )
        self.assertEqual(
            _profile_scalar_counts((), "trusted", 2),
            {
                "profile_item_count": 0,
                "profile_local_scalar_dense_count": 0,
            },
        )
        with self.assertRaisesRegex(IncompleteProfilerTrace, "incomplete"):
            _profile_scalar_counts(checked_events[:-1], "checked", 2)
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            _profile_scalar_counts(checked_events[:1], "trusted", 2)
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            _profile_scalar_counts(
                checked_events
                + (self._profile_event("aten::item", "cpu"),),
                "checked",
                2,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _profile_scalar_counts((), "unknown", 2)

    def test_profile_token_commits_or_aborts_through_public_engine_api(self):
        token = (("q0", "k0", "v0"), ("q1", "k1", "v1"))
        engine = _FakeEngine()
        _run_profile_token(engine, (1, 2), token, commit=False)
        self.assertEqual(
            [call[0] for call in engine.calls],
            ["begin", "layer", "layer", "abort"],
        )

        engine = _FakeEngine()
        _run_profile_token(engine, (1, 2), token, commit=True)
        self.assertEqual(
            [call[0] for call in engine.calls],
            ["begin", "layer", "layer", "commit"],
        )

    def test_profile_warmup_abort_restores_committed_state(self):
        class Cache:
            def request_state(self, _request_id):
                return {"seq_len": 128}

        class Engine:
            cache = Cache()

            def metrics(self):
                return {
                    "open_step_transaction_count": 0,
                    "cache": {"open_transaction_count": 0},
                }

            def validate_invariants(self):
                return True

        _validate_profile_warmup_abort(Engine(), (1, 2), 128)

        engine = Engine()
        engine.metrics = lambda: {
            "open_step_transaction_count": 1,
            "cache": {"open_transaction_count": 0},
        }
        with self.assertRaisesRegex(RuntimeError, "open Engine"):
            _validate_profile_warmup_abort(engine, (1, 2), 128)

        engine = Engine()
        engine.cache.request_state = lambda _request_id: {"seq_len": 129}
        with self.assertRaisesRegex(RuntimeError, "committed request length"):
            _validate_profile_warmup_abort(engine, (1, 2), 128)

    def test_profile_probe_uses_discarded_warmup_then_one_active_window(self):
        events = (object(),)
        calls = []

        class FakeProfiler:
            exited = False

            def __enter__(self):
                calls.append("enter")
                return self

            def step(self):
                calls.append("step")

            def __exit__(self, *_args):
                self.exited = True
                calls.append("exit")

            def events(self):
                self.assert_exited()
                calls.append("events")
                return events

            def assert_exited(self):
                if not self.exited:
                    raise AssertionError("events parsed before profiler exit")

        profiler = FakeProfiler()
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: calls.append("synchronize")
            ),
            profiler=SimpleNamespace(profile=lambda **_kwargs: profiler),
        )
        case = FastPathCase("l2_b4_c128", 2, 4, 128)
        active_inputs = (("active-1",), ("active-2",))
        warmup_inputs = (("warmup",),)

        def generate_inputs(_torch, _case, _dtype, count, _seed):
            return active_inputs if count == 2 else warmup_inputs

        def run_token(_engine, _request_ids, token, *, commit):
            calls.append(("token", token[0], commit))

        with (
            patch(
                "benchmarks.run_fused_transaction_fast_path._make_engine",
                return_value=(object(), (1, 2, 3, 4)),
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._seed_context"
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._generate_inputs",
                side_effect=generate_inputs,
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._profiler_kwargs",
                return_value={},
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._run_profile_token",
                side_effect=run_token,
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._validate_profile_warmup_abort",
                side_effect=lambda *_args: calls.append("validate-abort"),
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._raw_profile_range_totals",
                return_value={
                    "count": 4,
                    "cpu_time_us": 400.0,
                },
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._profile_scalar_counts",
                return_value={
                    "profile_item_count": 20,
                    "profile_local_scalar_dense_count": 20,
                },
            ),
        ):
            result = _profile_probe_once(
                torch,
                case,
                "dtype",
                2,
                701,
                "checked",
            )

        self.assertEqual(
            calls,
            [
                "enter",
                ("token", "warmup", False),
                "synchronize",
                "validate-abort",
                "step",
                ("token", "active-1", True),
                ("token", "active-2", True),
                "synchronize",
                "step",
                "exit",
                "events",
            ],
        )
        self.assertEqual(result["profile_token_count"], 2)
        self.assertEqual(result["profile_append_count"], 4)

    def test_profile_probe_retries_only_incomplete_traces(self):
        success = {"profile_steps": 2}
        with (
            patch(
                "benchmarks.run_fused_transaction_fast_path._profile_probe_once",
                side_effect=[
                    IncompleteProfilerTrace("missing first event"),
                    success,
                ],
            ) as probe_once,
            patch("sys.stderr", new=io.StringIO()),
        ):
            result = _profile_probe(
                "torch", "case", "dtype", 2, 701, "checked"
            )
        self.assertEqual(result["profile_attempt_count"], 2)
        self.assertEqual(result["profile_steps"], 2)
        self.assertEqual(
            probe_once.call_args_list[0],
            probe_once.call_args_list[1],
        )

        with (
            patch(
                "benchmarks.run_fused_transaction_fast_path._profile_probe_once",
                side_effect=IncompleteProfilerTrace("still incomplete"),
            ) as probe_once,
            patch("sys.stderr", new=io.StringIO()),
        ):
            with self.assertRaisesRegex(
                IncompleteProfilerTrace,
                f"after {MAX_PROFILE_ATTEMPTS} attempts",
            ):
                _profile_probe(
                    "torch", "case", "dtype", 2, 701, "checked"
                )
        self.assertEqual(probe_once.call_count, MAX_PROFILE_ATTEMPTS)

        with patch(
            "benchmarks.run_fused_transaction_fast_path._profile_probe_once",
            side_effect=RuntimeError("engine failure"),
        ) as probe_once:
            with self.assertRaisesRegex(RuntimeError, "engine failure"):
                _profile_probe(
                    "torch", "case", "dtype", 2, 701, "checked"
                )
        self.assertEqual(probe_once.call_count, 1)

        with self.assertRaisesRegex(ValueError, "max_attempts"):
            _profile_probe(
                "torch",
                "case",
                "dtype",
                2,
                701,
                "checked",
                max_attempts=MAX_PROFILE_ATTEMPTS + 1,
            )

    def test_paired_trial_finishes_both_wall_runs_before_attribution(self):
        calls = []
        args = SimpleNamespace(
            seed=701,
            transaction_paths=("checked", "trusted"),
            parity_steps=2,
        )
        case = FastPathCase("l2_b4_c128", 2, 4, 128)

        def wall(*call_args):
            path = call_args[4]
            calls.append(("wall", path))
            return {"path": path}

        def attribution(*call_args):
            path = call_args[5]
            wall_evidence = call_args[-1]
            self.assertEqual(wall_evidence["path"], path)
            calls.append(("attribution", path))
            return SimpleNamespace(metadata={"transaction_path": path})

        with (
            patch(
                "benchmarks.run_fused_transaction_fast_path._parity_probe",
                return_value={"validated": True},
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._run_wall_case",
                side_effect=wall,
            ),
            patch(
                "benchmarks.run_fused_transaction_fast_path._run_case_attribution",
                side_effect=attribution,
            ),
        ):
            paired = _run_paired_trial(
                "torch",
                args,
                case,
                "float16",
                "dtype",
                0,
                "abc1234",
                "run-abc1234",
            )

        self.assertEqual(
            calls,
            [
                ("wall", "checked"),
                ("wall", "trusted"),
                ("attribution", "checked"),
                ("attribution", "trusted"),
            ],
        )
        self.assertEqual(len(paired), 2)

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
            "profile_attempt_count": 1,
            "profile_token_count": 2,
            "profile_append_count": 4,
            "profile_decode_count": 4,
            "profile_append_cpu_ms_per_layer": 0.4,
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
        self.assertTrue(
            {
                "profile_cuda_event_count",
                "profile_append_device_ms_per_layer",
                "profile_decode_device_ms_per_layer",
            }.isdisjoint(row)
        )


if __name__ == "__main__":
    unittest.main()
