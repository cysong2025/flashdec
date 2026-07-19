"""Dependency-free tests for the R4-B persistent metadata runner."""

from types import SimpleNamespace
import unittest

from benchmarks.run_persistent_transaction_metadata import (
    CASES,
    METADATA_PATHS,
    PROFILE_TIMING_SCOPE,
    WALL_TIMING_SCOPE,
    _counter_delta,
    _metadata_path_context,
    _quick_case,
    _raw_profile_range_totals,
    _run_tokens,
    _selected_cases,
    _trial_path_order,
    _validate_args,
    parse_args,
)


class _FakeCache:
    def __init__(self):
        self.materializations = 0
        self.reuses = 0
        self.state = object()
        self.views = []
        self.num_free_blocks = 7

    def _transaction_device_metadata_for_layer(self, state):
        assert state is self.state
        self.reuses += 1
        return "persistent"

    def _materialize_transaction_device_metadata_for_layer(self, state):
        assert state is self.state
        self.materializations += 1
        return SimpleNamespace(
            positions=object(),
            block_tables=object(),
            effective_seq_lens=object(),
        )

    def _write_token_layer_fused_cuda_for_engine(
        self, transaction, layer_idx, q, k, v, *, rotary_dim=None, base=10_000.0
    ):
        metadata = self._transaction_device_metadata_for_layer(self.state)
        return q, metadata

    def write_token_layer_fused_cuda(
        self, transaction, layer_idx, q, k, v, *, rotary_dim=None, base=10_000.0
    ):
        self._transaction_device_metadata_for_layer(self.state)
        return q, self._transaction_view(self.state)

    def _commit_token_for_engine(self, transaction):
        return None

    def _abort_token_for_engine(self, transaction):
        return None

    def _require_open_transaction(self, transaction):
        return self.state

    def _transaction_view(self, state):
        assert state is self.state
        self.materializations += 1
        view = SimpleNamespace(
            positions=object(),
            block_tables=object(),
            effective_seq_lens=object(),
        )
        self.views.append(view)
        return view

    def commit_token(self, transaction):
        return self._transaction_view(self.state)

    def abort_token(self, transaction):
        return self._transaction_view(self.state)

    def seq_lens_tensor(self, request_ids):
        return ("seq_lens", tuple(request_ids))


class _HookEngine:
    STEP_OK = "ok"

    def __init__(self, cache):
        self.cache = cache

    def _write_transaction_layer(
        self, state, layer_idx, q, k, v, *, rotary_dim=None, base=10_000.0
    ):
        return self.cache._write_token_layer_fused_cuda_for_engine(
            state.cache_transaction,
            layer_idx,
            q,
            k,
            v,
            rotary_dim=rotary_dim,
            base=base,
        )

    def _prepare_layer_step_result(self, state, layer_idx, output, metadata):
        return object()

    def _commit_cache_transaction_and_prepare_result(self, state):
        self.cache._commit_token_for_engine(state.cache_transaction)
        return object()


class _WallCache:
    def __init__(self):
        self.counters = {
            "transaction_metadata_build_count": 10,
            "transaction_metadata_materialization_count": 10,
            "transaction_metadata_reuse_count": 20,
            "transaction_metadata_release_count": 10,
            "resident_transaction_metadata_count": 0,
        }

    def metrics(self):
        return dict(self.counters)


class _WallEngine:
    def __init__(self, layers):
        self.cache = _WallCache()
        self.layers = layers
        self.calls = []

    def begin_step(self, request_ids):
        self.calls.append("begin")
        self.cache.counters["transaction_metadata_build_count"] += 1
        self.cache.counters["transaction_metadata_materialization_count"] += 1
        self.cache.counters["resident_transaction_metadata_count"] += 1
        return object()

    def step_layer(self, transaction, layer_idx, q, k, v):
        self.calls.append(f"layer{layer_idx}")
        self.cache.counters["transaction_metadata_reuse_count"] += 1

    def commit_step(self, transaction):
        self.calls.append("commit")
        self.cache.counters["transaction_metadata_release_count"] += 1
        self.cache.counters["resident_transaction_metadata_count"] -= 1


class _FakeCuda:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1

    def Event(self, *args, **kwargs):  # pragma: no cover - failure guard
        raise AssertionError("pure wall timing must not construct CUDA Events")


class PersistentTransactionMetadataBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _event(key, cpu_us, *, annotation=True, device="DeviceType.CPU"):
        return SimpleNamespace(
            key=key,
            device_type=device,
            is_user_annotation=annotation,
            cpu_time_total=cpu_us,
        )

    def test_matrix_is_the_frozen_eight_case_grid(self):
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

    def test_path_order_rotates_without_mutating_input(self):
        paths = list(METADATA_PATHS)
        self.assertEqual(
            _trial_path_order(paths, 0), ["materialized", "persistent"]
        )
        self.assertEqual(
            _trial_path_order(paths, 1), ["persistent", "materialized"]
        )
        self.assertEqual(paths, list(METADATA_PATHS))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _trial_path_order(paths, -1)

    def test_quick_preserves_layers_and_batch_but_reduces_context(self):
        quick = _quick_case(CASES["l4_b16_c1024"])
        self.assertEqual(quick.name, "l4_b16_c64")
        self.assertEqual((quick.num_layers, quick.batch_size), (4, 16))
        self.assertEqual(
            _selected_cases("l4_b16_c1024", True), [quick]
        )

    def test_materialized_context_recreates_exact_two_l_plus_two_views(self):
        cache = _FakeCache()
        engine = _HookEngine(cache)
        transaction = object()
        begin_view = cache._transaction_view(cache.state)
        state = SimpleNamespace(
            cache_transaction=transaction,
            handle=SimpleNamespace(transaction_id=7, request_ids=(0, 1, 2, 3)),
            last_output="last",
            needed_new_blocks=4,
        )
        with _metadata_path_context(engine, "materialized"):
            for layer_idx in range(4):
                _q, post_view = engine._write_transaction_layer(
                    state,
                    layer_idx,
                    "q",
                    "k",
                    "v",
                    rotary_dim=None,
                    base=10_000.0,
                )
                layer_result = engine._prepare_layer_step_result(
                    state, layer_idx, "output", post_view
                )
                self.assertIs(layer_result.positions, post_view.positions)
                self.assertIs(layer_result.block_tables, post_view.block_tables)
                self.assertIs(
                    layer_result.effective_seq_lens,
                    post_view.effective_seq_lens,
                )
            commit_result = engine._commit_cache_transaction_and_prepare_result(
                state
            )
        self.assertEqual(cache.materializations, 2 * 4 + 2)
        self.assertEqual(cache.reuses, 0)
        terminal_view = cache.views[-1]
        self.assertIs(commit_result.positions, terminal_view.positions)
        self.assertIs(commit_result.block_tables, terminal_view.block_tables)
        self.assertEqual(commit_result.seq_lens, ("seq_lens", (0, 1, 2, 3)))
        self.assertIsNot(begin_view, terminal_view)

        # Restored production hooks use one persistent bundle per layer.
        cache.materializations = 1
        for layer_idx in range(4):
            cache._write_token_layer_fused_cuda_for_engine(
                transaction, layer_idx, "q", "k", "v"
            )
        cache._commit_token_for_engine(transaction)
        self.assertEqual(cache.materializations, 1)
        self.assertEqual(cache.reuses, 4)

    def test_persistent_context_is_noop_and_unknown_path_is_rejected(self):
        cache = _FakeCache()
        engine = SimpleNamespace(cache=cache)
        original = cache._transaction_device_metadata_for_layer.__func__
        with _metadata_path_context(engine, "persistent"):
            self.assertIs(
                cache._transaction_device_metadata_for_layer.__func__, original
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            with _metadata_path_context(engine, "unknown"):
                pass

    def test_counter_delta_is_per_token_and_prefill_independent(self):
        before = {
            "transaction_metadata_build_count": 100,
            "transaction_metadata_materialization_count": 700,
            "transaction_metadata_reuse_count": 400,
            "transaction_metadata_release_count": 100,
            "resident_transaction_metadata_count": 0,
        }
        after = {
            "transaction_metadata_build_count": 105,
            "transaction_metadata_materialization_count": 705,
            "transaction_metadata_reuse_count": 420,
            "transaction_metadata_release_count": 105,
            "resident_transaction_metadata_count": 0,
        }
        delta = _counter_delta(before, after, 5)
        self.assertEqual(delta["metadata_builds_per_token"], 1)
        self.assertEqual(delta["metadata_materializations_per_token"], 1)
        self.assertEqual(delta["metadata_reuses_per_token"], 4)
        self.assertEqual(delta["metadata_releases_per_token"], 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            _counter_delta(before, after, 0)

    def test_wall_runner_synchronizes_without_cuda_events(self):
        cuda = _FakeCuda()
        torch = SimpleNamespace(cuda=cuda)
        engine = _WallEngine(layers=2)
        token = (("q0", "k0", "v0"), ("q1", "k1", "v1"))
        timings = _run_tokens(torch, engine, (0, 1, 2, 3), (token, token))
        self.assertEqual(cuda.synchronize_count, 4)
        self.assertEqual(len(timings["wall_ms"]), 2)
        self.assertEqual(
            engine.calls,
            ["begin", "layer0", "layer1", "commit"] * 2,
        )
        self.assertEqual(timings["metadata"]["metadata_builds_per_token"], 1)
        self.assertEqual(
            timings["metadata"]["metadata_materializations_per_token"], 1
        )
        self.assertEqual(timings["metadata"]["metadata_reuses_per_token"], 2)
        self.assertEqual(timings["metadata"]["metadata_resident_after"], 0)

    def test_cpu_profile_range_requires_exact_positive_annotations(self):
        events = (
            self._event("append", 100),
            self._event("append", 120),
            self._event("append", 999, annotation=False),
            self._event("append", 999, device="DeviceType.CUDA"),
        )
        totals = _raw_profile_range_totals(events, "append", 2)
        self.assertEqual(totals, {"count": 2, "cpu_time_us": 220.0})
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            _raw_profile_range_totals(events, "append", 1)

    def test_args_keep_both_paths_and_quick_bounds_work(self):
        args = parse_args(
            [
                "--case",
                "l4_b4_c128",
                "--dtype",
                "float16",
                "--trials",
                "3",
                "--quick",
            ]
        )
        args = _validate_args(args)
        self.assertEqual(args.metadata_paths, list(METADATA_PATHS))
        self.assertEqual(args.warmup, 1)
        self.assertEqual(args.repeat, 5)
        self.assertEqual(args.parity_steps, 1)
        self.assertIn("no CUDA events", WALL_TIMING_SCOPE)
        self.assertIn("CPU-only", PROFILE_TIMING_SCOPE)

        bad = parse_args(["--metadata-paths", "persistent"])
        with self.assertRaisesRegex(SystemExit, "exactly"):
            _validate_args(bad)


if __name__ == "__main__":
    unittest.main()
