"""Dynamic decode execution engine for the FlashDec runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Hashable


PROFILE_RANGE_PREFLIGHT = "flashdec::engine_preflight"
PROFILE_RANGE_APPEND = "flashdec::rope_kv_append"
PROFILE_RANGE_DECODE = "flashdec::paged_decode"


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class AdmissionResult:
    """State transition result for a request submitted to the engine."""

    request_id: Hashable
    status: str


@dataclass(frozen=True)
class DecodeStepResult:
    """One dynamic decode-step result, or an explicit backpressure response."""

    status: str
    request_ids: tuple[Hashable, ...]
    output: Any | None
    positions: Any | None
    block_tables: Any | None
    seq_lens: Any | None
    needed_new_blocks: int
    free_blocks: int
    reason: str | None = None


@dataclass(frozen=True)
class DecodeStepTransaction:
    """Stable Engine handle for one open multi-layer token transaction."""

    transaction_id: int
    engine_version: int
    request_ids: tuple[Hashable, ...]
    positions: Any
    physical_block_ids: Any
    block_offsets: Any
    block_tables: Any
    effective_seq_lens: Any


@dataclass(frozen=True)
class DecodeLayerResult:
    """Output of one layer inside an open decode token transaction."""

    transaction_id: int
    layer_idx: int
    request_ids: tuple[Hashable, ...]
    output: Any
    positions: Any
    block_tables: Any
    effective_seq_lens: Any


@dataclass
class _EngineStepTransactionState:
    handle: DecodeStepTransaction
    cache_transaction: Any
    needed_new_blocks: int
    next_layer_idx: int = 0
    last_output: Any | None = None
    state: str = "open"


class DecodeEngine:
    """Coordinate request lifecycle, RoPE/KV append, and paged decode.

    The transaction API models one decode token across sequential layers.
    Model projection, sampling, full-model prefill, and network serving remain
    outside this project boundary.  The original single-layer :meth:`step`
    API remains available for compatibility.

    ``append_backend`` selects ``torch``, ``cuda``, or ``fused_cuda`` from the
    RoPE/KV data path. ``decode_backend='reference'`` keeps CPU and semantic
    tests available; ``'triton'`` uses the frozen paged decode kernel on CUDA.
    ``profile_ranges=True`` adds optional PyTorch profiler ranges without
    adding synchronization; it is disabled for normal execution/benchmarks.
    """

    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    STEP_OK = "ok"
    STEP_BACKPRESSURE = "backpressure"

    _APPEND_BACKENDS = ("torch", "cuda", "fused_cuda")
    _DECODE_BACKENDS = ("reference", "triton")

    def __init__(
        self,
        cache,
        append_backend="torch",
        decode_backend="reference",
        sm_scale=None,
        num_warps=2,
        num_stages=None,
        profile_ranges=False,
    ):
        from .cache import PagedKVCache

        if not isinstance(cache, PagedKVCache):
            raise TypeError("cache must be a PagedKVCache")
        if append_backend not in self._APPEND_BACKENDS:
            raise ValueError("append_backend must be 'torch', 'cuda', or 'fused_cuda'")
        if decode_backend not in self._DECODE_BACKENDS:
            raise ValueError("decode_backend must be 'reference' or 'triton'")
        if not isinstance(profile_ranges, bool):
            raise ValueError("profile_ranges must be a bool")

        self.cache = cache
        self.append_backend = append_backend
        self.decode_backend = decode_backend
        self.sm_scale = sm_scale
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.profile_ranges = profile_ranges
        self._profile_range_factory = (
            _torch().profiler.record_function if profile_ranges else None
        )
        self._statuses: dict[Hashable, str] = {}
        self._completed_step_count = 0
        self._appended_token_count = 0
        self._backpressure_count = 0
        self._state_version = 0
        self._observed_cache_state_version = cache.state_version
        self._scheduler_managed = False
        self._request_specs = {}
        self._stale_decision_count = 0
        self._applied_decision_count = 0
        self._pending_runnable_ids = None
        self._open_step_transaction: _EngineStepTransactionState | None = None
        self._transaction_layer_step_count = 0
        self._transaction_abort_count = 0

    def add_request(self, request_id):
        """Submit a new request in ``waiting`` state without allocating KV memory."""
        if self._scheduler_managed:
            raise RuntimeError("scheduler-managed engines require submit_request(RequestSpec)")
        return self._add_waiting_request(request_id, spec=None)

    def submit_request(self, spec):
        """Submit an immutable scheduler request specification.

        Calling this method enables scheduler-managed mode. Legacy
        ``add_request``/``admit`` remain available on engines that have not
        entered this mode, so existing unscheduled callers keep their API.
        """
        from .scheduler import RequestSpec

        if not isinstance(spec, RequestSpec):
            raise TypeError("spec must be a RequestSpec")
        if (
            not self._scheduler_managed
            and self.cache.state_version != self._observed_cache_state_version
        ):
            raise RuntimeError("cannot enable scheduler mode after external cache mutation")
        self._require_cache_synchronized()
        if self._statuses and not self._scheduler_managed:
            raise RuntimeError("cannot mix legacy requests with scheduler-managed requests")
        if not self._scheduler_managed:
            cache_metrics = self.cache.metrics()
            if any(
                cache_metrics[name]
                for name in ("active_requests", "finished_requests", "cancelled_requests")
            ):
                raise RuntimeError("scheduler-managed mode requires an empty PagedKVCache")
        self._scheduler_managed = True
        return self._add_waiting_request(spec.request_id, spec=spec)

    def _add_waiting_request(self, request_id, spec):
        self._require_no_open_step("submit a request")
        self._check_request_id(request_id)
        if request_id in self._statuses:
            raise RuntimeError(f"request_id {request_id!r} already exists in the DecodeEngine")
        if spec is not None:
            if any(
                existing.submission_order == spec.submission_order
                for existing in self._request_specs.values()
            ):
                raise ValueError("submission_order must be unique within one DecodeEngine")
            self._request_specs[request_id] = spec
        self._statuses[request_id] = self.WAITING
        self._bump_state_version()
        return AdmissionResult(request_id=request_id, status=self.WAITING)

    def admit(self, request_ids=None):
        """Move waiting requests to active state without reserving physical blocks."""
        self._require_no_open_step("admit requests")
        if self._scheduler_managed:
            raise RuntimeError("scheduler-managed engines require apply_scheduler_decision")
        ids = self._normalize_waiting_ids(request_ids)
        self._admit_ids(ids)
        self._bump_state_version()
        return tuple(AdmissionResult(request_id=request_id, status=self.ACTIVE) for request_id in ids)

    @property
    def state_version(self):
        """Return the monotonic Engine lifecycle/sequence version."""
        return self._state_version

    def scheduling_snapshot(
        self,
        logical_step,
        *,
        waiting_wait_steps=None,
        waiting_skip_counts=None,
        active_service_wait_steps=None,
    ):
        """Build scheduler metadata from authoritative Engine/Cache state.

        Scheduler-owned fairness counters are supplied as mappings; request
        specs, sequence lengths, physical ownership, and commitments are
        derived or validated here and cannot be forged by the caller.
        """
        from .scheduler import (
            ActiveRequestMetadata,
            SchedulingSnapshot,
            WaitingRequestMetadata,
        )

        if not self._scheduler_managed:
            raise RuntimeError("scheduling_snapshot requires submit_request(RequestSpec)")
        self._require_no_open_step("build a scheduling snapshot")
        self._require_cache_synchronized()
        if isinstance(logical_step, bool) or not isinstance(logical_step, int) or logical_step < 0:
            raise ValueError("logical_step must be a non-negative integer")

        waiting_ids = tuple(
            request_id
            for request_id, status in self._statuses.items()
            if status == self.WAITING
        )
        active_ids = self.active_request_ids()
        wait_steps = self._normalize_counter_mapping(
            waiting_wait_steps, waiting_ids, "waiting_wait_steps"
        )
        skip_counts = self._normalize_counter_mapping(
            waiting_skip_counts, waiting_ids, "waiting_skip_counts"
        )
        service_wait = self._normalize_counter_mapping(
            active_service_wait_steps, active_ids, "active_service_wait_steps"
        )

        waiting = tuple(
            WaitingRequestMetadata(
                self._request_specs[request_id],
                wait_steps=wait_steps[request_id],
                skip_count=skip_counts[request_id],
            )
            for request_id in sorted(
                waiting_ids,
                key=lambda item: self._request_specs[item].submission_order,
            )
        )
        active = []
        for request_id in active_ids:
            spec = self._request_specs[request_id]
            cache_state = self.cache.request_state(request_id)
            seq_len = cache_state["seq_len"]
            completed_tokens = seq_len - spec.initial_context_tokens
            if completed_tokens < 0:
                raise RuntimeError(
                    f"request_id {request_id!r} initial context has not been fully seeded"
                )
            remaining_tokens = spec.max_new_tokens - completed_tokens
            if remaining_tokens <= 0:
                raise RuntimeError(
                    f"request_id {request_id!r} exhausted max_new_tokens but remains active"
                )
            active.append(
                ActiveRequestMetadata(
                    spec=spec,
                    seq_len=seq_len,
                    remaining_tokens=remaining_tokens,
                    physical_blocks=len(cache_state["block_ids"]),
                    committed_blocks=spec.commitment_blocks(self.cache.block_size),
                    service_wait_steps=service_wait[request_id],
                )
            )

        return SchedulingSnapshot(
            state_version=self._state_version,
            logical_step=logical_step,
            block_size=self.cache.block_size,
            max_blocks=self.cache.max_blocks,
            free_blocks=self.cache.num_free_blocks,
            waiting=waiting,
            active=tuple(active),
        )

    def apply_scheduler_decision(self, decision):
        """Atomically apply admission/rejection from one fresh decision."""
        from .scheduler import SchedulerDecision

        if not isinstance(decision, SchedulerDecision):
            raise TypeError("decision must be a SchedulerDecision")
        if not self._scheduler_managed:
            raise RuntimeError("apply_scheduler_decision requires scheduler-managed mode")
        self._require_no_open_step("apply a scheduler decision")
        self._require_cache_synchronized(stale=True)
        if decision.snapshot_version != self._state_version:
            self._stale_decision_count += 1
            raise RuntimeError("stale scheduler decision")
        if self._pending_runnable_ids is not None:
            raise RuntimeError("a scheduler decision is already pending execution")

        waiting_ids = tuple(
            request_id
            for request_id, status in self._statuses.items()
            if status == self.WAITING
        )
        active_ids = self.active_request_ids()
        admit_ids = self._validate_decision_ids("admit_ids", decision.admit_ids)
        rejected_ids = self._validate_decision_ids("rejected_ids", decision.rejected_ids)
        runnable_ids = self._validate_decision_ids("runnable_ids", decision.runnable_ids)
        deferred_ids = self._validate_decision_ids("deferred_ids", decision.deferred_ids)
        decision_waiting_ids = self._validate_decision_ids(
            "waiting_ids", decision.waiting_ids
        )
        if set(admit_ids) & set(rejected_ids):
            raise ValueError("admit_ids and rejected_ids must be disjoint")
        if not set(admit_ids).issubset(waiting_ids):
            raise ValueError("admit_ids must contain waiting requests")
        if not set(rejected_ids).issubset(waiting_ids):
            raise ValueError("rejected_ids must contain waiting requests")

        active_after = set(active_ids) | set(admit_ids)
        if set(runnable_ids) & set(deferred_ids):
            raise ValueError("runnable_ids and deferred_ids must be disjoint")
        if set(runnable_ids) | set(deferred_ids) != active_after:
            raise ValueError("runnable_ids and deferred_ids must partition active requests")
        waiting_after = set(waiting_ids) - set(admit_ids) - set(rejected_ids)
        if set(decision_waiting_ids) != waiting_after:
            raise ValueError("waiting_ids do not match the post-decision waiting set")
        if decision.free_blocks_before_step != self.cache.num_free_blocks:
            raise RuntimeError("scheduler decision free-block snapshot is stale")

        committed_before = sum(
            self._request_specs[request_id].commitment_blocks(self.cache.block_size)
            for request_id in active_ids
        )
        committed_after = committed_before + sum(
            self._request_specs[request_id].commitment_blocks(self.cache.block_size)
            for request_id in admit_ids
        )
        if decision.committed_blocks_before != committed_before:
            raise ValueError("committed_blocks_before does not match Engine state")
        if decision.committed_blocks_after != committed_after:
            raise ValueError("committed_blocks_after does not match admitted requests")
        if committed_after > self.cache.max_blocks:
            raise ValueError("committed blocks exceed physical cache capacity")

        changed = bool(admit_ids or rejected_ids)
        if admit_ids:
            self._admit_ids(admit_ids)
        for request_id in rejected_ids:
            self._statuses[request_id] = self.REJECTED
        if changed:
            self._bump_state_version()
        self._applied_decision_count += 1

        contexts_ready = all(
            self.cache.request_state(request_id)["seq_len"]
            >= self._request_specs[request_id].initial_context_tokens
            for request_id in runnable_ids
        )
        if runnable_ids and contexts_ready:
            self._pending_runnable_ids = runnable_ids
        return tuple(
            AdmissionResult(request_id, self._statuses[request_id])
            for request_id in (*admit_ids, *rejected_ids)
        )

    def prefill_request(self, request_id, k, v, layer_idx=0):
        """Seed one scheduler-managed prompt token outside decode timing."""
        if not self._scheduler_managed:
            raise RuntimeError("prefill_request requires scheduler-managed mode")
        self._require_no_open_step("prefill a request")
        if self.cache.num_layers != 1:
            raise RuntimeError("multi-layer prompt prefill is not implemented in R2-B")
        self._require_cache_synchronized()
        self._require_status(request_id, self.ACTIVE)
        spec = self._request_specs[request_id]
        current = self.cache.request_state(request_id)["seq_len"]
        if current >= spec.initial_context_tokens:
            raise RuntimeError("request initial context is already fully seeded")
        self.cache.append(layer_idx, [request_id], k, v)
        self._sync_cache_version()
        self._bump_state_version()
        return self.cache.request_state(request_id)["seq_len"]

    def active_request_ids(self):
        """Return active request ids in deterministic admission order."""
        return tuple(
            request_id
            for request_id, status in self._statuses.items()
            if status == self.ACTIVE
        )

    def request_state(self, request_id):
        """Return engine state plus cache state when the request has been admitted."""
        status = self._require_known_request(request_id)
        result = {"request_id": request_id, "status": status}
        if status not in (self.WAITING, self.REJECTED):
            result["cache"] = self.cache.request_state(request_id)
        spec = self._request_specs.get(request_id)
        if spec is not None:
            result["spec"] = spec
        return result

    def begin_step(self, request_ids=None):
        """Begin one token transaction for a stable batch of active requests.

        R2-B supports the readable PyTorch append path.  Native/fused
        location-only transaction writes are added in R2-C; their existing
        single-layer ``step()`` path remains unchanged until then.
        """
        self._require_no_open_step("begin another decode step")
        self._require_cache_synchronized()
        if self.append_backend != "torch":
            raise RuntimeError(
                "R2-B multi-layer transactions require append_backend='torch'"
            )
        ids = self._normalize_active_ids(request_ids)
        self._validate_scheduler_runnable_ids(ids)
        needed_new_blocks = self._needed_new_blocks(ids)
        if needed_new_blocks > self.cache.num_free_blocks:
            if self._scheduler_managed:
                raise RuntimeError(
                    "scheduler commitment invariant violated by physical backpressure"
                )
            self._backpressure_count += 1
            raise RuntimeError("insufficient physical blocks to begin decode step")

        cache_transaction = self.cache.begin_token(ids)
        try:
            handle = DecodeStepTransaction(
                transaction_id=cache_transaction.transaction_id,
                engine_version=self._state_version + 1,
                request_ids=ids,
                positions=cache_transaction.positions.clone(),
                physical_block_ids=cache_transaction.physical_block_ids.clone(),
                block_offsets=cache_transaction.block_offsets.clone(),
                block_tables=cache_transaction.block_tables.clone(),
                effective_seq_lens=cache_transaction.effective_seq_lens.clone(),
            )
        except Exception:
            self.cache.abort_token(cache_transaction)
            self._sync_cache_version()
            self._bump_state_version()
            raise
        self._sync_cache_version()
        self._bump_state_version()
        self._open_step_transaction = _EngineStepTransactionState(
            handle=handle,
            cache_transaction=cache_transaction,
            needed_new_blocks=needed_new_blocks,
        )
        return handle

    def step_layer(
        self,
        transaction,
        layer_idx,
        q,
        k,
        v,
        *,
        rotary_dim=None,
        base=10_000.0,
    ):
        """Run RoPE, reference transaction write, and paged decode for one layer.

        Any input, write, or decode exception aborts the whole open token so
        committed lengths and block ownership remain unchanged.
        """
        state = self._require_open_step(transaction)
        try:
            layer_idx = int(layer_idx)
            if layer_idx != state.next_layer_idx:
                raise RuntimeError(
                    f"decode transaction requires layer {state.next_layer_idx}, "
                    f"got layer {layer_idx}"
                )
            self._validate_step_inputs(state.handle.request_ids, q, k, v)
            if k.dim() == 2:
                k = k.unsqueeze(0)
            if v.dim() == 2:
                v = v.unsqueeze(0)

            range_factory = self._profile_range_factory
            if range_factory is None:
                q_rotated, cache_view = self._write_transaction_layer(
                    state,
                    layer_idx,
                    q,
                    k,
                    v,
                    rotary_dim=rotary_dim,
                    base=base,
                )
                output = self._decode(
                    q_rotated,
                    cache_view.block_tables,
                    cache_view.effective_seq_lens,
                    layer_idx=layer_idx,
                )
            else:
                with range_factory(PROFILE_RANGE_APPEND):
                    q_rotated, cache_view = self._write_transaction_layer(
                        state,
                        layer_idx,
                        q,
                        k,
                        v,
                        rotary_dim=rotary_dim,
                        base=base,
                    )
                with range_factory(PROFILE_RANGE_DECODE):
                    output = self._decode(
                        q_rotated,
                        cache_view.block_tables,
                        cache_view.effective_seq_lens,
                        layer_idx=layer_idx,
                    )
        except Exception:
            self._abort_open_step_state(state)
            raise

        state.next_layer_idx += 1
        state.last_output = output
        self._transaction_layer_step_count += 1
        return DecodeLayerResult(
            transaction_id=state.handle.transaction_id,
            layer_idx=layer_idx,
            request_ids=state.handle.request_ids,
            output=output,
            positions=cache_view.positions,
            block_tables=cache_view.block_tables,
            effective_seq_lens=cache_view.effective_seq_lens,
        )

    def commit_step(self, transaction):
        """Commit an open token after every cache layer has executed."""
        state = self._require_open_step(transaction)
        if state.next_layer_idx != self.cache.num_layers:
            raise RuntimeError("cannot commit decode step before all layers are complete")
        committed = self.cache.commit_token(state.cache_transaction)
        state.state = "committed"
        self._open_step_transaction = None
        self._completed_step_count += 1
        self._appended_token_count += len(state.handle.request_ids)
        self._sync_cache_version()
        self._bump_state_version()
        return DecodeStepResult(
            status=self.STEP_OK,
            request_ids=state.handle.request_ids,
            output=state.last_output,
            positions=committed.positions,
            block_tables=committed.block_tables,
            seq_lens=self.cache.seq_lens_tensor(state.handle.request_ids),
            needed_new_blocks=state.needed_new_blocks,
            free_blocks=self.cache.num_free_blocks,
        )

    def abort_step(self, transaction):
        """Explicitly abort one open token transaction."""
        state = self._require_open_step(transaction)
        return self._abort_open_step_state(state)

    def step(
        self,
        q,
        k,
        v,
        request_ids=None,
        layer_idx=0,
        rotary_dim=None,
        base=10_000.0,
    ):
        """Execute one append -> paged decode step for active request rows.

        On capacity pressure this returns a ``status='backpressure'`` result
        without changing cache ownership, request seq_len, or lifecycle state.
        The supplied row order is preserved in all successful result tensors.
        """
        self._require_no_open_step("run the single-layer step wrapper")
        if self.cache.num_layers != 1:
            raise RuntimeError(
                "multi-layer caches require begin_step/step_layer/commit_step"
            )
        self._require_cache_synchronized()
        range_factory = self._profile_range_factory
        if range_factory is None:
            ids = self._normalize_active_ids(request_ids)
            self._validate_step_inputs(ids, q, k, v)
            needed_new_blocks = self._needed_new_blocks(ids)
        else:
            with range_factory(PROFILE_RANGE_PREFLIGHT):
                ids = self._normalize_active_ids(request_ids)
                self._validate_step_inputs(ids, q, k, v)
                needed_new_blocks = self._needed_new_blocks(ids)
        self._validate_scheduler_runnable_ids(ids)
        if needed_new_blocks > self.cache.num_free_blocks:
            if self._scheduler_managed:
                raise RuntimeError(
                    "scheduler commitment invariant violated by physical backpressure"
                )
            self._backpressure_count += 1
            return DecodeStepResult(
                status=self.STEP_BACKPRESSURE,
                request_ids=ids,
                output=None,
                positions=None,
                block_tables=None,
                seq_lens=None,
                needed_new_blocks=needed_new_blocks,
                free_blocks=self.cache.num_free_blocks,
                reason="insufficient_physical_blocks",
            )

        if self.append_backend == "torch":
            if int(layer_idx) != 0:
                raise ValueError("single-layer step requires layer_idx=0")
            transaction = self.begin_step(ids)
            self.step_layer(
                transaction,
                0,
                q,
                k,
                v,
                rotary_dim=rotary_dim,
                base=base,
            )
            return self.commit_step(transaction)

        from .rope import rope_paged_kv_append

        if range_factory is None:
            append_result = rope_paged_kv_append(
                self.cache,
                layer_idx,
                ids,
                q,
                k,
                v,
                rotary_dim=rotary_dim,
                base=base,
                append_backend=self.append_backend,
            )
            output = self._decode(
                append_result.q,
                append_result.block_tables,
                append_result.seq_lens,
            )
        else:
            with range_factory(PROFILE_RANGE_APPEND):
                append_result = rope_paged_kv_append(
                    self.cache,
                    layer_idx,
                    ids,
                    q,
                    k,
                    v,
                    rotary_dim=rotary_dim,
                    base=base,
                    append_backend=self.append_backend,
                )
            with range_factory(PROFILE_RANGE_DECODE):
                output = self._decode(
                    append_result.q,
                    append_result.block_tables,
                    append_result.seq_lens,
                )
        self._completed_step_count += 1
        self._appended_token_count += len(ids)
        self._sync_cache_version()
        self._bump_state_version()
        return DecodeStepResult(
            status=self.STEP_OK,
            request_ids=ids,
            output=output,
            positions=append_result.positions,
            block_tables=append_result.block_tables,
            seq_lens=append_result.seq_lens,
            needed_new_blocks=needed_new_blocks,
            free_blocks=self.cache.num_free_blocks,
        )

    def finish_request(self, request_id):
        """Finish an active request and release its physical cache blocks."""
        self._require_no_open_step("finish a request")
        self._require_cache_synchronized()
        self._require_status(request_id, self.ACTIVE)
        released = self.cache.finish_request(request_id)
        self._statuses[request_id] = self.FINISHED
        self._sync_cache_version()
        self._bump_state_version()
        return released

    def cancel_request(self, request_id):
        """Cancel an active request and release its physical cache blocks."""
        self._require_no_open_step("cancel a request")
        self._require_cache_synchronized()
        self._require_status(request_id, self.ACTIVE)
        released = self.cache.cancel_request(request_id)
        self._statuses[request_id] = self.CANCELLED
        self._sync_cache_version()
        self._bump_state_version()
        return released

    def metrics(self):
        """Return engine counters together with the underlying cache metrics."""
        return {
            "waiting_requests": sum(status == self.WAITING for status in self._statuses.values()),
            "active_requests": sum(status == self.ACTIVE for status in self._statuses.values()),
            "finished_requests": sum(status == self.FINISHED for status in self._statuses.values()),
            "cancelled_requests": sum(status == self.CANCELLED for status in self._statuses.values()),
            "rejected_requests": sum(status == self.REJECTED for status in self._statuses.values()),
            "completed_step_count": self._completed_step_count,
            "appended_token_count": self._appended_token_count,
            "backpressure_count": self._backpressure_count,
            "state_version": self._state_version,
            "cache_state_version": self.cache.state_version,
            "scheduler_managed": self._scheduler_managed,
            "committed_blocks": self._committed_blocks(),
            "committed_but_unallocated_blocks": (
                self._committed_blocks() - self.cache.num_used_blocks
                if self._scheduler_managed
                else 0
            ),
            "stale_decision_count": self._stale_decision_count,
            "applied_decision_count": self._applied_decision_count,
            "open_step_transaction_count": int(
                self._open_step_transaction is not None
            ),
            "transaction_layer_step_count": self._transaction_layer_step_count,
            "transaction_abort_count": self._transaction_abort_count,
            "append_backend": self.append_backend,
            "decode_backend": self.decode_backend,
            "cache": self.cache.metrics(),
        }

    def validate_invariants(self):
        """Check that engine lifecycle state agrees with cache ownership state."""
        self._require_cache_synchronized()
        self.cache.validate_invariants()
        if self._open_step_transaction is None:
            if self.cache.has_open_transaction:
                raise RuntimeError("cache has an open transaction not owned by DecodeEngine")
        else:
            state = self._open_step_transaction
            if not self.cache.has_open_transaction:
                raise RuntimeError("Engine transaction is open but cache transaction is not")
            cache_view = self.cache.transaction_view(state.cache_transaction)
            if cache_view.request_ids != state.handle.request_ids:
                raise RuntimeError("Engine/cache transaction request rows diverged")
            if cache_view.next_layer_idx != state.next_layer_idx:
                raise RuntimeError("Engine/cache transaction layer progress diverged")
        for request_id, status in self._statuses.items():
            if status in (self.WAITING, self.REJECTED):
                try:
                    self.cache.request_state(request_id)
                except KeyError:
                    continue
                raise RuntimeError("waiting request unexpectedly exists in PagedKVCache")
            cache_status = self.cache.request_state(request_id)["status"]
            if cache_status != status:
                raise RuntimeError("DecodeEngine and PagedKVCache request status diverged")
        if self._scheduler_managed:
            for request_id in self.active_request_ids():
                spec = self._request_specs[request_id]
                physical_blocks = len(self.cache.request_block_ids(request_id))
                if physical_blocks > spec.commitment_blocks(self.cache.block_size):
                    raise RuntimeError("physical ownership exceeds scheduler commitment")
            if self.cache.num_used_blocks > self._committed_blocks():
                raise RuntimeError("used physical blocks exceed active commitments")
        return True

    def _admit_ids(self, ids):
        for request_id in ids:
            self.cache.add_request(request_id)
            self._statuses[request_id] = self.ACTIVE
        self._sync_cache_version()

    def _bump_state_version(self):
        self._state_version += 1
        self._pending_runnable_ids = None

    def _sync_cache_version(self):
        self._observed_cache_state_version = self.cache.state_version

    def _require_cache_synchronized(self, *, stale=False):
        if not self._scheduler_managed:
            return
        if self.cache.state_version != self._observed_cache_state_version:
            if stale:
                self._stale_decision_count += 1
            raise RuntimeError("PagedKVCache mutated outside scheduler-managed DecodeEngine")

    def _committed_blocks(self):
        if not self._scheduler_managed:
            return 0
        return sum(
            self._request_specs[request_id].commitment_blocks(self.cache.block_size)
            for request_id in self.active_request_ids()
        )

    def _require_no_open_step(self, action):
        if self._open_step_transaction is not None:
            raise RuntimeError(
                f"cannot {action} during an open decode step transaction"
            )

    def _require_open_step(self, transaction):
        if not isinstance(transaction, DecodeStepTransaction):
            raise TypeError("transaction must be a DecodeStepTransaction")
        state = self._open_step_transaction
        if state is None:
            raise RuntimeError("DecodeEngine has no open decode step transaction")
        handle = state.handle
        if (
            transaction.transaction_id != handle.transaction_id
            or transaction.engine_version != handle.engine_version
            or transaction.request_ids != handle.request_ids
        ):
            raise RuntimeError("stale or invalid DecodeEngine transaction handle")
        if state.state != "open":
            raise RuntimeError(f"decode step transaction is already {state.state}")
        return state

    def _validate_scheduler_runnable_ids(self, ids):
        if not self._scheduler_managed:
            return
        if self._pending_runnable_ids is None:
            raise RuntimeError("scheduler-managed step requires an applied decision")
        if ids != self._pending_runnable_ids:
            raise RuntimeError("request_ids must match the applied scheduler runnable_ids")

    def _write_transaction_layer(
        self,
        state,
        layer_idx,
        q,
        k,
        v,
        *,
        rotary_dim,
        base,
    ):
        from .rope import apply_rope

        cache_view = self.cache.transaction_view(state.cache_transaction)
        q_rotated = apply_rope(
            q,
            cache_view.positions,
            rotary_dim=rotary_dim,
            base=base,
        )
        k_rotated = apply_rope(
            k,
            cache_view.positions,
            rotary_dim=rotary_dim,
            base=base,
        )
        cache_view = self.cache.write_token_layer(
            state.cache_transaction,
            layer_idx,
            k_rotated,
            v,
        )
        return q_rotated, cache_view

    def _abort_open_step_state(self, state):
        if self._open_step_transaction is not state or state.state != "open":
            raise RuntimeError("decode step transaction is not open")
        aborted = self.cache.abort_token(state.cache_transaction)
        state.state = "aborted"
        self._open_step_transaction = None
        self._transaction_abort_count += 1
        self._sync_cache_version()
        self._bump_state_version()
        return aborted

    def _normalize_counter_mapping(self, values, request_ids, name):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        unknown = set(values) - set(request_ids)
        if unknown:
            raise ValueError(f"{name} contains unknown request ids")
        result = {}
        for request_id in request_ids:
            value = values.get(request_id, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} values must be non-negative integers")
            result[request_id] = value
        return result

    def _validate_decision_ids(self, name, values):
        try:
            ids = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be iterable") from exc
        for request_id in ids:
            self._check_request_id(request_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} must contain unique request ids")
        return ids

    def _decode(self, q, block_tables, seq_lens, layer_idx=0):
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.cache.num_layers:
            raise ValueError("layer_idx must be in [0, num_layers)")
        if self.decode_backend == "reference":
            from .paged_reference import paged_decode_attention_ref

            return paged_decode_attention_ref(
                q,
                self.cache.k_cache[layer_idx],
                self.cache.v_cache[layer_idx],
                block_tables,
                seq_lens,
                sm_scale=self.sm_scale,
            )

        from .kernels.paged_decode import paged_decode_attention

        return paged_decode_attention(
            q,
            self.cache.k_cache[layer_idx],
            self.cache.v_cache[layer_idx],
            block_tables,
            seq_lens,
            sm_scale=self.sm_scale,
            block_size=self.cache.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

    def _validate_step_inputs(self, ids, q, k, v):
        torch = _torch()
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("q, k, and v must be torch tensors")
        if q.dim() != 3:
            raise ValueError("q must have shape [num_requests, num_q_heads, head_dim]")
        if k.dim() == 2 and len(ids) == 1:
            k = k.unsqueeze(0)
        if v.dim() == 2 and len(ids) == 1:
            v = v.unsqueeze(0)
        expected_kv_shape = (len(ids), self.cache.num_kv_heads, self.cache.head_dim)
        if q.shape[0] != len(ids) or q.shape[1] <= 0 or q.shape[2] != self.cache.head_dim:
            raise ValueError("q shape must match active request count and cache head_dim")
        if k.shape != expected_kv_shape or v.shape != expected_kv_shape:
            raise ValueError("k and v shapes must match active request count and cache dimensions")
        if q.device != self.cache.device or k.device != self.cache.device or v.device != self.cache.device:
            raise ValueError("q, k, and v must be on the cache device")
        if q.dtype != self.cache.dtype or k.dtype != self.cache.dtype or v.dtype != self.cache.dtype:
            raise ValueError("q, k, and v must use the cache dtype")

    def _needed_new_blocks(self, ids):
        return sum(
            self.cache.request_state(request_id)["seq_len"] % self.cache.block_size == 0
            for request_id in ids
        )

    def _normalize_waiting_ids(self, request_ids):
        if request_ids is None:
            ids = tuple(
                request_id
                for request_id, status in self._statuses.items()
                if status == self.WAITING
            )
        else:
            ids = self._normalize_ids(request_ids)
        if not ids:
            raise ValueError("at least one waiting request is required for admission")
        for request_id in ids:
            self._require_status(request_id, self.WAITING)
        return ids

    def _normalize_active_ids(self, request_ids):
        if request_ids is None:
            ids = self.active_request_ids()
        else:
            ids = self._normalize_ids(request_ids)
        if not ids:
            raise ValueError("at least one active request is required for a decode step")
        for request_id in ids:
            self._require_status(request_id, self.ACTIVE)
        return ids

    def _normalize_ids(self, request_ids):
        if isinstance(request_ids, (str, bytes)):
            ids = (request_ids,)
        else:
            try:
                ids = tuple(request_ids)
            except TypeError:
                ids = (request_ids,)
        for request_id in ids:
            self._check_request_id(request_id)
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique")
        return ids

    @staticmethod
    def _check_request_id(request_id):
        try:
            hash(request_id)
        except TypeError as exc:
            raise ValueError("request ids must be hashable") from exc

    def _require_known_request(self, request_id):
        self._check_request_id(request_id)
        if request_id not in self._statuses:
            raise KeyError(f"unknown request_id: {request_id!r}")
        return self._statuses[request_id]

    def _require_status(self, request_id, expected_status):
        status = self._require_known_request(request_id)
        if status != expected_status:
            raise RuntimeError(
                f"request_id {request_id!r} is {status}, expected {expected_status}"
            )
