"""Single-layer dynamic decode execution engine for the FlashDec runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable


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


class DecodeEngine:
    """Coordinate request lifecycle, RoPE/KV append, and paged decode.

    The engine intentionally models one layer and one decode token per active
    request on each call to :meth:`step`. Model projection, sampling, prefill,
    and network serving remain outside this project boundary.

    ``append_backend`` selects ``torch``, ``cuda``, or ``fused_cuda`` from the
    RoPE/KV data path. ``decode_backend='reference'`` keeps CPU and semantic
    tests available; ``'triton'`` uses the frozen paged decode kernel on CUDA.
    """

    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    CANCELLED = "cancelled"

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
    ):
        from .cache import PagedKVCache

        if not isinstance(cache, PagedKVCache):
            raise TypeError("cache must be a PagedKVCache")
        if append_backend not in self._APPEND_BACKENDS:
            raise ValueError("append_backend must be 'torch', 'cuda', or 'fused_cuda'")
        if decode_backend not in self._DECODE_BACKENDS:
            raise ValueError("decode_backend must be 'reference' or 'triton'")

        self.cache = cache
        self.append_backend = append_backend
        self.decode_backend = decode_backend
        self.sm_scale = sm_scale
        self.num_warps = num_warps
        self.num_stages = num_stages
        self._statuses: dict[Hashable, str] = {}
        self._completed_step_count = 0
        self._appended_token_count = 0
        self._backpressure_count = 0

    def add_request(self, request_id):
        """Submit a new request in ``waiting`` state without allocating KV memory."""
        self._check_request_id(request_id)
        if request_id in self._statuses:
            raise RuntimeError(f"request_id {request_id!r} already exists in the DecodeEngine")
        self._statuses[request_id] = self.WAITING
        return AdmissionResult(request_id=request_id, status=self.WAITING)

    def admit(self, request_ids=None):
        """Move waiting requests to active state without reserving physical blocks."""
        ids = self._normalize_waiting_ids(request_ids)
        for request_id in ids:
            self.cache.add_request(request_id)
            self._statuses[request_id] = self.ACTIVE
        return tuple(AdmissionResult(request_id=request_id, status=self.ACTIVE) for request_id in ids)

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
        if status != self.WAITING:
            result["cache"] = self.cache.request_state(request_id)
        return result

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
        ids = self._normalize_active_ids(request_ids)
        self._validate_step_inputs(ids, q, k, v)
        needed_new_blocks = self._needed_new_blocks(ids)
        if needed_new_blocks > self.cache.num_free_blocks:
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

        from .rope import rope_paged_kv_append

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
        output = self._decode(append_result.q, append_result.block_tables, append_result.seq_lens)
        self._completed_step_count += 1
        self._appended_token_count += len(ids)
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
        self._require_status(request_id, self.ACTIVE)
        released = self.cache.finish_request(request_id)
        self._statuses[request_id] = self.FINISHED
        return released

    def cancel_request(self, request_id):
        """Cancel an active request and release its physical cache blocks."""
        self._require_status(request_id, self.ACTIVE)
        released = self.cache.cancel_request(request_id)
        self._statuses[request_id] = self.CANCELLED
        return released

    def metrics(self):
        """Return engine counters together with the underlying cache metrics."""
        return {
            "waiting_requests": sum(status == self.WAITING for status in self._statuses.values()),
            "active_requests": sum(status == self.ACTIVE for status in self._statuses.values()),
            "finished_requests": sum(status == self.FINISHED for status in self._statuses.values()),
            "cancelled_requests": sum(status == self.CANCELLED for status in self._statuses.values()),
            "completed_step_count": self._completed_step_count,
            "appended_token_count": self._appended_token_count,
            "backpressure_count": self._backpressure_count,
            "append_backend": self.append_backend,
            "decode_backend": self.decode_backend,
            "cache": self.cache.metrics(),
        }

    def validate_invariants(self):
        """Check that engine lifecycle state agrees with cache ownership state."""
        self.cache.validate_invariants()
        for request_id, status in self._statuses.items():
            if status == self.WAITING:
                try:
                    self.cache.request_state(request_id)
                except KeyError:
                    continue
                raise RuntimeError("waiting request unexpectedly exists in PagedKVCache")
            cache_status = self.cache.request_state(request_id)["status"]
            if cache_status != status:
                raise RuntimeError("DecodeEngine and PagedKVCache request status diverged")
        return True

    def _decode(self, q, block_tables, seq_lens):
        if self.decode_backend == "reference":
            from .paged_reference import paged_decode_attention_ref

            return paged_decode_attention_ref(
                q,
                self.cache.k_cache[0],
                self.cache.v_cache[0],
                block_tables,
                seq_lens,
                sm_scale=self.sm_scale,
            )

        from .kernels.paged_decode import paged_decode_attention

        return paged_decode_attention(
            q,
            self.cache.k_cache[0],
            self.cache.v_cache[0],
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
