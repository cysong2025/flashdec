"""Paged KV cache runtime used by the paged attention reference."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
from typing import Hashable


@dataclass
class _RequestState:
    status: str = "active"
    seq_len: int = 0
    block_ids: list[int] = field(default_factory=list)
    transaction_id: int | None = None
    prefix_id: Hashable | None = None
    shared_block_count: int = 0


@dataclass
class _PrefixState:
    block_ids: tuple[int, ...]
    active_refcount: int = 0


@dataclass(frozen=True)
class KVTokenTransactionView:
    """Detached public snapshot of one multi-layer token transaction."""

    transaction_id: int
    cache_version: int
    request_ids: tuple[Hashable, ...]
    positions: object
    physical_block_ids: object
    block_offsets: object
    block_tables: object
    effective_seq_lens: object
    next_layer_idx: int
    state: str


@dataclass
class _KVTokenTransactionState:
    transaction_id: int
    cache_version: int
    request_ids: tuple[Hashable, ...]
    positions: tuple[int, ...]
    locations: tuple[tuple[int, int], ...]
    block_tables: object
    newly_allocated_by_request: dict[Hashable, int]
    allocation_order: tuple[int, ...]
    next_layer_idx: int = 0
    written_layers: set[int] = field(default_factory=set)
    state: str = "open"


def _torch():
    import torch

    return torch


class PagedKVCache:
    """Fixed-block KV cache runtime for single-token decode.

    The physical cache layout is:

    [num_layers, max_blocks, num_kv_heads, block_size, head_dim]

    Each request owns a logical list of physical block ids. Appending one token
    writes into the current tail block or allocates a new physical block when
    the logical position crosses a block boundary. Requests may also reference
    immutable full blocks held by the optional shared-prefix registry; their
    mutable tail remains private. Finished or cancelled requests release their
    private blocks back to a deterministic free list that prioritizes reuse.
    """

    ACTIVE = "active"
    FINISHED = "finished"
    CANCELLED = "cancelled"

    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks,
        dtype=None,
        device=None,
        prefix_cache_capacity_blocks=0,
    ):
        torch = _torch()
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        for name, value in [
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("block_size", block_size),
            ("max_blocks", max_blocks),
        ]:
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be a floating point torch dtype")

        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.prefix_cache_capacity_blocks = int(prefix_cache_capacity_blocks)
        if self.prefix_cache_capacity_blocks < 0:
            raise ValueError("prefix_cache_capacity_blocks must be non-negative")
        if self.prefix_cache_capacity_blocks > self.max_blocks:
            raise ValueError("prefix_cache_capacity_blocks must not exceed max_blocks")
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (
            self.num_layers,
            self.max_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )
        self.k_cache = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.device = self.k_cache.device
        self.v_cache = torch.zeros_like(self.k_cache)
        self._free_blocks = deque(range(self.max_blocks))
        self._requests: dict[Hashable, _RequestState] = {}
        self._prefixes: OrderedDict[Hashable, _PrefixState] = OrderedDict()
        self._ever_allocated_blocks: set[int] = set()
        self._allocation_count = 0
        self._fresh_allocation_count = 0
        self._free_count = 0
        self._reuse_count = 0
        self._capacity_failure_count = 0
        self._state_version = 0
        self._next_transaction_id = 1
        self._open_transaction_id: int | None = None
        self._transactions: dict[int, _KVTokenTransactionState] = {}
        self._transaction_begin_count = 0
        self._transaction_commit_count = 0
        self._transaction_abort_count = 0
        self._transaction_layer_write_count = 0
        self._transaction_rollback_block_count = 0
        self._transaction_failure_count = 0
        self._prefix_registration_count = 0
        self._prefix_hit_count = 0
        self._prefix_miss_count = 0
        self._prefix_eviction_count = 0
        self._prefix_capacity_failure_count = 0

    def add_request(self, request_id):
        """Register an empty request if it does not already exist."""
        if self._open_transaction_id is not None:
            raise RuntimeError("cannot add a request during an open token transaction")
        self._check_request_id(request_id)
        state = self._requests.get(request_id)
        if state is None:
            self._requests[request_id] = _RequestState()
            self._state_version += 1
        elif state.status != self.ACTIVE:
            raise RuntimeError(
                f"request_id {request_id!r} is {state.status} and cannot be reactivated"
            )
        return self

    def register_prefix(self, prefix_id, k_blocks, v_blocks):
        """Copy immutable full K/V blocks into the shared-prefix registry.

        ``k_blocks`` and ``v_blocks`` use shape
        ``[num_layers, num_prefix_blocks, num_kv_heads, block_size, head_dim]``.
        Inactive least-recently-used prefixes are evicted when the configured
        prefix capacity or the shared physical pool requires space.
        """
        if self._open_transaction_id is not None:
            raise RuntimeError("cannot register a prefix during an open token transaction")
        self._check_prefix_id(prefix_id)
        if prefix_id in self._prefixes:
            raise RuntimeError(f"prefix_id {prefix_id!r} is already registered")
        k_blocks, v_blocks = self._prepare_prefix_blocks(k_blocks, v_blocks)
        num_blocks = int(k_blocks.shape[1])
        eviction_ids = self._prefix_eviction_plan(num_blocks)
        if eviction_ids is None:
            self._capacity_failure_count += 1
            self._prefix_capacity_failure_count += 1
            raise RuntimeError("PagedKVCache has insufficient evictable prefix capacity")

        for evicted_id in eviction_ids:
            self._evict_prefix_entry(evicted_id)

        block_ids = tuple(self._allocate_block() for _ in range(num_blocks))
        block_index = _torch().tensor(block_ids, device=self.device, dtype=_torch().long)
        self.k_cache.index_copy_(1, block_index, k_blocks)
        self.v_cache.index_copy_(1, block_index, v_blocks)
        self._prefixes[prefix_id] = _PrefixState(block_ids=block_ids)
        self._prefix_registration_count += 1
        self._state_version += 1
        return self.prefix_state(prefix_id)

    def attach_prefix(self, request_id, prefix_id):
        """Attach one resident immutable prefix to an empty active request."""
        if self._open_transaction_id is not None:
            raise RuntimeError("cannot attach a prefix during an open token transaction")
        self._check_prefix_id(prefix_id)
        state = self._require_active_request(request_id)
        if state.transaction_id is not None:
            raise RuntimeError("cannot attach a prefix to a request in a token transaction")
        if state.seq_len or state.block_ids or state.prefix_id is not None:
            raise RuntimeError("shared prefixes can only attach to an empty active request")
        prefix = self._prefixes.get(prefix_id)
        if prefix is None:
            self._prefix_miss_count += 1
            raise KeyError(f"unknown prefix_id: {prefix_id!r}")

        state.block_ids.extend(prefix.block_ids)
        state.seq_len = len(prefix.block_ids) * self.block_size
        state.prefix_id = prefix_id
        state.shared_block_count = len(prefix.block_ids)
        prefix.active_refcount += 1
        self._prefixes.move_to_end(prefix_id)
        self._prefix_hit_count += 1
        self._state_version += 1
        return tuple(prefix.block_ids)

    def evict_prefix(self, prefix_id):
        """Evict one inactive prefix and return its blocks to the free list."""
        if self._open_transaction_id is not None:
            raise RuntimeError("cannot evict a prefix during an open token transaction")
        self._check_prefix_id(prefix_id)
        prefix = self._prefixes.get(prefix_id)
        if prefix is None:
            raise KeyError(f"unknown prefix_id: {prefix_id!r}")
        if prefix.active_refcount:
            raise RuntimeError("cannot evict a prefix with active request references")
        released = self._evict_prefix_entry(prefix_id)
        self._state_version += 1
        return released

    def prefix_state(self, prefix_id):
        """Return a detached snapshot of one resident shared prefix."""
        self._check_prefix_id(prefix_id)
        prefix = self._prefixes.get(prefix_id)
        if prefix is None:
            raise KeyError(f"unknown prefix_id: {prefix_id!r}")
        return {
            "prefix_id": prefix_id,
            "block_ids": tuple(prefix.block_ids),
            "num_blocks": len(prefix.block_ids),
            "token_count": len(prefix.block_ids) * self.block_size,
            "active_refcount": prefix.active_refcount,
        }

    def append(self, layer_idx, request_ids, k, v):
        """Append one token of K/V for each request.

        Args:
            layer_idx: Layer to write. Legacy append is restricted to a
                single-layer cache; multi-layer caches use token transactions.
            request_ids: Request ids in the same order as the first dimension of
                k and v.
            k/v: [num_requests, num_kv_heads, head_dim]. For one request,
                [num_kv_heads, head_dim] is also accepted.

        Returns:
            A padded block table tensor for the provided request ids.
        """
        self._require_legacy_append_available()
        layer_idx, ids, k, v = self._prepare_append_inputs(layer_idx, request_ids, k, v)
        self._preflight_append(ids)
        locations = self._allocate_append_locations(ids)
        for row, (physical_block, block_offset) in enumerate(locations):
            self.k_cache[layer_idx, physical_block, :, block_offset, :] = k[row]
            self.v_cache[layer_idx, physical_block, :, block_offset, :] = v[row]
        self._advance_request_lengths(ids)
        self._state_version += 1
        return self.block_tables(ids)

    def append_cuda(self, layer_idx, request_ids, k, v):
        """Append one token per request through the native CUDA K/V write path.

        Python retains ownership of request lifecycle, capacity preflight, and
        physical block allocation. The JIT-built CUDA extension receives only
        validated physical block ids/offsets and copies K/V values into the
        token-major cache. This keeps allocator semantics identical to
        :meth:`append` while isolating the native data-movement primitive.

        This first native path supports CUDA cache tensors with FP16, BF16, or
        FP32 dtype. It intentionally does not apply RoPE; callers can use the
        PyTorch RoPE reference before appending, or ``append_fused_cuda`` for
        the separate fused data path.
        """
        self._require_legacy_append_available()
        torch = _torch()
        layer_idx, ids, k, v = self._prepare_append_inputs(layer_idx, request_ids, k, v)
        if self.device.type != "cuda":
            raise ValueError("append_cuda requires a CUDA-resident PagedKVCache")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("append_cuda supports cache dtype float16, bfloat16, or float32")
        if not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("append_cuda requires contiguous k and v tensors")

        # Build before mutating allocator state, so toolchain/build failures do
        # not create requests or consume physical blocks.
        from ._cuda_kv_append import cuda_kv_append, load_cuda_kv_append_extension

        load_cuda_kv_append_extension()
        self._preflight_append(ids)
        locations = self._allocate_append_locations(ids)
        block_ids = torch.tensor(
            [physical_block for physical_block, _ in locations],
            device=self.device,
            dtype=torch.int64,
        )
        block_offsets = torch.tensor(
            [block_offset for _, block_offset in locations],
            device=self.device,
            dtype=torch.int64,
        )
        cuda_kv_append(
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            block_ids,
            block_offsets,
            k,
            v,
        )
        self._advance_request_lengths(ids)
        self._state_version += 1
        return self.block_tables(ids)

    def append_fused_cuda(
        self,
        layer_idx,
        request_ids,
        q,
        k,
        v,
        positions,
        rotary_dim=None,
        base=10_000.0,
    ):
        """Fuse split-half RoPE with K/V append while preserving allocator semantics.

        This method is intentionally internal-runtime facing: callers provide
        pre-append ``positions`` from :meth:`next_positions`, while the cache
        retains capacity preflight, physical block allocation, and seq_len
        mutation. It returns ``(rotated_q, block_tables)``.
        """
        self._require_legacy_append_available()
        torch = _torch()
        layer_idx, ids, k, v = self._prepare_append_inputs(layer_idx, request_ids, k, v)
        if self.device.type != "cuda":
            raise ValueError("append_fused_cuda requires a CUDA-resident PagedKVCache")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("append_fused_cuda supports cache dtype float16, bfloat16, or float32")
        if q.dim() != 3 or q.shape[0] != len(ids) or q.shape[1] <= 0 or q.shape[2] != self.head_dim:
            raise ValueError("q shape must match request count and cache head_dim")
        if q.device != self.device or q.dtype != self.dtype:
            raise ValueError("q must use the cache device and dtype")
        if positions.dim() != 1 or positions.numel() != len(ids):
            raise ValueError("positions must have shape [num_requests]")
        if positions.device != self.device or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32/int64 on the cache device")
        if bool(torch.any(positions < 0).item()):
            raise ValueError("positions must be non-negative")
        if rotary_dim is None:
            rotary_dim = self.head_dim
        if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int):
            raise ValueError("rotary_dim must be an even integer")
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
        base = float(base)
        if not math.isfinite(base) or base <= 0.0:
            raise ValueError("base must be a positive finite number")
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("append_fused_cuda requires contiguous q, k, and v tensors")

        # Keep build/configuration errors before any allocator mutation.
        from ._fused_rope_kv_append import (
            fused_rope_kv_append,
            load_fused_rope_kv_append_extension,
        )

        load_fused_rope_kv_append_extension()
        self._preflight_append(ids)
        locations = self._allocate_append_locations(ids)
        block_ids = torch.tensor(
            [physical_block for physical_block, _ in locations],
            device=self.device,
            dtype=torch.int64,
        )
        block_offsets = torch.tensor(
            [block_offset for _, block_offset in locations],
            device=self.device,
            dtype=torch.int64,
        )
        q_rotated = fused_rope_kv_append(
            q,
            k,
            v,
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            block_ids,
            block_offsets,
            positions,
            rotary_dim=rotary_dim,
            base=base,
        )
        self._advance_request_lengths(ids)
        self._state_version += 1
        return q_rotated, self.block_tables(ids)

    def begin_token(self, request_ids):
        """Reserve one shared token location for every layer in the batch.

        Requests must already be active.  The committed sequence lengths stay
        unchanged until :meth:`commit_token`; boundary blocks allocated here
        are returned by :meth:`abort_token`.
        """
        if self._open_transaction_id is not None:
            raise RuntimeError("PagedKVCache already has an open token transaction")
        ids = tuple(self._normalize_request_ids(request_ids))
        if not ids:
            raise ValueError("request_ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique within one token transaction")
        states = [self._require_active_request(request_id) for request_id in ids]
        if any(state.transaction_id is not None for state in states):
            raise RuntimeError("request already belongs to an open token transaction")

        needed_new_blocks = sum(
            state.seq_len % self.block_size == 0 for state in states
        )
        if needed_new_blocks > len(self._free_blocks):
            self._capacity_failure_count += 1
            self._transaction_failure_count += 1
            raise RuntimeError("PagedKVCache is out of physical blocks")

        transaction_id = self._next_transaction_id
        positions = tuple(state.seq_len for state in states)
        locations = []
        newly_allocated = {}
        allocation_order = []
        try:
            for request_id, state, position in zip(ids, states, positions):
                logical_block = position // self.block_size
                block_offset = position % self.block_size
                if block_offset == 0:
                    block_id = self._allocate_block()
                    state.block_ids.append(block_id)
                    newly_allocated[request_id] = block_id
                    allocation_order.append(block_id)
                locations.append((state.block_ids[logical_block], block_offset))
            block_tables = self.block_tables(ids)
            for state in states:
                state.transaction_id = transaction_id
        except Exception:
            for request_id, block_id in newly_allocated.items():
                state = self._requests[request_id]
                if state.block_ids and state.block_ids[-1] == block_id:
                    state.block_ids.pop()
                state.transaction_id = None
            for block_id in reversed(allocation_order):
                self._free_blocks.appendleft(block_id)
            self._transaction_failure_count += 1
            raise

        cache_version = self._state_version + 1
        transaction = _KVTokenTransactionState(
            transaction_id=transaction_id,
            cache_version=cache_version,
            request_ids=ids,
            positions=positions,
            locations=tuple(locations),
            block_tables=block_tables,
            newly_allocated_by_request=newly_allocated,
            allocation_order=tuple(allocation_order),
        )
        self._transactions[transaction_id] = transaction
        self._open_transaction_id = transaction_id
        self._next_transaction_id += 1
        self._transaction_begin_count += 1
        self._state_version = cache_version
        return self._transaction_view(transaction)

    def write_token_layer(self, transaction, layer_idx, k, v):
        """Write one layer at the locations reserved by ``begin_token``."""
        state = self._require_open_transaction(transaction)
        try:
            layer_idx, k, v = self._prepare_transaction_layer_write(
                state,
                layer_idx,
                k,
                v,
            )

            for row, (physical_block, block_offset) in enumerate(state.locations):
                self.k_cache[layer_idx, physical_block, :, block_offset, :] = k[row]
                self.v_cache[layer_idx, physical_block, :, block_offset, :] = v[row]
        except Exception:
            self._transaction_failure_count += 1
            raise
        return self._record_transaction_layer_write(state, layer_idx)

    def write_token_layer_fused_cuda(
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
        """Run fused RoPE + K/V write at transaction-reserved locations.

        The transaction remains the sole owner of allocation, rollback, and
        committed sequence-length mutation.  The CUDA primitive receives only
        the already reserved physical block ids/offsets and writes one layer;
        it does not allocate blocks or publish the token.
        """
        state = self._require_open_transaction(transaction)
        try:
            torch = _torch()
            layer_idx, k, v = self._prepare_transaction_layer_write(
                state,
                layer_idx,
                k,
                v,
            )
            if self.device.type != "cuda":
                raise ValueError(
                    "write_token_layer_fused_cuda requires a CUDA-resident PagedKVCache"
                )
            if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "write_token_layer_fused_cuda supports float16, bfloat16, or float32"
                )
            if q.dim() != 3 or q.shape[0] != len(state.request_ids):
                raise ValueError(
                    "q shape must match transaction request count and cache head_dim"
                )
            if q.shape[1] <= 0 or q.shape[2] != self.head_dim:
                raise ValueError(
                    "q shape must match transaction request count and cache head_dim"
                )
            if q.device != self.device or q.dtype != self.dtype:
                raise ValueError("q must use the cache device and dtype")
            if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
                raise ValueError(
                    "write_token_layer_fused_cuda requires contiguous q, k, and v tensors"
                )

            from ._fused_rope_kv_append import fused_rope_kv_append

            view = self._transaction_view(state)
            q_rotated = fused_rope_kv_append(
                q,
                k,
                v,
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                view.physical_block_ids,
                view.block_offsets,
                view.positions,
                rotary_dim=rotary_dim,
                base=base,
            )
        except Exception:
            self._transaction_failure_count += 1
            raise
        return q_rotated, self._record_transaction_layer_write(state, layer_idx)

    def commit_token(self, transaction):
        """Publish one token after every layer has been written successfully."""
        state = self._require_open_transaction(transaction)
        if state.next_layer_idx != self.num_layers:
            raise RuntimeError("cannot commit token before all layers are written")
        for request_id in state.request_ids:
            request = self._requests[request_id]
            request.seq_len += 1
            request.transaction_id = None
        state.state = "committed"
        self._open_transaction_id = None
        self._transaction_commit_count += 1
        self._state_version += 1
        return self._transaction_view(state)

    def abort_token(self, transaction):
        """Abort an open token and return any boundary blocks it reserved."""
        state = self._require_open_transaction(transaction)
        for request_id, block_id in state.newly_allocated_by_request.items():
            request = self._requests[request_id]
            if not request.block_ids or request.block_ids[-1] != block_id:
                raise RuntimeError("transaction rollback block ownership is inconsistent")
            request.block_ids.pop()
        for block_id in reversed(state.allocation_order):
            self._free_blocks.appendleft(block_id)
        for request_id in state.request_ids:
            self._requests[request_id].transaction_id = None
        state.state = "aborted"
        self._open_transaction_id = None
        self._transaction_abort_count += 1
        self._transaction_rollback_block_count += len(state.allocation_order)
        self._state_version += 1
        return self._transaction_view(state)

    def transaction_view(self, transaction):
        """Return the latest detached snapshot for a transaction handle."""
        return self._transaction_view(self._require_transaction(transaction))

    def finish_request(self, request_id):
        """Mark an active request finished and release all owned blocks."""
        return self._close_request(request_id, self.FINISHED)

    def cancel_request(self, request_id):
        """Mark an active request cancelled and release all owned blocks."""
        return self._close_request(request_id, self.CANCELLED)

    def request_state(self, request_id):
        """Return a detached snapshot of one request's lifecycle state."""
        state = self._require_request(request_id)
        return {
            "request_id": request_id,
            "status": state.status,
            "seq_len": state.seq_len,
            "block_ids": tuple(state.block_ids),
            "prefix_id": state.prefix_id,
            "shared_block_count": state.shared_block_count,
        }

    def next_positions(self, request_ids, device=None):
        """Return each request's pre-append token position without mutation."""
        torch = _torch()
        ids = self._normalize_request_ids(request_ids)
        if not ids:
            raise ValueError("request_ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique")

        positions = []
        for request_id in ids:
            state = self._requests.get(request_id)
            if state is None:
                positions.append(0)
            elif state.status != self.ACTIVE:
                raise RuntimeError(
                    f"request_id {request_id!r} is {state.status} and cannot accept append"
                )
            elif state.transaction_id is not None:
                raise RuntimeError(
                    f"request_id {request_id!r} belongs to an open token transaction"
                )
            else:
                positions.append(state.seq_len)
        if device is None:
            device = self.device
        return torch.tensor(positions, device=device, dtype=torch.int64)

    def metrics(self):
        """Return block-pool, fragmentation, lifecycle, and reuse counters."""
        states = list(self._requests.values())
        active_states = [state for state in states if state.status == self.ACTIVE]
        used_blocks = self.num_used_blocks
        resident_prefix_blocks = self._resident_prefix_blocks()
        active_prefix_references = sum(
            prefix.active_refcount for prefix in self._prefixes.values()
        )
        shared_prefix_blocks = sum(
            len(prefix.block_ids)
            for prefix in self._prefixes.values()
            if prefix.active_refcount >= 2
        )
        saved_prefix_blocks = sum(
            max(0, prefix.active_refcount - 1) * len(prefix.block_ids)
            for prefix in self._prefixes.values()
        )
        bytes_per_block = (
            self.num_layers
            * 2
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
            * self.k_cache.element_size()
        )
        open_transaction = (
            self._transactions.get(self._open_transaction_id)
            if self._open_transaction_id is not None
            else None
        )
        reserved_transaction_blocks = (
            len(open_transaction.allocation_order) if open_transaction is not None else 0
        )
        reserved_tokens = used_blocks * self.block_size
        active_tokens = sum(state.seq_len for state in active_states)
        physical_data_tokens = resident_prefix_blocks * self.block_size + sum(
            max(0, state.seq_len - state.shared_block_count * self.block_size)
            for state in active_states
        )
        fragmentation_tokens = reserved_tokens - physical_data_tokens
        prefix_lookups = self._prefix_hit_count + self._prefix_miss_count
        return {
            "max_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.num_free_blocks,
            "block_utilization": used_blocks / self.max_blocks,
            "active_tokens": active_tokens,
            "physical_data_tokens": physical_data_tokens,
            "reserved_tokens": reserved_tokens,
            "internal_fragmentation_tokens": fragmentation_tokens,
            "internal_fragmentation_ratio": (
                fragmentation_tokens / reserved_tokens if reserved_tokens else 0.0
            ),
            "allocation_count": self._allocation_count,
            "fresh_allocation_count": self._fresh_allocation_count,
            "free_count": self._free_count,
            "reuse_count": self._reuse_count,
            "capacity_failure_count": self._capacity_failure_count,
            "active_requests": sum(state.status == self.ACTIVE for state in states),
            "finished_requests": sum(state.status == self.FINISHED for state in states),
            "cancelled_requests": sum(state.status == self.CANCELLED for state in states),
            "transaction_begin_count": self._transaction_begin_count,
            "transaction_commit_count": self._transaction_commit_count,
            "transaction_abort_count": self._transaction_abort_count,
            "open_transaction_count": int(open_transaction is not None),
            "pending_request_count": (
                len(open_transaction.request_ids) if open_transaction is not None else 0
            ),
            "reserved_transaction_blocks": reserved_transaction_blocks,
            "transaction_layer_write_count": self._transaction_layer_write_count,
            "transaction_rollback_block_count": self._transaction_rollback_block_count,
            "transaction_failure_count": self._transaction_failure_count,
            "bytes_per_block": bytes_per_block,
            "allocated_kv_bytes": used_blocks * bytes_per_block,
            "reserved_transaction_bytes": reserved_transaction_blocks * bytes_per_block,
            "prefix_cache_capacity_blocks": self.prefix_cache_capacity_blocks,
            "resident_prefix_count": len(self._prefixes),
            "resident_prefix_blocks": resident_prefix_blocks,
            "active_prefix_references": active_prefix_references,
            "shared_prefix_blocks": shared_prefix_blocks,
            "saved_prefix_blocks": saved_prefix_blocks,
            "saved_prefix_bytes": saved_prefix_blocks * bytes_per_block,
            "prefix_registration_count": self._prefix_registration_count,
            "prefix_hit_count": self._prefix_hit_count,
            "prefix_miss_count": self._prefix_miss_count,
            "prefix_hit_ratio": (
                self._prefix_hit_count / prefix_lookups if prefix_lookups else 0.0
            ),
            "prefix_eviction_count": self._prefix_eviction_count,
            "prefix_capacity_failure_count": self._prefix_capacity_failure_count,
        }

    def validate_invariants(self):
        """Raise RuntimeError if request ownership and the free list diverge."""
        free_blocks = list(self._free_blocks)
        if len(free_blocks) != len(set(free_blocks)):
            raise RuntimeError("PagedKVCache free list contains duplicate blocks")

        prefix_blocks = []
        expected_prefix_refcounts = {prefix_id: 0 for prefix_id in self._prefixes}
        for prefix in self._prefixes.values():
            if not prefix.block_ids:
                raise RuntimeError("resident prefix must own at least one block")
            if prefix.active_refcount < 0:
                raise RuntimeError("prefix active_refcount must be non-negative")
            prefix_blocks.extend(prefix.block_ids)
        if len(prefix_blocks) != len(set(prefix_blocks)):
            raise RuntimeError("a physical block belongs to multiple prefixes")
        if len(prefix_blocks) > self.prefix_cache_capacity_blocks:
            raise RuntimeError("resident prefixes exceed configured prefix capacity")

        private_blocks = []
        for state in self._requests.values():
            if state.status == self.ACTIVE:
                pending_tokens = int(state.transaction_id is not None)
                expected_blocks = (
                    state.seq_len + pending_tokens + self.block_size - 1
                ) // self.block_size
                if len(state.block_ids) != expected_blocks:
                    raise RuntimeError("active request block count does not match seq_len")
                if state.prefix_id is None:
                    if state.shared_block_count:
                        raise RuntimeError("request without prefix has shared blocks")
                else:
                    prefix = self._prefixes.get(state.prefix_id)
                    if prefix is None:
                        raise RuntimeError("request references a non-resident prefix")
                    if state.shared_block_count != len(prefix.block_ids):
                        raise RuntimeError("request shared block count does not match prefix")
                    if tuple(state.block_ids[: state.shared_block_count]) != prefix.block_ids:
                        raise RuntimeError("request shared block ids do not match prefix")
                    if state.seq_len < state.shared_block_count * self.block_size:
                        raise RuntimeError("request seq_len is shorter than its shared prefix")
                    expected_prefix_refcounts[state.prefix_id] += 1
                private_blocks.extend(state.block_ids[state.shared_block_count :])
            else:
                if state.block_ids:
                    raise RuntimeError("terminal request still owns physical blocks")
                if state.transaction_id is not None:
                    raise RuntimeError("terminal request belongs to a token transaction")
                if state.prefix_id is not None or state.shared_block_count:
                    raise RuntimeError("terminal request still references a shared prefix")

        for prefix_id, prefix in self._prefixes.items():
            if prefix.active_refcount != expected_prefix_refcounts[prefix_id]:
                raise RuntimeError("prefix active_refcount does not match attached requests")
        if len(private_blocks) != len(set(private_blocks)):
            raise RuntimeError("a private physical block is owned by multiple requests")
        if set(private_blocks) & set(prefix_blocks):
            raise RuntimeError("a physical block is both private and prefix-owned")
        owned_blocks = prefix_blocks + private_blocks
        if set(owned_blocks) & set(free_blocks):
            raise RuntimeError("a physical block is both owned and free")
        if set(owned_blocks) | set(free_blocks) != set(range(self.max_blocks)):
            raise RuntimeError("physical block accounting does not cover the cache")

        open_states = [
            transaction
            for transaction in self._transactions.values()
            if transaction.state == "open"
        ]
        if self._open_transaction_id is None:
            if open_states:
                raise RuntimeError("open transaction is missing from cache state")
            if any(state.transaction_id is not None for state in self._requests.values()):
                raise RuntimeError("request has an in-flight marker without an open transaction")
        else:
            if len(open_states) != 1:
                raise RuntimeError("PagedKVCache must have exactly one open transaction")
            transaction = open_states[0]
            if transaction.transaction_id != self._open_transaction_id:
                raise RuntimeError("open transaction id does not match cache state")
            if transaction.written_layers != set(range(transaction.next_layer_idx)):
                raise RuntimeError("transaction written layers are not sequential")
            if not 0 <= transaction.next_layer_idx <= self.num_layers:
                raise RuntimeError("transaction next layer is out of range")
            for request_id, position, location in zip(
                transaction.request_ids,
                transaction.positions,
                transaction.locations,
            ):
                request = self._require_active_request(request_id)
                if request.transaction_id != transaction.transaction_id:
                    raise RuntimeError("request transaction marker does not match")
                if position != request.seq_len:
                    raise RuntimeError("transaction position is not the committed seq_len")
                logical_block = position // self.block_size
                expected_location = (
                    request.block_ids[logical_block],
                    position % self.block_size,
                )
                if location != expected_location:
                    raise RuntimeError("transaction physical location does not match ownership")
        return True

    def block_tables(self, request_ids=None, max_blocks_per_seq=None, pad_value=-1):
        """Return padded physical block ids for each request.

        Shape: [num_requests, max_blocks_per_seq].
        """
        torch = _torch()
        ids = self._normalize_active_request_ids(request_ids)
        if max_blocks_per_seq is None:
            max_blocks_per_seq = max(1, max((len(self._requests[rid].block_ids) for rid in ids), default=0))
        max_blocks_per_seq = int(max_blocks_per_seq)
        if max_blocks_per_seq <= 0:
            raise ValueError("max_blocks_per_seq must be positive")

        table = torch.full(
            (len(ids), max_blocks_per_seq),
            int(pad_value),
            device=self.device,
            dtype=torch.int32,
        )
        for row, request_id in enumerate(ids):
            block_ids = self._requests[request_id].block_ids
            if len(block_ids) > max_blocks_per_seq:
                raise ValueError("max_blocks_per_seq is smaller than a request's block count")
            if block_ids:
                table[row, : len(block_ids)] = torch.tensor(block_ids, device=self.device, dtype=torch.int32)
        return table

    def seq_lens_tensor(self, request_ids=None, device=None):
        """Return sequence lengths for the requested rows."""
        torch = _torch()
        ids = self._normalize_active_request_ids(request_ids)
        if device is None:
            device = self.device
        lengths = [self._requests[request_id].seq_len for request_id in ids]
        return torch.tensor(lengths, device=device, dtype=torch.int32)

    def request_block_ids(self, request_id):
        """Return the logical-to-physical block list for one request."""
        state = self._require_request(request_id)
        return tuple(state.block_ids)

    def to_dense(self, layer_idx=0, request_ids=None, max_seq_len=None):
        """Materialize the paged cache into dense cache tensors.

        Returns:
            (k_dense, v_dense, seq_lens)

        k_dense/v_dense shape:
            [num_requests, max_seq_len, num_kv_heads, head_dim]
        """
        torch = _torch()
        layer_idx = self._validate_layer_idx(layer_idx)
        ids = self._normalize_active_request_ids(request_ids)
        max_actual_seq_len = max((self._requests[rid].seq_len for rid in ids), default=0)
        if max_seq_len is None:
            max_seq_len = max(1, max_actual_seq_len)
        max_seq_len = int(max_seq_len)
        if max_seq_len < max_actual_seq_len:
            raise ValueError("max_seq_len is smaller than an existing request length")

        dense_shape = (len(ids), max_seq_len, self.num_kv_heads, self.head_dim)
        k_dense = torch.zeros(dense_shape, device=self.device, dtype=self.dtype)
        v_dense = torch.zeros_like(k_dense)

        for row, request_id in enumerate(ids):
            state = self._requests[request_id]
            if state.seq_len == 0:
                continue
            block_ids = torch.tensor(state.block_ids, device=self.device, dtype=torch.long)
            k_blocks = self.k_cache[layer_idx].index_select(0, block_ids)
            v_blocks = self.v_cache[layer_idx].index_select(0, block_ids)
            k_tokens = k_blocks.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
            v_tokens = v_blocks.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
            k_dense[row, : state.seq_len] = k_tokens[: state.seq_len]
            v_dense[row, : state.seq_len] = v_tokens[: state.seq_len]

        return k_dense, v_dense, self.seq_lens_tensor(ids)

    @property
    def num_free_blocks(self):
        """Return currently unowned physical blocks."""
        return len(self._free_blocks)

    @property
    def num_used_blocks(self):
        """Return resident prefix and request-private physical blocks in use."""
        return self.max_blocks - len(self._free_blocks)

    @property
    def state_version(self):
        """Return the logical ownership/sequence mutation version."""
        return self._state_version

    @property
    def has_open_transaction(self):
        """Return whether a multi-layer token transaction is open."""
        return self._open_transaction_id is not None

    def _transaction_view(self, transaction):
        torch = _torch()
        block_ids = torch.tensor(
            [location[0] for location in transaction.locations],
            device=self.device,
            dtype=torch.int64,
        )
        block_offsets = torch.tensor(
            [location[1] for location in transaction.locations],
            device=self.device,
            dtype=torch.int64,
        )
        positions = torch.tensor(
            transaction.positions,
            device=self.device,
            dtype=torch.int64,
        )
        effective_seq_lens = torch.tensor(
            [position + 1 for position in transaction.positions],
            device=self.device,
            dtype=torch.int32,
        )
        return KVTokenTransactionView(
            transaction_id=transaction.transaction_id,
            cache_version=transaction.cache_version,
            request_ids=transaction.request_ids,
            positions=positions,
            physical_block_ids=block_ids,
            block_offsets=block_offsets,
            block_tables=transaction.block_tables.clone(),
            effective_seq_lens=effective_seq_lens,
            next_layer_idx=transaction.next_layer_idx,
            state=transaction.state,
        )

    def _require_transaction(self, transaction):
        if not isinstance(transaction, KVTokenTransactionView):
            raise TypeError("transaction must be a KVTokenTransactionView")
        state = self._transactions.get(transaction.transaction_id)
        if state is None:
            raise RuntimeError("unknown token transaction")
        if (
            transaction.cache_version != state.cache_version
            or transaction.request_ids != state.request_ids
        ):
            raise RuntimeError("stale or invalid token transaction handle")
        return state

    def _require_open_transaction(self, transaction):
        state = self._require_transaction(transaction)
        if state.state != "open":
            raise RuntimeError(f"token transaction is already {state.state}")
        if self._open_transaction_id != state.transaction_id:
            raise RuntimeError("token transaction is not the current open transaction")
        return state

    def _require_legacy_append_available(self):
        if self.num_layers != 1:
            raise RuntimeError("multi-layer cache writes require the token transaction API")
        if self._open_transaction_id is not None:
            raise RuntimeError("legacy append is not allowed during an open token transaction")

    def _prepare_append_inputs(self, layer_idx, request_ids, k, v):
        layer_idx = self._validate_layer_idx(layer_idx)
        ids = self._normalize_request_ids(request_ids)
        if not ids:
            raise ValueError("request_ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique within one append call")

        if k.dim() == 2 and len(ids) == 1:
            k = k.unsqueeze(0)
        if v.dim() == 2 and len(ids) == 1:
            v = v.unsqueeze(0)
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError("k and v must have shape [num_requests, num_kv_heads, head_dim]")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if k.shape != (len(ids), self.num_kv_heads, self.head_dim):
            raise ValueError("k and v shapes must match request count, num_kv_heads, and head_dim")
        if k.device != self.device or v.device != self.device:
            raise ValueError("k and v must be on the cache device")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError("k and v must use the cache dtype")
        if not k.is_floating_point() or not v.is_floating_point():
            raise ValueError("k and v must be floating point tensors")
        return layer_idx, ids, k, v

    def _prepare_transaction_layer_write(self, state, layer_idx, k, v):
        layer_idx = self._validate_layer_idx(layer_idx)
        if layer_idx != state.next_layer_idx:
            raise RuntimeError(
                f"transaction requires layer {state.next_layer_idx}, got layer {layer_idx}"
            )
        _, ids, k, v = self._prepare_append_inputs(
            layer_idx,
            state.request_ids,
            k,
            v,
        )
        if tuple(ids) != state.request_ids:
            raise RuntimeError("transaction request order changed unexpectedly")
        return layer_idx, k, v

    def _record_transaction_layer_write(self, state, layer_idx):
        state.written_layers.add(layer_idx)
        state.next_layer_idx += 1
        self._transaction_layer_write_count += 1
        return self._transaction_view(state)

    def _preflight_append(self, ids):
        needed_new_blocks = 0
        for request_id in ids:
            state = self._requests.get(request_id)
            if state is not None and state.status != self.ACTIVE:
                raise RuntimeError(
                    f"request_id {request_id!r} is {state.status} and cannot accept append"
                )
            if state is None:
                state = _RequestState()
            if state.seq_len % self.block_size == 0:
                needed_new_blocks += 1
        if needed_new_blocks > len(self._free_blocks):
            self._capacity_failure_count += 1
            raise RuntimeError("PagedKVCache is out of physical blocks")

    def _allocate_append_locations(self, ids):
        locations = []
        for request_id in ids:
            state = self._requests.get(request_id)
            if state is None:
                state = _RequestState()
                self._requests[request_id] = state
            token_index = state.seq_len
            logical_block = token_index // self.block_size
            block_offset = token_index % self.block_size
            if block_offset == 0:
                state.block_ids.append(self._allocate_block())
            locations.append((state.block_ids[logical_block], block_offset))
        return locations

    def _advance_request_lengths(self, ids):
        for request_id in ids:
            self._requests[request_id].seq_len += 1

    def _allocate_block(self):
        if not self._free_blocks:
            raise RuntimeError("PagedKVCache is out of physical blocks")
        block_id = self._free_blocks.popleft()
        self._allocation_count += 1
        if block_id in self._ever_allocated_blocks:
            self._reuse_count += 1
        else:
            self._ever_allocated_blocks.add(block_id)
            self._fresh_allocation_count += 1
        return block_id

    def _close_request(self, request_id, terminal_status):
        if self._open_transaction_id is not None:
            raise RuntimeError("cannot close a request during an open token transaction")
        state = self._require_active_request(request_id)
        prefix = None
        if state.prefix_id is not None:
            prefix = self._prefixes.get(state.prefix_id)
            if prefix is None or prefix.active_refcount <= 0:
                raise RuntimeError("shared prefix reference accounting is inconsistent")
        released_blocks = tuple(state.block_ids[state.shared_block_count :])
        # Do not zero K/V here: ownership plus the next request's seq_len masks
        # stale tail slots, while avoiding an extra device-wide cleanup path.
        for block_id in reversed(released_blocks):
            self._free_blocks.appendleft(block_id)
        self._free_count += len(released_blocks)
        state.block_ids.clear()
        if prefix is not None:
            prefix.active_refcount -= 1
            self._prefixes.move_to_end(state.prefix_id)
        state.prefix_id = None
        state.shared_block_count = 0
        state.status = terminal_status
        self._state_version += 1
        return released_blocks

    def _validate_layer_idx(self, layer_idx):
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError("layer_idx must be in [0, num_layers)")
        return layer_idx

    def _normalize_request_ids(self, request_ids):
        torch = _torch()
        if isinstance(request_ids, torch.Tensor):
            if request_ids.dim() == 0:
                ids = [int(request_ids.detach().cpu().item())]
            else:
                ids = [int(value) for value in request_ids.detach().cpu().reshape(-1).tolist()]
        elif isinstance(request_ids, (str, bytes)):
            ids = [request_ids]
        else:
            try:
                ids = list(request_ids)
            except TypeError:
                ids = [request_ids]

        for request_id in ids:
            self._check_request_id(request_id)
        return ids

    def _normalize_active_request_ids(self, request_ids):
        if request_ids is None:
            ids = [
                request_id
                for request_id, state in self._requests.items()
                if state.status == self.ACTIVE
            ]
        else:
            ids = self._normalize_request_ids(request_ids)
        for request_id in ids:
            self._require_active_request(request_id)
        return ids

    @staticmethod
    def _check_request_id(request_id):
        try:
            hash(request_id)
        except TypeError as exc:
            raise ValueError("request ids must be hashable") from exc

    @staticmethod
    def _check_prefix_id(prefix_id):
        if prefix_id is None:
            raise ValueError("prefix_id must not be None")
        try:
            hash(prefix_id)
        except TypeError as exc:
            raise ValueError("prefix ids must be hashable") from exc

    def _prepare_prefix_blocks(self, k_blocks, v_blocks):
        torch = _torch()
        if not isinstance(k_blocks, torch.Tensor) or not isinstance(v_blocks, torch.Tensor):
            raise TypeError("k_blocks and v_blocks must be torch tensors")
        if k_blocks.dim() != 5 or v_blocks.dim() != 5:
            raise ValueError(
                "prefix K/V must have shape "
                "[num_layers, num_prefix_blocks, num_kv_heads, block_size, head_dim]"
            )
        if k_blocks.shape != v_blocks.shape:
            raise ValueError("prefix k_blocks and v_blocks must have the same shape")
        expected_tail = (
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )
        if k_blocks.shape[0] != self.num_layers or tuple(k_blocks.shape[2:]) != expected_tail:
            raise ValueError("prefix K/V shape does not match the PagedKVCache layout")
        if k_blocks.shape[1] <= 0:
            raise ValueError("prefix must contain at least one full block")
        if k_blocks.device != self.device or v_blocks.device != self.device:
            raise ValueError("prefix K/V must be on the cache device")
        if k_blocks.dtype != self.dtype or v_blocks.dtype != self.dtype:
            raise ValueError("prefix K/V must use the cache dtype")
        return k_blocks, v_blocks

    def _resident_prefix_blocks(self):
        return sum(len(prefix.block_ids) for prefix in self._prefixes.values())

    def _prefix_eviction_plan(self, required_blocks):
        if required_blocks > self.prefix_cache_capacity_blocks:
            return None
        resident_blocks = self._resident_prefix_blocks()
        needed_release = max(
            0,
            resident_blocks + required_blocks - self.prefix_cache_capacity_blocks,
            required_blocks - len(self._free_blocks),
        )
        if not needed_release:
            return ()

        released = 0
        eviction_ids = []
        for prefix_id, prefix in self._prefixes.items():
            if prefix.active_refcount:
                continue
            eviction_ids.append(prefix_id)
            released += len(prefix.block_ids)
            if released >= needed_release:
                return tuple(eviction_ids)
        return None

    def _evict_prefix_entry(self, prefix_id):
        prefix = self._prefixes[prefix_id]
        if prefix.active_refcount:
            raise RuntimeError("cannot evict a prefix with active request references")
        del self._prefixes[prefix_id]
        for block_id in reversed(prefix.block_ids):
            self._free_blocks.appendleft(block_id)
        self._free_count += len(prefix.block_ids)
        self._prefix_eviction_count += 1
        return tuple(prefix.block_ids)

    def _require_request(self, request_id):
        state = self._requests.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id: {request_id!r}")
        return state

    def _require_active_request(self, request_id):
        state = self._require_request(request_id)
        if state.status != self.ACTIVE:
            raise RuntimeError(f"request_id {request_id!r} is {state.status}, not active")
        return state
