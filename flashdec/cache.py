"""Paged KV cache runtime used by the paged attention reference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable


@dataclass
class _RequestState:
    status: str = "active"
    seq_len: int = 0
    block_ids: list[int] = field(default_factory=list)


def _torch():
    import torch

    return torch


class PagedKVCache:
    """A small fixed-block KV cache for single-token decode experiments.

    The physical cache layout is:

    [num_layers, max_blocks, num_kv_heads, block_size, head_dim]

    Each request owns a logical list of physical block ids. Appending one token
    writes into the current tail block or allocates a new physical block when
    the logical position crosses a block boundary. Finished or cancelled
    requests release their blocks back to a deterministic free list that
    prioritizes reuse.
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
        if int(num_layers) != 1:
            raise ValueError("PagedKVCache runtime v2 currently supports num_layers=1")

        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be a floating point torch dtype")

        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
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
        self._ever_allocated_blocks: set[int] = set()
        self._allocation_count = 0
        self._fresh_allocation_count = 0
        self._free_count = 0
        self._reuse_count = 0
        self._capacity_failure_count = 0

    def add_request(self, request_id):
        """Register an empty request if it does not already exist."""
        self._check_request_id(request_id)
        state = self._requests.get(request_id)
        if state is None:
            self._requests[request_id] = _RequestState()
        elif state.status != self.ACTIVE:
            raise RuntimeError(
                f"request_id {request_id!r} is {state.status} and cannot be reactivated"
            )
        return self

    def append(self, layer_idx, request_ids, k, v):
        """Append one token of K/V for each request.

        Args:
            layer_idx: Layer to write. Week 5 validates the single-layer path,
                but the storage keeps a layer dimension for later extension.
            request_ids: Request ids in the same order as the first dimension of
                k and v.
            k/v: [num_requests, num_kv_heads, head_dim]. For one request,
                [num_kv_heads, head_dim] is also accepted.

        Returns:
            A padded block table tensor for the provided request ids.
        """
        layer_idx, ids, k, v = self._prepare_append_inputs(layer_idx, request_ids, k, v)
        self._preflight_append(ids)
        locations = self._allocate_append_locations(ids)
        for row, (physical_block, block_offset) in enumerate(locations):
            self.k_cache[layer_idx, physical_block, :, block_offset, :] = k[row]
            self.v_cache[layer_idx, physical_block, :, block_offset, :] = v[row]
        self._advance_request_lengths(ids)
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
        PyTorch RoPE reference before appending until the fused path exists.
        """
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
        return self.block_tables(ids)

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
        reserved_tokens = used_blocks * self.block_size
        active_tokens = sum(state.seq_len for state in active_states)
        fragmentation_tokens = reserved_tokens - active_tokens
        return {
            "max_blocks": self.max_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.num_free_blocks,
            "block_utilization": used_blocks / self.max_blocks,
            "active_tokens": active_tokens,
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
        }

    def validate_invariants(self):
        """Raise RuntimeError if request ownership and the free list diverge."""
        free_blocks = list(self._free_blocks)
        if len(free_blocks) != len(set(free_blocks)):
            raise RuntimeError("PagedKVCache free list contains duplicate blocks")

        owned_blocks = []
        for state in self._requests.values():
            if state.status == self.ACTIVE:
                expected_blocks = (state.seq_len + self.block_size - 1) // self.block_size
                if len(state.block_ids) != expected_blocks:
                    raise RuntimeError("active request block count does not match seq_len")
                owned_blocks.extend(state.block_ids)
            elif state.block_ids:
                raise RuntimeError("terminal request still owns physical blocks")

        if len(owned_blocks) != len(set(owned_blocks)):
            raise RuntimeError("a physical block is owned by multiple requests")
        if set(owned_blocks) & set(free_blocks):
            raise RuntimeError("a physical block is both owned and free")
        if set(owned_blocks) | set(free_blocks) != set(range(self.max_blocks)):
            raise RuntimeError("physical block accounting does not cover the cache")
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
        return len(self._free_blocks)

    @property
    def num_used_blocks(self):
        return self.max_blocks - len(self._free_blocks)

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
            self.add_request(request_id)
            state = self._requests[request_id]
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
        state = self._require_active_request(request_id)
        released_blocks = tuple(state.block_ids)
        # Do not zero K/V here: ownership plus the next request's seq_len masks
        # stale tail slots, while avoiding an extra device-wide cleanup path.
        for block_id in reversed(released_blocks):
            self._free_blocks.appendleft(block_id)
        self._free_count += len(released_blocks)
        state.block_ids.clear()
        state.status = terminal_status
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
