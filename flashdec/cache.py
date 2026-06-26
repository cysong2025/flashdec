"""Paged KV cache runtime used by the paged attention reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable


@dataclass
class _RequestState:
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
    the logical position crosses a block boundary.
    """

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
        self.v_cache = torch.zeros_like(self.k_cache)
        self._free_blocks = list(range(self.max_blocks))
        self._requests: dict[Hashable, _RequestState] = {}

    def add_request(self, request_id):
        """Register an empty request if it does not already exist."""
        self._check_request_id(request_id)
        if request_id not in self._requests:
            self._requests[request_id] = _RequestState()
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

        needed_new_blocks = 0
        for request_id in ids:
            state = self._requests.get(request_id, _RequestState())
            if state.seq_len % self.block_size == 0:
                needed_new_blocks += 1
        if needed_new_blocks > len(self._free_blocks):
            raise RuntimeError("PagedKVCache is out of physical blocks")

        for row, request_id in enumerate(ids):
            self.add_request(request_id)
            state = self._requests[request_id]
            token_index = state.seq_len
            logical_block = token_index // self.block_size
            block_offset = token_index % self.block_size

            if block_offset == 0:
                state.block_ids.append(self._allocate_block())
            physical_block = state.block_ids[logical_block]

            self.k_cache[layer_idx, physical_block, :, block_offset, :] = k[row]
            self.v_cache[layer_idx, physical_block, :, block_offset, :] = v[row]
            state.seq_len += 1

        return self.block_tables(ids)

    def block_tables(self, request_ids=None, max_blocks_per_seq=None, pad_value=-1):
        """Return padded physical block ids for each request.

        Shape: [num_requests, max_blocks_per_seq].
        """
        torch = _torch()
        ids = self._normalize_existing_request_ids(request_ids)
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
        ids = self._normalize_existing_request_ids(request_ids)
        if device is None:
            device = self.device
        lengths = [self._requests[request_id].seq_len for request_id in ids]
        return torch.tensor(lengths, device=device, dtype=torch.int32)

    def request_block_ids(self, request_id):
        """Return the logical-to-physical block list for one request."""
        self._require_request(request_id)
        return tuple(self._requests[request_id].block_ids)

    def to_dense(self, layer_idx=0, request_ids=None, max_seq_len=None):
        """Materialize the paged cache into dense cache tensors.

        Returns:
            (k_dense, v_dense, seq_lens)

        k_dense/v_dense shape:
            [num_requests, max_seq_len, num_kv_heads, head_dim]
        """
        torch = _torch()
        layer_idx = self._validate_layer_idx(layer_idx)
        ids = self._normalize_existing_request_ids(request_ids)
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

    def _allocate_block(self):
        if not self._free_blocks:
            raise RuntimeError("PagedKVCache is out of physical blocks")
        return self._free_blocks.pop(0)

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

    def _normalize_existing_request_ids(self, request_ids):
        if request_ids is None:
            ids = list(self._requests.keys())
        else:
            ids = self._normalize_request_ids(request_ids)
        for request_id in ids:
            self._require_request(request_id)
        return ids

    @staticmethod
    def _check_request_id(request_id):
        try:
            hash(request_id)
        except TypeError as exc:
            raise ValueError("request ids must be hashable") from exc

    def _require_request(self, request_id):
        if request_id not in self._requests:
            raise KeyError(f"unknown request_id: {request_id!r}")
