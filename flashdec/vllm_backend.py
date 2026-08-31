"""Out-of-tree vLLM attention backend backed by FlashDec decode kernels.

The integration deliberately inherits vLLM's Triton backend contract. vLLM
continues to own KV-cache allocation/update, metadata construction, prefill,
mixed batches, and all unsupported attention features. FlashDec replaces only
uniform single-token decoder attention over FP16/BF16 paged KV cache.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from .kernels.paged_decode import _vllm_paged_decode_attention_into
from .vllm_attestation import (
    SPLIT_ATTESTATION_BACKEND,
    SPLIT_ATTESTATION_ENV,
    SPLIT_ATTESTATION_SCHEMA_VERSION,
    canonical_attestation_bytes,
    validate_split_attestation,
)


_VALID_REQUESTED_SPLITS = (0, 1, 2, 4, 8, 16)
_VALID_NUM_SPLITS = _VALID_REQUESTED_SPLITS[1:]
_ATTESTATION_LOCK = threading.Lock()
_ATTESTED_PROCESS_PATHS: set[tuple[int, str]] = set()


def _split_attestation_binding_from_env() -> dict[str, Any] | None:
    values = {
        field: os.environ.get(name)
        for field, name in SPLIT_ATTESTATION_ENV.items()
    }
    present = {field for field, value in values.items() if value is not None}
    if not present:
        return None
    if present != set(SPLIT_ATTESTATION_ENV):
        missing = sorted(set(SPLIT_ATTESTATION_ENV) - present)
        raise RuntimeError(
            "incomplete FlashDec split-attestation environment; missing "
            f"{missing}"
        )
    path = Path(values["path"])
    if not path.is_absolute():
        raise RuntimeError("FlashDec split-attestation path must be absolute")
    try:
        trial = int(values["trial"])
    except ValueError as error:
        raise RuntimeError(
            "FlashDec split-attestation trial must be an integer"
        ) from error
    if trial <= 0:
        raise RuntimeError("FlashDec split-attestation trial must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", values["nonce"]):
        raise RuntimeError(
            "FlashDec split-attestation nonce must be 64 lowercase hex digits"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", values["case"]):
        raise RuntimeError("FlashDec split-attestation case is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", values["dataset_sha256"]):
        raise RuntimeError("FlashDec split-attestation dataset SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{7,64}", values["git_commit"]):
        raise RuntimeError("FlashDec split-attestation Git commit is invalid")
    return {
        "path": str(path),
        "nonce": values["nonce"],
        "case": values["case"],
        "trial": trial,
        "dataset_sha256": values["dataset_sha256"],
        "git_commit": values["git_commit"],
    }


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while creating split-attestation marker")
        offset += written


def _write_split_attestation(
    binding: dict[str, Any],
    *,
    max_seq_len: int,
    logical_blocks: int,
    num_reqs: int,
    num_splits: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    query_dtype: str,
    kv_cache_dtype: str,
    cuda_graph_capture: bool,
) -> None:
    """Atomically record the first successful split launch in this engine."""

    process_key = (os.getpid(), binding["path"])
    if process_key in _ATTESTED_PROCESS_PATHS:
        return
    payload = {
        "schema_version": SPLIT_ATTESTATION_SCHEMA_VERSION,
        "nonce": binding["nonce"],
        "engine_pid": os.getpid(),
        "backend": SPLIT_ATTESTATION_BACKEND,
        "case": binding["case"],
        "trial": binding["trial"],
        "dataset_sha256": binding["dataset_sha256"],
        "git_commit": binding["git_commit"],
        "max_seq_len": max_seq_len,
        "logical_blocks": logical_blocks,
        "num_reqs": num_reqs,
        "num_splits": num_splits,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "query_dtype": query_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "cuda_graph_capture": cuda_graph_capture,
    }
    validate_split_attestation(payload)
    encoded = canonical_attestation_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with _ATTESTATION_LOCK:
        if process_key in _ATTESTED_PROCESS_PATHS:
            return
        fd = os.open(binding["path"], flags, 0o600)
        try:
            _write_all(fd, encoded)
        finally:
            os.close(fd)
        _ATTESTED_PROCESS_PATHS.add(process_key)


def _parse_requested_splits(value: str) -> int:
    try:
        requested_splits = int(value)
    except ValueError:
        requested_splits = -1
    if requested_splits not in _VALID_REQUESTED_SPLITS:
        raise ValueError(
            "FLASHDEC_VLLM_NUM_SPLITS must be one of 0, 1, 2, 4, 8, or 16 "
            "(0 selects auto)"
        )
    return requested_splits


def _select_num_splits(
    requested_splits: int,
    *,
    num_reqs: int,
    num_kv_heads: int,
    logical_blocks: int,
) -> int:
    max_context_splits = max(
        split for split in _VALID_NUM_SPLITS if split <= max(1, logical_blocks)
    )
    if requested_splits == 0:
        target_programs = 128
        programs_per_split = max(1, num_reqs * num_kv_heads)
        requested_splits = min(
            _VALID_NUM_SPLITS,
            key=lambda split: (
                abs(programs_per_split * split - target_programs),
                split,
            ),
        )
    return min(requested_splits, max_context_splits)


class FlashDecAttentionBackend(TritonAttentionBackend):
    """vLLM backend that substitutes FlashDec for eligible decode batches."""

    # Keep vLLM's separate KV-update op and its explicit torch.compile data
    # dependency for every path. The FlashDec kernel currently writes the same
    # K/V token again on eligible decode; that redundant store is preferable to
    # bypassing vLLM's graph-level ordering contract.
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        # vLLM 0.25.1 maps the selected class back through the registry using
        # get_name(); third-party backends must therefore return their enum
        # slot rather than a package-specific display label.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["FlashDecAttentionImpl"]:
        return FlashDecAttentionImpl


class FlashDecAttentionImpl(TritonAttentionImpl):
    """Use FlashDec for the narrow path it implements; delegate everything else."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._requested_splits = _parse_requested_splits(
            os.environ.get("FLASHDEC_VLLM_NUM_SPLITS", "0")
        )
        # The normal integration path has no attestation environment and does
        # no filesystem work. Formal workers opt in before constructing LLM so
        # every per-layer backend instance inherits the same one-shot binding.
        self._split_attestation_binding = _split_attestation_binding_from_env()

    def _supports_flashdec_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if attn_metadata is None:
            return False
        if self.attn_type != AttentionType.DECODER:
            return False
        if attn_metadata.max_seq_len < 512:
            return False
        if attn_metadata.max_query_len != 1 or attn_metadata.use_cascade:
            return False
        if attn_metadata.causal is not True:
            return False
        if self.need_to_return_lse_for_decode:
            return False
        if output_scale is not None or output_block_scale is not None:
            return False
        if self.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if kv_cache.dtype != query.dtype:
            return False
        if (
            query.ndim != 3
            or key.ndim != 3
            or value.ndim != 3
            or output.ndim != 3
            or kv_cache.ndim != 5
        ):
            return False
        if self.head_size != 128:
            return False
        if (
            kv_cache.shape[1] != 2
            or kv_cache.shape[2] not in (16, 32)
            or kv_cache.shape[3] != self.num_kv_heads
            or kv_cache.shape[4] != self.head_size
        ):
            return False
        if self.alibi_slopes is not None or self.sinks is not None:
            return False
        if self.sliding_window != (-1, -1) or self.logits_soft_cap != 0:
            return False
        return (
            self.num_heads % self.num_kv_heads == 0
            and self.num_heads // self.num_kv_heads in (4, 8, 16)
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        owns_kv = (
            getattr(layer, "kv_sharing_target_layer_name", None) is None
            and key is not None
            and value is not None
        )
        if not owns_kv or not self._supports_flashdec_decode(
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # Uniform single-token decode has one query token per metadata row. In
        # CUDA Graph replay this row count includes padding; slot_mapping=-1
        # makes those rows inert inside the kernel without a device-to-host read.
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        block_size = kv_cache.shape[2]
        logical_blocks = (
            attn_metadata.max_seq_len + block_size - 1
        ) // block_size
        num_splits = _select_num_splits(
            self._requested_splits,
            num_reqs=num_reqs,
            num_kv_heads=self.num_kv_heads,
            logical_blocks=logical_blocks,
        )
        if num_reqs > attn_metadata.softmax_segm_output.shape[0]:
            num_splits = 1
        if num_splits == 1:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        _vllm_paged_decode_attention_into(
            query,
            key,
            value,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            output,
            attn_metadata.slot_mapping,
            attn_metadata.softmax_segm_output,
            attn_metadata.softmax_segm_max,
            attn_metadata.softmax_segm_expsum,
            num_reqs=num_reqs,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            block_size=block_size,
            sm_scale=self.scale,
            num_splits=num_splits,
        )
        binding = getattr(self, "_split_attestation_binding", None)
        if binding is not None:
            _write_split_attestation(
                binding,
                max_seq_len=attn_metadata.max_seq_len,
                logical_blocks=logical_blocks,
                num_reqs=num_reqs,
                num_splits=num_splits,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                block_size=block_size,
                query_dtype=str(query.dtype).removeprefix("torch."),
                kv_cache_dtype=str(kv_cache.dtype).removeprefix("torch."),
                cuda_graph_capture=bool(
                    torch.cuda.is_current_stream_capturing()
                ),
            )
            # Each layer pays at most one set lookup. CUDA Graph replay never
            # re-enters this Python branch, and ordinary users never enable it.
            self._split_attestation_binding = None
        return output
