"""Out-of-tree vLLM attention backend backed by FlashDec decode kernels.

The integration deliberately inherits vLLM's Triton backend contract. vLLM
continues to own KV-cache allocation/update, metadata construction, prefill,
mixed batches, and all unsupported attention features. FlashDec replaces only
uniform single-token decoder attention over FP16/BF16 paged KV cache.
"""

from __future__ import annotations

import os

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from .kernels.paged_decode import _vllm_paged_decode_attention_into


_VALID_REQUESTED_SPLITS = (0, 1, 2, 4, 8, 16)
_VALID_NUM_SPLITS = _VALID_REQUESTED_SPLITS[1:]


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

    # Eligible decode writes the current K/V token from the attention kernel.
    # Unsupported paths explicitly invoke Triton's update before delegating.
    forward_includes_kv_cache_update = True

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
            if attn_metadata is not None and owns_kv:
                self.do_kv_cache_update(
                    layer,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.slot_mapping,
                )
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
        return output
