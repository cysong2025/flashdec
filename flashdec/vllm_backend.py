"""Out-of-tree vLLM attention backend backed by FlashDec decode kernels.

The integration deliberately inherits vLLM's Triton backend contract. vLLM
continues to own KV-cache allocation/update, metadata construction, prefill,
mixed batches, and all unsupported attention features. FlashDec replaces only
uniform single-token decoder attention over FP16/BF16 paged KV cache.
"""

from __future__ import annotations

import math
import os

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from .kernels.paged_decode import paged_decode_attention_into


class FlashDecAttentionBackend(TritonAttentionBackend):
    """vLLM backend that substitutes FlashDec for eligible decode batches."""

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
        requested_splits = int(os.environ.get("FLASHDEC_VLLM_NUM_SPLITS", "0"))
        if requested_splits < 0 or requested_splits > 16:
            raise ValueError("FLASHDEC_VLLM_NUM_SPLITS must be in [0, 16]")
        self._requested_splits = requested_splits

    def _supports_flashdec_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata | None,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if attn_metadata is None:
            return False
        if self.attn_type != AttentionType.DECODER:
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
        if kv_cache.dtype != query.dtype or kv_cache.ndim != 5:
            return False
        if self.head_size not in (64, 128):
            return False
        if kv_cache.shape[2] not in (8, 16, 32):
            return False
        if self.alibi_slopes is not None or self.sinks is not None:
            return False
        if self.sliding_window != (-1, -1) or self.logits_soft_cap != 0:
            return False
        return self.num_heads % self.num_kv_heads == 0

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
        if not self._supports_flashdec_decode(
            query,
            kv_cache,
            attn_metadata,
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

        # Uniform single-token decode has one query token per request. Shapes
        # are Python metadata only; no device-to-host read enters the hot path.
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        query_3d = query[:num_reqs].view(num_reqs, self.num_heads, self.head_size)
        output_3d = output[:num_reqs].view(
            num_reqs, self.num_heads, self.head_size
        )

        # vLLM's logical NHD cache halves are [block, token, kv_head, dim].
        # Permutation creates the FlashDec [block, kv_head, token, dim] view;
        # paged_decode_attention_into consumes its strides without a copy.
        key_cache, value_cache = kv_cache.unbind(1)
        key_view = key_cache.permute(0, 2, 1, 3)
        value_view = value_cache.permute(0, 2, 1, 3)

        logical_blocks = math.ceil(attn_metadata.max_seq_len / key_cache.shape[1])
        requested_splits = self._requested_splits
        if requested_splits == 0:
            target_programs = 128
            requested_splits = math.ceil(
                target_programs / max(1, num_reqs * self.num_kv_heads)
            )
        num_splits = min(16, logical_blocks, max(1, requested_splits))
        if attn_metadata.max_seq_len < 512:
            num_splits = 1

        paged_decode_attention_into(
            query_3d,
            key_view,
            value_view,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            output_3d,
            sm_scale=self.scale,
            block_size=key_cache.shape[1],
            num_warps=2,
            split_kv_workspace=(
                attn_metadata.softmax_segm_output,
                attn_metadata.softmax_segm_max,
                attn_metadata.softmax_segm_expsum,
            ),
            num_splits=num_splits,
        )
        return output
