"""vLLM general-plugin entry point for the FlashDec attention backend."""

from __future__ import annotations


def register() -> None:
    """Register FlashDec as vLLM's explicit ``CUSTOM`` attention backend.

    Imports stay inside the entry point so installing FlashDec does not make
    vLLM a mandatory runtime dependency for the standalone kernel/runtime API.
    vLLM calls this function in the frontend, engine-core, and worker processes.
    """
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "flashdec.vllm_backend.FlashDecAttentionBackend",
    )
