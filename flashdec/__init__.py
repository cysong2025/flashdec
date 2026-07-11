"""FlashDec: LLM decode attention and paged KV cache experiments."""

__all__ = [
    "PagedKVCache",
    "__version__",
    "decode",
    "dense_decode_attention_ref",
    "paged_decode_attention",
    "paged_decode_attention_ref",
]

__version__ = "0.0.0"


def __getattr__(name):
    if name == "PagedKVCache":
        from .cache import PagedKVCache

        return PagedKVCache
    if name == "dense_decode_attention_ref":
        from .reference import dense_decode_attention_ref

        return dense_decode_attention_ref
    if name in ("decode", "paged_decode_attention"):
        from .kernels.paged_decode import paged_decode_attention

        return paged_decode_attention
    if name == "paged_decode_attention_ref":
        from .paged_reference import paged_decode_attention_ref

        return paged_decode_attention_ref
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
