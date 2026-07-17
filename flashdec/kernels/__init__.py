"""Triton kernels for FlashDec.

Imports are lazy so documentation and CPU-only helper code can be used on
machines without PyTorch/Triton installed.
"""

__all__ = [
    "dense_decode_attention",
    "matmul",
    "matmul_autotuned",
    "paged_decode_attention",
    "rmsnorm",
    "row_softmax",
    "vector_add",
]


def dense_decode_attention(*args, **kwargs):
    """Lazily dispatch to the Triton dense decode attention kernel."""
    from .dense_decode import dense_decode_attention as _dense_decode_attention

    return _dense_decode_attention(*args, **kwargs)


def matmul(*args, **kwargs):
    """Lazily dispatch to the fixed-configuration Triton matmul kernel."""
    from .matmul import matmul as _matmul

    return _matmul(*args, **kwargs)


def matmul_autotuned(*args, **kwargs):
    """Lazily dispatch to the autotuned Triton matmul kernel."""
    from .matmul import matmul_autotuned as _matmul_autotuned

    return _matmul_autotuned(*args, **kwargs)


def paged_decode_attention(*args, **kwargs):
    """Lazily dispatch to the Triton paged decode attention kernel."""
    from .paged_decode import paged_decode_attention as _paged_decode_attention

    return _paged_decode_attention(*args, **kwargs)


def vector_add(*args, **kwargs):
    """Lazily dispatch to the Triton vector-add kernel."""
    from .vector_add import vector_add as _vector_add

    return _vector_add(*args, **kwargs)


def row_softmax(*args, **kwargs):
    """Lazily dispatch to the Triton row-softmax kernel."""
    from .softmax import row_softmax as _row_softmax

    return _row_softmax(*args, **kwargs)


def rmsnorm(*args, **kwargs):
    """Lazily dispatch to the Triton RMSNorm kernel."""
    from .rmsnorm import rmsnorm as _rmsnorm

    return _rmsnorm(*args, **kwargs)
