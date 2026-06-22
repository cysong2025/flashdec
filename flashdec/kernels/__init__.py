"""Triton kernels for FlashDec.

Imports are lazy so documentation and CPU-only helper code can be used on
machines without PyTorch/Triton installed.
"""

__all__ = ["rmsnorm", "row_softmax", "vector_add"]


def vector_add(*args, **kwargs):
    from .vector_add import vector_add as _vector_add

    return _vector_add(*args, **kwargs)


def row_softmax(*args, **kwargs):
    from .softmax import row_softmax as _row_softmax

    return _row_softmax(*args, **kwargs)


def rmsnorm(*args, **kwargs):
    from .rmsnorm import rmsnorm as _rmsnorm

    return _rmsnorm(*args, **kwargs)
