"""PyTorch reference implementations used for correctness tests."""

from __future__ import annotations


def vector_add_ref(x, y):
    """Reference for elementwise addition."""
    return x + y


def row_softmax_ref(x):
    """Reference for row-wise softmax over the last dimension."""
    import torch

    return torch.softmax(x, dim=-1)


def rmsnorm_ref(x, weight, eps=1e-6):
    """Reference RMSNorm over the last dimension."""
    import torch

    x_fp32 = x.to(torch.float32)
    variance = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.to(torch.float32)).to(x.dtype)


def matmul_ref(a, b):
    """Reference 2D matrix multiplication."""
    import torch

    return torch.matmul(a, b)
