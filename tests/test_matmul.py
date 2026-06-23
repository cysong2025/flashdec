import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashdec.kernels.matmul import matmul, matmul_autotuned
from flashdec.reference import matmul_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for Triton kernel tests",
)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (128, 64, 256),
        (257, 129, 65),
    ],
)
def test_matmul_matches_torch(shape):
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    actual = matmul(a, b)
    expected = matmul_ref(a, b)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(32, 32, 32), (128, 128, 128), (257, 129, 65)])
def test_matmul_autotuned_matches_torch(shape):
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    actual = matmul_autotuned(a, b)
    expected = matmul_ref(a, b)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
