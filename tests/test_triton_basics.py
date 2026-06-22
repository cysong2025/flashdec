import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flashdec.kernels.rmsnorm import rmsnorm
from flashdec.kernels.softmax import row_softmax
from flashdec.kernels.vector_add import vector_add
from flashdec.reference import rmsnorm_ref, row_softmax_ref, vector_add_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is required for Triton kernel tests",
)


@pytest.mark.parametrize("size", [1, 17, 1024, 100_003])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vector_add_matches_torch(size, dtype):
    x = torch.randn(size, device="cuda", dtype=dtype)
    y = torch.randn(size, device="cuda", dtype=dtype)

    actual = vector_add(x, y)
    expected = vector_add_ref(x, y)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(1, 16), (4, 17), (32, 128), (8, 1025)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_row_softmax_matches_torch(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)

    actual = row_softmax(x)
    expected = row_softmax_ref(x)

    if dtype is torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 16), (4, 64), (16, 128), (8, 513)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm_matches_torch(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    actual = rmsnorm(x, weight)
    expected = rmsnorm_ref(x, weight)

    if dtype is torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

