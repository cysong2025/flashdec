import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("vllm")

from flashdec.vllm_backend import (
    FlashDecAttentionBackend,
    FlashDecAttentionImpl,
)


def test_vllm_backend_identity_and_cache_contract():
    assert FlashDecAttentionBackend.get_name() == "CUSTOM"
    assert FlashDecAttentionBackend.get_impl_cls() is FlashDecAttentionImpl
    assert FlashDecAttentionBackend.forward_includes_kv_cache_update is True
    assert FlashDecAttentionBackend.get_kv_cache_shape(7, 16, 2, 128) == (
        7,
        2,
        16,
        2,
        128,
    )
