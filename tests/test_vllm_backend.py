from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("vllm")

import flashdec.vllm_backend as backend_module
from flashdec.vllm_backend import (
    FlashDecAttentionBackend,
    FlashDecAttentionImpl,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


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


def _impl_without_vllm_initialization():
    impl = object.__new__(FlashDecAttentionImpl)
    impl.num_heads = 16
    impl.num_kv_heads = 2
    impl.head_size = 128
    impl.scale = 128**-0.5
    impl._requested_splits = 0
    return impl


def _metadata(num_reqs=3, max_seq_len=1024):
    return SimpleNamespace(
        query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32),
        max_seq_len=max_seq_len,
        block_table=torch.empty((num_reqs, 64), dtype=torch.int32),
        seq_lens=torch.empty(num_reqs, dtype=torch.int32),
        slot_mapping=torch.empty(num_reqs, dtype=torch.int64),
        softmax_segm_output=SimpleNamespace(shape=(64, 16, 16, 128)),
        softmax_segm_max=object(),
        softmax_segm_expsum=object(),
    )


@pytest.mark.parametrize("value", ["0", "1", "2", "4", "8", "16"])
def test_requested_split_parser_accepts_only_supported_values(value):
    assert backend_module._parse_requested_splits(value) == int(value)


@pytest.mark.parametrize("value", ["auto", "-1", "3", "6", "11", "12", "17"])
def test_requested_split_parser_rejects_unsupported_values(value):
    with pytest.raises(
        ValueError,
        match=(
            r"FLASHDEC_VLLM_NUM_SPLITS must be one of "
            r"0, 1, 2, 4, 8, or 16 \(0 selects auto\)"
        ),
    ):
        backend_module._parse_requested_splits(value)


@pytest.mark.parametrize(
    ("num_reqs", "expected"),
    [
        (4, 16),
        (5, 16),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 8),
        (10, 8),
        (11, 4),
        (12, 4),
    ],
)
def test_auto_split_policy_uses_power_of_two_near_batch_boundaries(
    num_reqs, expected
):
    selected = backend_module._select_num_splits(
        0,
        num_reqs=num_reqs,
        num_kv_heads=2,
        logical_blocks=64,
    )

    assert selected == expected
    assert selected in (1, 2, 4, 8, 16)


@pytest.mark.parametrize(
    ("logical_blocks", "expected"),
    [(1, 1), (2, 2), (3, 2), (6, 4), (11, 8), (12, 8), (16, 8)],
)
def test_auto_split_policy_caps_to_power_of_two_context_boundary(
    logical_blocks, expected
):
    assert (
        backend_module._select_num_splits(
            0,
            num_reqs=8,
            num_kv_heads=2,
            logical_blocks=logical_blocks,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("requested", "logical_blocks", "expected"),
    [(2, 1, 1), (4, 3, 2), (8, 6, 4), (16, 11, 8), (16, 12, 8)],
)
def test_explicit_split_policy_caps_to_power_of_two_context_boundary(
    requested, logical_blocks, expected
):
    assert (
        backend_module._select_num_splits(
            requested,
            num_reqs=8,
            num_kv_heads=2,
            logical_blocks=logical_blocks,
        )
        == expected
    )


def test_eligible_forward_passes_original_vllm_tensors_to_unchecked_launcher(
    monkeypatch,
):
    impl = _impl_without_vllm_initialization()
    monkeypatch.setattr(impl, "_supports_flashdec_decode", lambda *args: True)

    def unexpected_update(*args):
        pytest.fail("eligible fused decode must not launch a separate KV update")

    monkeypatch.setattr(impl, "do_kv_cache_update", unexpected_update)
    recorded = {}

    def record_launcher(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return args[6]

    monkeypatch.setattr(
        backend_module,
        "_vllm_paged_decode_attention_into",
        record_launcher,
    )
    query = torch.empty((4, 16, 128), dtype=torch.float16)
    key = torch.empty((4, 2, 128), dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((5, 2, 16, 2, 128), dtype=torch.float16)
    metadata = _metadata()
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)

    returned = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )

    assert returned is output
    args = recorded["args"]
    assert args[0] is query
    assert args[1] is key
    assert args[2] is value
    assert args[3] is kv_cache
    assert args[4] is metadata.block_table
    assert args[5] is metadata.seq_lens
    assert args[6] is output
    assert args[7] is metadata.slot_mapping
    assert args[8] is metadata.softmax_segm_output
    assert args[9] is metadata.softmax_segm_max
    assert args[10] is metadata.softmax_segm_expsum
    assert recorded["kwargs"] == {
        "num_reqs": 3,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "block_size": 16,
        "sm_scale": 128**-0.5,
        "num_splits": 16,
    }


def test_unsupported_forward_keeps_single_update_then_native_fallback(monkeypatch):
    impl = _impl_without_vllm_initialization()
    monkeypatch.setattr(impl, "_supports_flashdec_decode", lambda *args: False)
    calls = []

    def record_update(*args):
        calls.append(("update", args))

    marker = object()

    def record_fallback(self, *args):
        calls.append(("fallback", args))
        return marker

    monkeypatch.setattr(impl, "do_kv_cache_update", record_update)
    monkeypatch.setattr(TritonAttentionImpl, "forward", record_fallback)
    values = [object() for _ in range(5)]
    query, key, value, kv_cache, output = values
    metadata = SimpleNamespace(slot_mapping=object())
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)

    returned = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )

    assert returned is marker
    assert calls == [
        (
            "update",
            (layer, key, value, kv_cache, metadata.slot_mapping),
        ),
        (
            "fallback",
            (layer, query, key, value, kv_cache, metadata, output, None, None),
        ),
    ]


@pytest.mark.parametrize(
    ("layer", "metadata"),
    [
        (SimpleNamespace(kv_sharing_target_layer_name="shared.0"), _metadata()),
        (SimpleNamespace(kv_sharing_target_layer_name=None), None),
    ],
)
def test_non_owning_or_profiling_fallback_never_updates_kv(
    monkeypatch, layer, metadata
):
    impl = _impl_without_vllm_initialization()

    def unexpected_update(*args):
        pytest.fail("this fallback does not own a KV update")

    monkeypatch.setattr(impl, "do_kv_cache_update", unexpected_update)
    marker = object()
    monkeypatch.setattr(
        TritonAttentionImpl,
        "forward",
        lambda self, *args: marker,
    )

    returned = impl.forward(
        layer,
        object(),
        object(),
        object(),
        object(),
        metadata,
        object(),
    )

    assert returned is marker
