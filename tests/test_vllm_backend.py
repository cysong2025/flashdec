import json
import stat
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
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def test_vllm_backend_identity_and_cache_contract():
    assert FlashDecAttentionBackend.get_name() == "CUSTOM"
    assert FlashDecAttentionBackend.get_impl_cls() is FlashDecAttentionImpl
    assert FlashDecAttentionBackend.forward_includes_kv_cache_update is False
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
    impl.attn_type = AttentionType.DECODER
    impl.need_to_return_lse_for_decode = False
    impl.kv_cache_dtype = "auto"
    impl.alibi_slopes = None
    impl.sinks = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0
    return impl


def _metadata(num_reqs=3, max_seq_len=1024, workspace_capacity=64):
    return SimpleNamespace(
        query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32),
        max_seq_len=max_seq_len,
        max_query_len=1,
        use_cascade=False,
        causal=True,
        block_table=torch.empty((num_reqs, 64), dtype=torch.int32),
        seq_lens=torch.empty(num_reqs, dtype=torch.int32),
        slot_mapping=torch.empty(num_reqs, dtype=torch.int64),
        softmax_segm_output=SimpleNamespace(
            shape=(workspace_capacity, 16, 16, 128)
        ),
        softmax_segm_max=object(),
        softmax_segm_expsum=object(),
    )


def _attestation_binding(path):
    return {
        "path": str(path.resolve()),
        "nonce": "a" * 64,
        "case": "qwen_b8_i512_o2",
        "trial": 1,
        "dataset_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }


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


@pytest.mark.parametrize("max_seq_len", [512, 1024])
def test_eligible_context_boundary_passes_original_tensors_and_legal_split(
    monkeypatch, max_seq_len
):
    impl = _impl_without_vllm_initialization()

    def unexpected_update(*args):
        pytest.fail("eligible fused decode must not launch a separate KV update")

    monkeypatch.setattr(impl, "do_kv_cache_update", unexpected_update)
    monkeypatch.setattr(
        TritonAttentionImpl,
        "forward",
        lambda self, *args: pytest.fail(
            "eligible multi-split decode must not use native fallback"
        ),
    )
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
    metadata = _metadata(max_seq_len=max_seq_len)
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


def test_successful_multi_split_writes_one_canonical_attestation(
    monkeypatch, tmp_path
):
    impl = _impl_without_vllm_initialization()
    marker_path = tmp_path / "split.json"
    impl._split_attestation_binding = _attestation_binding(marker_path)
    monkeypatch.setattr(
        TritonAttentionImpl,
        "forward",
        lambda self, *args: pytest.fail("eligible split must not fall back"),
    )
    launches = []

    def record_launcher(*args, **kwargs):
        launches.append((args, kwargs))
        return args[6]

    monkeypatch.setattr(
        backend_module, "_vllm_paged_decode_attention_into", record_launcher
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    query = torch.empty((8, 16, 128), dtype=torch.bfloat16)
    key = torch.empty((8, 2, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((257, 2, 16, 2, 128), dtype=torch.bfloat16)
    metadata = _metadata(num_reqs=8, max_seq_len=8192)
    layer = SimpleNamespace(kv_sharing_target_layer_name=None)

    assert impl.forward(
        layer, query, key, value, kv_cache, metadata, output
    ) is output
    assert len(launches) == 1
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_path.read_bytes() == backend_module.canonical_attestation_bytes(
        payload
    )
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600
    assert payload == {
        "schema_version": 1,
        "nonce": "a" * 64,
        "engine_pid": backend_module.os.getpid(),
        "backend": "CUSTOM",
        "case": "qwen_b8_i512_o2",
        "trial": 1,
        "dataset_sha256": "b" * 64,
        "git_commit": "c" * 40,
        "max_seq_len": 8192,
        "logical_blocks": 512,
        "num_reqs": 8,
        "num_splits": 8,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
        "block_size": 16,
        "query_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "cuda_graph_capture": True,
    }
    assert impl._split_attestation_binding is None


def test_single_split_fallback_does_not_write_attestation(monkeypatch, tmp_path):
    impl = _impl_without_vllm_initialization()
    impl._requested_splits = 1
    marker_path = tmp_path / "split.json"
    impl._split_attestation_binding = _attestation_binding(marker_path)
    fallback_result = object()
    monkeypatch.setattr(
        TritonAttentionImpl,
        "forward",
        lambda self, *args: fallback_result,
    )
    query = torch.empty((8, 16, 128), dtype=torch.bfloat16)
    key = torch.empty((8, 2, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((65, 2, 16, 2, 128), dtype=torch.bfloat16)

    assert impl.forward(
        SimpleNamespace(kv_sharing_target_layer_name=None),
        query,
        key,
        value,
        kv_cache,
        _metadata(num_reqs=8),
        output,
    ) is fallback_result
    assert not marker_path.exists()


def test_failed_split_launcher_does_not_write_attestation(monkeypatch, tmp_path):
    impl = _impl_without_vllm_initialization()
    marker_path = tmp_path / "split.json"
    impl._split_attestation_binding = _attestation_binding(marker_path)
    monkeypatch.setattr(
        backend_module,
        "_vllm_paged_decode_attention_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    query = torch.empty((8, 16, 128), dtype=torch.bfloat16)
    key = torch.empty((8, 2, 128), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((65, 2, 16, 2, 128), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="launch failed"):
        impl.forward(
            SimpleNamespace(kv_sharing_target_layer_name=None),
            query,
            key,
            value,
            kv_cache,
            _metadata(num_reqs=8),
            output,
        )
    assert not marker_path.exists()


@pytest.mark.parametrize(
    ("requested_splits", "num_reqs", "workspace_capacity"),
    [
        (1, 3, 64),
        (0, 64, 64),
        (0, 3, 2),
    ],
    ids=("explicit-one", "auto-one-high-batch", "workspace-forces-one"),
)
def test_final_single_split_relies_on_vllm_update_then_uses_native_fallback(
    monkeypatch, requested_splits, num_reqs, workspace_capacity
):
    impl = _impl_without_vllm_initialization()
    impl._requested_splits = requested_splits

    def unexpected_update(*args):
        pytest.fail("vLLM must own the separate KV update for this backend")

    marker = object()
    fallback_args = []

    def record_fallback(self, *args):
        fallback_args.append(args)
        return marker

    def unexpected_launcher(*args, **kwargs):
        pytest.fail("single-split decode must not enter the FlashDec kernel")

    monkeypatch.setattr(impl, "do_kv_cache_update", unexpected_update)
    monkeypatch.setattr(TritonAttentionImpl, "forward", record_fallback)
    monkeypatch.setattr(
        backend_module,
        "_vllm_paged_decode_attention_into",
        unexpected_launcher,
    )
    query = torch.empty((num_reqs, 16, 128), dtype=torch.float16)
    key = torch.empty((num_reqs, 2, 128), dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((65, 2, 16, 2, 128), dtype=torch.float16)
    metadata = _metadata(
        num_reqs=num_reqs,
        max_seq_len=1024,
        workspace_capacity=workspace_capacity,
    )
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
    assert fallback_args == [
        (layer, query, key, value, kv_cache, metadata, output, None, None),
    ]


def test_context_511_relies_on_vllm_update_then_uses_native_fallback(monkeypatch):
    impl = _impl_without_vllm_initialization()

    def unexpected_update(*args):
        pytest.fail("vLLM must own the separate KV update for this backend")

    marker = object()
    fallback_args = []

    def record_fallback(self, *args):
        fallback_args.append(args)
        return marker

    monkeypatch.setattr(impl, "do_kv_cache_update", unexpected_update)
    monkeypatch.setattr(TritonAttentionImpl, "forward", record_fallback)
    query = torch.empty((4, 16, 128), dtype=torch.float16)
    key = torch.empty((4, 2, 128), dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    kv_cache = torch.empty((5, 2, 16, 2, 128), dtype=torch.float16)
    metadata = _metadata(max_seq_len=511)
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
    assert fallback_args == [
        (layer, query, key, value, kv_cache, metadata, output, None, None),
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
