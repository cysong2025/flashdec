import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "run_vllm_attention_microbench.py"
)


def _load_runner(monkeypatch):
    torch = types.ModuleType("torch")
    triton = types.ModuleType("triton")
    flashdec_backend = types.ModuleType("flashdec.vllm_backend")
    flashdec_backend.FlashDecAttentionImpl = type("FlashDecAttentionImpl", (), {})

    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.25.1"
    vllm_v1 = types.ModuleType("vllm.v1")
    attention = types.ModuleType("vllm.v1.attention")
    backends = types.ModuleType("vllm.v1.attention.backends")
    triton_attn = types.ModuleType("vllm.v1.attention.backends.triton_attn")
    triton_attn.TritonAttentionImpl = type("TritonAttentionImpl", (), {})
    triton_attn.TritonAttentionMetadata = type("TritonAttentionMetadata", (), {})

    for name, module in {
        "torch": torch,
        "triton": triton,
        "flashdec.vllm_backend": flashdec_backend,
        "vllm": vllm,
        "vllm.v1": vllm_v1,
        "vllm.v1.attention": attention,
        "vllm.v1.attention.backends": backends,
        "vllm.v1.attention.backends.triton_attn": triton_attn,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "vllm_attention_microbench_runner_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecordingImpl:
    def __init__(self):
        self.calls = []
        self.result = object()

    def do_kv_cache_update(self, *args):
        self.calls.append(("update", args))

    def forward(self, *args):
        self.calls.append(("forward", args))
        return self.result


def test_native_timed_operation_updates_kv_before_forward(monkeypatch):
    module = _load_runner(monkeypatch)
    impl = _RecordingImpl()
    values = {
        name: object()
        for name in (
            "layer",
            "query",
            "key",
            "value",
            "kv_cache",
            "slot_mapping",
            "output",
        )
    }
    metadata = types.SimpleNamespace(slot_mapping=values["slot_mapping"])

    result = module._run_native_once(
        impl,
        values["layer"],
        values["query"],
        values["key"],
        values["value"],
        values["kv_cache"],
        metadata,
        values["output"],
    )

    assert result is impl.result
    assert impl.calls == [
        (
            "update",
            (
                values["layer"],
                values["key"],
                values["value"],
                values["kv_cache"],
                values["slot_mapping"],
            ),
        ),
        (
            "forward",
            (
                values["layer"],
                values["query"],
                values["key"],
                values["value"],
                values["kv_cache"],
                metadata,
                values["output"],
            ),
        ),
    ]


def test_flashdec_timed_operation_uses_only_fused_forward(monkeypatch):
    module = _load_runner(monkeypatch)
    impl = _RecordingImpl()
    args = [object() for _ in range(7)]

    result = module._run_flashdec_once(impl, *args)

    assert result is impl.result
    assert impl.calls == [("forward", tuple(args))]
    assert module.TIMED_OPERATIONS == {
        "vllm_triton_attn": "do_kv_cache_update_then_forward",
        "flashdec": "fused_kv_append_forward",
    }


def test_append_slots_point_to_each_requests_current_token(monkeypatch):
    module = _load_runner(monkeypatch)

    assert module._append_slot_indices(module.Case("b1", 1, 128)) == (127,)
    assert module._append_slot_indices(module.Case("b4", 4, 1024)) == (
        1023,
        2047,
        3071,
        4095,
    )
