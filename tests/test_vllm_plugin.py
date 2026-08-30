import importlib
import sys
import types


def test_vllm_plugin_module_is_import_safe_without_vllm():
    module = importlib.import_module("flashdec.vllm_plugin")

    assert callable(module.register)


def test_vllm_plugin_registers_custom_backend(monkeypatch):
    calls = []
    registry = types.ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = types.SimpleNamespace(CUSTOM="custom")

    def fake_register_backend(backend, class_path):
        calls.append((backend, class_path))

    registry.register_backend = fake_register_backend
    package_names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    ]
    for name in package_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    from flashdec.vllm_plugin import register

    register()

    assert calls == [
        ("custom", "flashdec.vllm_backend.FlashDecAttentionBackend")
    ]
