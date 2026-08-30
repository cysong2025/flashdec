import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_vllm_model_correctness.py"
)
SPEC = importlib.util.spec_from_file_location("vllm_model_correctness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _payload(backend, token_ids=(1, 2, 3)):
    return {
        "schema_version": 1,
        "backend": backend,
        "git_commit": "abc123",
        "git_worktree_clean": True,
        "device": "RTX 5070",
        "torch_version": "2.11.0",
        "torch_cuda": "13.0",
        "vllm_version": "0.25.1",
        "model_path": "/model",
        "model_config_sha256": "config",
        "model_manifest_sha256": "manifest",
        "seed": 7,
        "max_tokens": 3,
        "prompts_sha256": "prompts",
        "outputs": [{"prompt": "hello", "token_ids": list(token_ids), "text": "x"}],
    }


def test_correctness_summary_accepts_identical_tokens(tmp_path):
    native = tmp_path / "native.json"
    flashdec = tmp_path / "flashdec.json"
    output = tmp_path / "summary.md"
    native.write_text(json.dumps(_payload("TRITON_ATTN")), encoding="utf-8")
    flashdec.write_text(json.dumps(_payload("CUSTOM")), encoding="utf-8")

    text = MODULE.summarize(native, flashdec, output)

    assert "equal: 1/1" in text
    assert "**PASS**" in text


def test_correctness_summary_rejects_token_mismatch(tmp_path):
    native = tmp_path / "native.json"
    flashdec = tmp_path / "flashdec.json"
    output = tmp_path / "summary.md"
    native.write_text(json.dumps(_payload("TRITON_ATTN")), encoding="utf-8")
    flashdec.write_text(json.dumps(_payload("CUSTOM", (1, 4, 3))), encoding="utf-8")

    with pytest.raises(ValueError, match="token mismatch"):
        MODULE.summarize(native, flashdec, output)

    assert "**FAIL**" in output.read_text(encoding="utf-8")
