import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "benchmarks" / "summarize_vllm_model_latency.py"
)
SPEC = importlib.util.spec_from_file_location("vllm_model_latency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(
    path, target_ratio=0.96, guardrail_ratio=1.01, trials=(1, 2, 3)
):
    fields = [
        "schema_version",
        "git_commit",
        "git_worktree_clean",
        "device",
        "torch_version",
        "torch_cuda",
        "triton_version",
        "vllm_version",
        "model_id",
        "model_path",
        "model_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "compilation_mode",
        "flashdec_num_splits",
        "warmup_iters",
        "num_iters",
        "case",
        "backend",
        "trial",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "output_tokens_per_s",
    ]
    cases = {
        "qwen_b8_i128_o128": guardrail_ratio,
        "qwen_b8_i2048_o128": target_ratio,
    }
    rows = []
    for case, ratio in cases.items():
        for trial in trials:
            for backend, p50 in (
                ("vllm_triton_attn", 1000.0),
                ("flashdec", 1000.0 * ratio),
            ):
                rows.append(
                    {
                        "schema_version": 1,
                        "git_commit": "abc123",
                        "git_worktree_clean": True,
                        "device": "RTX 5070",
                        "torch_version": "2.11.0",
                        "torch_cuda": "13.0",
                        "triton_version": "3.6.0",
                        "vllm_version": "0.25.1",
                        "model_id": "Qwen2.5-3B-Instruct",
                        "model_path": "/model",
                        "model_config_sha256": "config",
                        "model_manifest_sha256": "manifest",
                        "dtype": "bfloat16",
                        "kv_cache_dtype": "bfloat16",
                        "max_model_len": 4096,
                        "max_num_seqs": 8,
                        "max_num_batched_tokens": 2048,
                        "gpu_memory_utilization": 0.78,
                        "compilation_mode": "default_inductor_cudagraph",
                        "flashdec_num_splits": "auto",
                        "warmup_iters": 3,
                        "num_iters": 5,
                        "case": case,
                        "backend": backend,
                        "trial": trial,
                        "avg_latency_ms": p50,
                        "p50_latency_ms": p50,
                        "p90_latency_ms": p50,
                        "output_tokens_per_s": 1000.0,
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_accepts_target_win_and_guardrail(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)

    text = MODULE.summarize(source, output)

    assert "Overall external-model gate: **PASS**" in text
    assert "4.00%" in text
    assert output.read_text(encoding="utf-8") == text


def test_summary_rejects_missing_target_win(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, target_ratio=0.996)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "target <= 0.995x: FAIL" in output.read_text(encoding="utf-8")


def test_summary_rejects_short_context_regression(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, guardrail_ratio=1.04)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "guardrail <= 1.03x: FAIL" in output.read_text(encoding="utf-8")


def test_summary_rejects_unstable_ratio(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if (
            row["case"] == "qwen_b8_i2048_o128"
            and row["backend"] == "flashdec"
            and row["trial"] == "3"
        ):
            row["p50_latency_ms"] = "850.0"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "paired-ratio spread <= 0.03: FAIL" in output.read_text(
        encoding="utf-8"
    )


def test_summary_requires_three_paired_process_trials(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, trials=(1, 2))

    with pytest.raises(ValueError, match="at least 3"):
        MODULE.summarize(source, output)
