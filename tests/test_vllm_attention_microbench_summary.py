import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_vllm_attention_microbench.py"
)
SPEC = importlib.util.spec_from_file_location("vllm_attention_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(path, regression_ratio=1.03):
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
        "comparison_scope",
        "timed_operation",
        "cache_state_policy",
        "input_seed",
        "flashdec_num_splits",
        "dtype",
        "case",
        "backend",
        "trial",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "correctness",
    ]
    cases = {
        "qwen_b1_ctx128": regression_ratio,
        "qwen_b1_ctx1024": 1.00,
        "qwen_b4_ctx1024": 1.00,
        "qwen_b8_ctx1024": 0.75,
        "qwen_b8_ctx2048": 0.80,
    }
    rows = []
    for case, ratio in cases.items():
        for backend, p50 in (("vllm_triton_attn", 0.05), ("flashdec", 0.05 * ratio)):
            timed_operation = {
                "vllm_triton_attn": "do_kv_cache_update_then_forward",
                "flashdec": "fused_kv_append_forward",
            }[backend]
            rows.append(
                {
                    "schema_version": 2,
                    "git_commit": "abc123",
                    "git_worktree_clean": True,
                    "device": "RTX 5070",
                    "torch_version": "2.11.0",
                    "torch_cuda": "13.0",
                    "triton_version": "3.6.0",
                    "vllm_version": "0.25.1",
                    "model_id": "Qwen2.5-3B-Instruct",
                    "comparison_scope": (
                        "current_token_kv_append_plus_single_token_decode_attention"
                    ),
                    "timed_operation": timed_operation,
                    "cache_state_policy": (
                        "paired_deterministic_snapshot_reset_per_trial_"
                        "idempotent_append"
                    ),
                    "input_seed": 20260831,
                    "flashdec_num_splits": "auto",
                    "dtype": "bfloat16",
                    "case": case,
                    "backend": backend,
                    "trial": 1,
                    "p50_ms": p50,
                    "p90_ms": p50,
                    "p99_ms": p50,
                    "correctness": "PASS",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_accepts_frozen_win_and_guardrail(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)

    text = MODULE.summarize(source, output)

    assert "Overall external-kernel gate: **PASS**" in text
    assert "current-token KV-cache append plus single-token decode attention" in text
    assert "full-output and full-cache" in text
    assert output.read_text(encoding="utf-8") == text


def test_summary_rejects_guardrail_regression(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, regression_ratio=1.10)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "guardrail: FAIL" in output.read_text(encoding="utf-8")


def test_summary_rejects_unstable_paired_ratios(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    duplicate = []
    for row in rows:
        copied = dict(row)
        copied["trial"] = "2"
        if copied["case"] == "qwen_b1_ctx128" and copied["backend"] == "flashdec":
            copied["p50_ms"] = "0.04"
        duplicate.append(copied)
    with source.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writerows(duplicate)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "paired-ratio spread <= 0.15: FAIL" in output.read_text(
        encoding="utf-8"
    )


def test_summary_rejects_mislabeled_timed_operation(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    rows[0]["timed_operation"] = "forward_only"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="unexpected timed_operation"):
        MODULE.summarize(source, output)


def test_summary_rejects_mismatched_paired_seed(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    rows[0]["input_seed"] = "7"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="same input_seed"):
        MODULE.summarize(source, output)
