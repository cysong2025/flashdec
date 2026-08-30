import csv
import hashlib
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
        "started_at",
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
        "tokenizer_config_sha256",
        "model_manifest_sha256",
        "dtype",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "compilation_mode",
        "vllm_cache_root",
        "flashdec_num_splits",
        "warmup_iters",
        "num_iters",
        "dataset_seed",
        "dataset_generation_protocol",
        "sampling_seed",
        "prompt_format",
        "skip_tokenizer_init",
        "sampling_n",
        "sampling_temperature",
        "sampling_min_tokens",
        "sampling_max_tokens",
        "sampling_ignore_eos",
        "sampling_detokenize",
        "timing_scope",
        "vllm_engine_multiprocessing",
        "accuracy_prefix_len",
        "case",
        "batch_size",
        "input_len",
        "output_len",
        "backend",
        "backend_arg",
        "trial",
        "run_order",
        "dataset_path",
        "dataset_sha256",
        "output_token_ids_sha256",
        "cross_backend_exact_sequences",
        "cross_backend_common_prefix_tokens",
        "cross_backend_min_common_prefix_tokens",
        "cross_backend_generated_tokens",
        "cross_backend_full_hash_equal",
        "cross_backend_accuracy_prefix_pass",
        "avg_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "output_tokens_per_s",
        "raw_json",
        "log",
        "command",
    ]
    cases = {
        "qwen_b8_i128_o128": guardrail_ratio,
        "qwen_b8_i2048_o128": target_ratio,
    }
    rows = []
    for case, ratio in cases.items():
        dataset_sha256 = hashlib.sha256(case.encode("utf-8")).hexdigest()
        for trial in trials:
            for backend, p50 in (
                ("vllm_triton_attn", 1000.0),
                ("flashdec", 1000.0 * ratio),
            ):
                rows.append(
                    {
                        "schema_version": 3,
                        "started_at": "2026-08-30T00:00:00+08:00",
                        "git_commit": "abc1234",
                        "git_worktree_clean": True,
                        "device": "RTX 5070",
                        "torch_version": "2.11.0",
                        "torch_cuda": "13.0",
                        "triton_version": "3.6.0",
                        "vllm_version": "0.25.1",
                        "model_id": "Qwen2.5-3B-Instruct",
                        "model_path": "/model",
                        "model_config_sha256": "a" * 64,
                        "tokenizer_config_sha256": "b" * 64,
                        "model_manifest_sha256": "c" * 64,
                        "dtype": "bfloat16",
                        "kv_cache_dtype": "bfloat16",
                        "max_model_len": 4096,
                        "max_num_seqs": 8,
                        "max_num_batched_tokens": 2048,
                        "gpu_memory_utilization": 0.78,
                        "compilation_mode": "default_inductor_cudagraph",
                        "vllm_cache_root": "/tmp/vllm-cache/abc1234",
                        "flashdec_num_splits": "auto",
                        "warmup_iters": 3,
                        "num_iters": 5,
                        "dataset_seed": 20260830,
                        "dataset_generation_protocol": (
                            "sha256-indexed-u64be-mod-model-tokenizer-"
                            "nonspecial-v2"
                        ),
                        "sampling_seed": 20260830,
                        "prompt_format": "token_ids",
                        "skip_tokenizer_init": True,
                        "sampling_n": 1,
                        "sampling_temperature": 0.0,
                        "sampling_min_tokens": 128,
                        "sampling_max_tokens": 128,
                        "sampling_ignore_eos": True,
                        "sampling_detokenize": False,
                        "timing_scope": (
                            "wall-clock blocking LLM.generate call after "
                            "full-length warmup; model load, engine startup, "
                            "JIT/graph capture, and result hashing excluded"
                        ),
                        "vllm_engine_multiprocessing": True,
                        "accuracy_prefix_len": 2,
                        "case": case,
                        "batch_size": 8,
                        "input_len": 128 if "i128" in case else 2048,
                        "output_len": 128,
                        "backend": backend,
                        "backend_arg": {
                            "vllm_triton_attn": "TRITON_ATTN",
                            "flashdec": "CUSTOM",
                        }[backend],
                        "trial": trial,
                        "run_order": (
                            1
                            if (trial % 2 == 1) == (backend == "vllm_triton_attn")
                            else 2
                        ),
                        "dataset_path": f"/datasets/{case}.json",
                        "dataset_sha256": dataset_sha256,
                        "output_token_ids_sha256": hashlib.sha256(
                            f"output:{case}".encode("utf-8")
                        ).hexdigest(),
                        "cross_backend_exact_sequences": 8,
                        "cross_backend_common_prefix_tokens": 1024,
                        "cross_backend_min_common_prefix_tokens": 128,
                        "cross_backend_generated_tokens": 1024,
                        "cross_backend_full_hash_equal": True,
                        "cross_backend_accuracy_prefix_pass": True,
                        "avg_latency_ms": p50,
                        "p50_latency_ms": p50,
                        "p90_latency_ms": p50,
                        "output_tokens_per_s": 1024.0 * 1000.0 / p50,
                        "raw_json": f"/raw/{case}-{trial}-{backend}.json",
                        "log": f"/raw/{case}-{trial}-{backend}.log",
                        "command": "python worker.py",
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
    assert "Commit-scoped vLLM cache: `/tmp/vllm-cache/abc1234`" in text
    assert "fixed token IDs; seed `20260830`" in text
    assert hashlib.sha256(b"qwen_b8_i2048_o128").hexdigest() in text
    assert output.read_text(encoding="utf-8") == text


def test_summary_rejects_cache_root_from_another_commit(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        row["vllm_cache_root"] = "/tmp/vllm-cache/def5678"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="must contain the Git commit"):
        MODULE.summarize(source, output)


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

    assert "guardrail <= 1.02x: FAIL" in output.read_text(encoding="utf-8")


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

    with pytest.raises(ValueError, match="frozen paired trials"):
        MODULE.summarize(source, output)


def test_summary_rejects_different_dataset_for_paired_backend(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if (
            row["case"] == "qwen_b8_i2048_o128"
            and row["backend"] == "flashdec"
            and row["trial"] == "2"
        ):
            row["dataset_sha256"] = "f" * 64
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="exact same dataset"):
        MODULE.summarize(source, output)


def test_summary_accepts_late_autoregressive_output_divergence(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["case"] == "qwen_b8_i2048_o128":
            row["cross_backend_exact_sequences"] = "7"
            row["cross_backend_common_prefix_tokens"] = "1023"
            row["cross_backend_min_common_prefix_tokens"] = "127"
            row["cross_backend_full_hash_equal"] = "False"
            if row["backend"] == "flashdec":
                row["output_token_ids_sha256"] = "f" * 64
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    text = MODULE.summarize(source, output)

    assert "2 unique full-rollout SHA-256 (descriptive only)" in text


def test_summary_rejects_divergence_before_first_custom_decode_decision(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["case"] == "qwen_b8_i2048_o128":
            row["cross_backend_exact_sequences"] = "7"
            row["cross_backend_common_prefix_tokens"] = "897"
            row["cross_backend_min_common_prefix_tokens"] = "1"
            row["cross_backend_full_hash_equal"] = "False"
            row["cross_backend_accuracy_prefix_pass"] = "False"
            if row["backend"] == "flashdec":
                row["output_token_ids_sha256"] = "f" * 64
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "second token exercises custom decode): FAIL" in output.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vllm_version", "9.9.9", "vLLM 0.25.1"),
        ("warmup_iters", "0", "trial strength"),
        ("max_num_seqs", "1", "capacity/compilation"),
    ],
)
def test_summary_rejects_nonformal_environment_fields(
    tmp_path, field, value, message
):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        row[field] = value
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        MODULE.summarize(source, output)


def test_summary_rejects_wrong_case_shape(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["case"] == "qwen_b8_i2048_o128":
            row["batch_size"] = "1"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="case name/shape"):
        MODULE.summarize(source, output)


def test_summary_rejects_nonfinite_latency(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    rows[0]["avg_latency_ms"] = "nan"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="latencies must be positive"):
        MODULE.summarize(source, output)


def test_summary_rejects_backend_run_order_mismatch(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    rows[0]["run_order"] = "2"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="run order"):
        MODULE.summarize(source, output)
