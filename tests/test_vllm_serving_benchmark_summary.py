import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_vllm_serving_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("vllm_serving_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_fixture(
    path,
    *,
    tpot_ratio=0.995,
    throughput_ratio=1.005,
    p90_tpot_ratio=1.00,
    ttft_ratio=1.00,
    trials=(1, 2, 3),
):
    common = {
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
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.78,
        "compilation_mode": "default_inductor_cudagraph",
        "flashdec_num_splits": "auto",
        "num_prompts": 128,
        "num_warmups": 8,
        "input_len": 4096,
        "output_len": 128,
        "max_concurrency": 8,
        "request_rate": "inf",
        "prefix_caching": False,
        "case": "qwen_c8_i4096_o128",
    }
    rows = []
    for trial in trials:
        for backend in MODULE.BACKENDS:
            custom = backend == "flashdec"
            rows.append(
                {
                    **common,
                    "backend": backend,
                    "trial": trial,
                    "completed": 128,
                    "failed": 0,
                    "median_ttft_ms": 1000.0 * (ttft_ratio if custom else 1.0),
                    "p90_ttft_ms": 1200.0,
                    "median_tpot_ms": 30.0 * (tpot_ratio if custom else 1.0),
                    "p90_tpot_ms": 35.0 * (p90_tpot_ratio if custom else 1.0),
                    "median_itl_ms": 12.0,
                    "p90_itl_ms": 15.0,
                    "median_e2el_ms": 5000.0,
                    "p90_e2el_ms": 6000.0,
                    "request_throughput": 1.5,
                    "output_throughput": (
                        200.0 * throughput_ratio if custom else 200.0
                    ),
                    "total_token_throughput": 6500.0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summary_accepts_tpot_win_and_guardrails(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)

    text = MODULE.summarize(source, output)

    assert "Overall external-serving gate: **PASS**" in text
    assert "Median TPOT ratio <= 0.998x: PASS" in text
    assert output.read_text(encoding="utf-8") == text


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tpot_ratio": 0.999}, "Median TPOT ratio <= 0.998x: FAIL"),
        (
            {"throughput_ratio": 1.001},
            "Output-throughput ratio >= 1.002x: FAIL",
        ),
        ({"p90_tpot_ratio": 1.03}, "p90 TPOT ratio <= 1.02x: FAIL"),
        ({"ttft_ratio": 1.06}, "Median TTFT ratio <= 1.05x: FAIL"),
    ],
)
def test_summary_rejects_failed_gate(tmp_path, kwargs, expected):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, **kwargs)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert expected in output.read_text(encoding="utf-8")


def test_summary_rejects_unstable_tpot_ratio(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["backend"] == "flashdec" and row["trial"] == "3":
            row["median_tpot_ms"] = "28.5"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="performance gate failed"):
        MODULE.summarize(source, output)

    assert "TPOT paired-ratio spread <= 0.02: FAIL" in output.read_text(
        encoding="utf-8"
    )


def test_summary_requires_three_paired_trials(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "summary.md"
    _write_fixture(source, trials=(1, 2))

    with pytest.raises(ValueError, match="at least 3"):
        MODULE.summarize(source, output)
