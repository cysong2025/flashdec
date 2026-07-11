from benchmarks.profile_paged_decode import (
    CASES,
    _dtype_names,
    _num_stages_label,
    _parse_num_stages,
    _selected_cases,
    _write_summary,
)


def test_selected_cases_all_includes_large_batch():
    assert _selected_cases("all") == list(CASES.values())
    assert _selected_cases("large_batch") == [CASES["large_batch"]]


def test_dtype_names_supports_combined_run():
    assert _dtype_names("float16") == ["float16"]
    assert _dtype_names("bfloat16") == ["bfloat16"]
    assert _dtype_names("both") == ["float16", "bfloat16"]


def test_num_stages_preserves_implicit_default_label():
    assert _parse_num_stages("default") is None
    assert _parse_num_stages("3") == 3
    assert _num_stages_label(None) == "default"
    assert _num_stages_label(3) == "3"


def test_write_summary_records_final_config_and_environment(tmp_path):
    output = tmp_path / "summary.md"
    row = {
        "case": "medium_b16_ctx1024",
        "shape": "16x32x8x128x1024",
        "impl": "triton",
        "dtype": "float16",
        "kv_layout": "token_major",
        "block_size": 32,
        "num_warps": 2,
        "num_stages": "default",
        "p50_ms": "0.100000",
        "p90_ms": "0.110000",
        "mean_ms": "0.105000",
        "effective_total_gbps_p50": "1200.0000",
        "device": "NVIDIA GeForce RTX 5070",
        "torch": "2.11.0+cu128",
        "cuda": "12.8",
        "profile": "benchmarks/profiles/example.txt",
    }

    _write_summary(output, [row])

    text = output.read_text()
    assert "16x32x8x128x1024" in text
    assert "token_major" in text
    assert "| 32 | 2 | default |" in text
    assert "NVIDIA GeForce RTX 5070" in text
    assert "2.11.0+cu128" in text
    assert "12.8" in text
