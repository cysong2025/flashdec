"""Pure-Python helper coverage for complete DecodeEngine profiling."""

from types import SimpleNamespace

from benchmarks.profile_decode_engine import (
    PROFILE_RANGE_APPEND,
    PROFILE_RANGE_DECODE,
    PROFILE_RANGE_ENGINE_STEP,
    PROFILE_RANGE_PREFLIGHT,
    _dtype_names,
    _device_event_count,
    _selected_workloads,
    _stage_rows,
    _write_summary,
)
from benchmarks.run_decode_engine_workload import WORKLOADS


def test_profile_decode_engine_selects_workloads_and_dtypes():
    assert _dtype_names("float16") == ["float16"]
    assert _dtype_names("both") == ["float16", "bfloat16"]
    assert _selected_workloads("mixed_steady") == [WORKLOADS["mixed_steady"]]
    assert _selected_workloads("all") == list(WORKLOADS.values())


def test_stage_rows_supports_device_time_and_missing_ranges():
    events = [
        SimpleNamespace(
            key=PROFILE_RANGE_ENGINE_STEP,
            count=3,
            cpu_time_total=3_000.0,
            self_cpu_time_total=300.0,
            device_time_total=2_400.0,
            self_device_time_total=100.0,
        ),
        SimpleNamespace(
            key=PROFILE_RANGE_APPEND,
            count=3,
            cpu_time_total=1_000.0,
            self_cpu_time_total=200.0,
            device_time_total=800.0,
            self_device_time_total=800.0,
        ),
    ]

    rows = {row["range"]: row for row in _stage_rows(events)}
    assert rows[PROFILE_RANGE_ENGINE_STEP] == {
        "range": PROFILE_RANGE_ENGINE_STEP,
        "count": 3,
        "cpu_total_ms": 3.0,
        "cpu_self_ms": 0.3,
        "device_total_ms": 2.4,
        "device_self_ms": 0.1,
    }
    assert rows[PROFILE_RANGE_APPEND]["device_total_ms"] == 0.8
    assert rows[PROFILE_RANGE_PREFLIGHT]["count"] == 0
    assert rows[PROFILE_RANGE_DECODE]["device_total_ms"] == 0.0


def test_device_event_count_uses_cuda_device_records_only():
    events = [
        SimpleNamespace(device_type="DeviceType.CPU", count=10),
        SimpleNamespace(device_type="DeviceType.CUDA", count=7),
        SimpleNamespace(device_type="CUDA", count=3),
    ]
    assert _device_event_count(events) == 10


def test_write_decode_engine_profile_summary_records_stage_boundaries(tmp_path):
    output = tmp_path / "summary.md"
    _write_summary(
        output,
        [
            {
                "workload": "mixed_steady",
                "dtype": "float16",
                "append_backend": "fused_cuda",
                "steps": 32,
                "successful_steps": 32,
                "backpressure_steps": 0,
                "cuda_event_count": 96,
                "p50_ms": 1.25,
                "p99_ms": 1.75,
                "engine_cpu_ms": 10.0,
                "engine_device_ms": 30.0,
                "append_device_ms": 8.0,
                "decode_device_ms": 22.0,
                "profile": "profile.txt",
                "trace": "trace.json",
            }
        ],
    )

    text = output.read_text()
    assert "mixed_steady" in text
    assert "fused_cuda" in text
    assert "1.250000" in text
    assert "| 96 |" in text
    assert "8.000000" in text
    assert "22.000000" in text
    assert "profile.txt" in text
    assert "trace.json" in text
    assert "must not be added blindly" in text
