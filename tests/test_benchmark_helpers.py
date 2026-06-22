from flashdec.benchmark import percentile, summarize_latencies


def test_percentile_nearest_rank():
    values = [4.0, 1.0, 3.0, 2.0]

    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 4.0


def test_summarize_latencies():
    result = summarize_latencies(
        "case",
        [1.0, 2.0, 3.0],
        metadata={"shape": "3"},
    )

    assert result.name == "case"
    assert result.mean_ms == 2.0
    assert result.p50_ms == 2.0
    assert result.p90_ms == 3.0
    assert result.metadata == {"shape": "3"}
    assert result.as_row()["shape"] == "3"

