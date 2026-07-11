import argparse

import pytest

from benchmarks.run_num_stages_sweep import CASES, _parse_num_stages, _selected_cases


@pytest.mark.parametrize(
    ("value", "expected"),
    [("default", None), ("1", 1), ("2", 2), ("3", 3), ("4", 4)],
)
def test_parse_num_stages(value, expected):
    assert _parse_num_stages(value) == expected


@pytest.mark.parametrize("value", ["0", "5", "-1", "invalid"])
def test_parse_num_stages_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_num_stages(value)


def test_selected_cases_preserves_requested_order():
    assert _selected_cases(["large_batch", "medium"]) == [
        CASES["large_batch"],
        CASES["medium"],
    ]
