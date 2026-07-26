#!/usr/bin/env python3
"""Generate the auditable FlashDec research-evidence overview SVGs."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "benchmarks" / "results" / "public_results_snapshot.json"
DEFAULT_OUTPUTS = {
    "light": ROOT / "docs" / "assets" / "flashdec-results-overview-light.svg",
    "dark": ROOT / "docs" / "assets" / "flashdec-results-overview-dark.svg",
}

PALETTES = {
    "light": {
        "background": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel_alt": "#F8FAFC",
        "text": "#172033",
        "muted": "#5C6B7A",
        "faint": "#8B98A7",
        "border": "#D8E0EA",
        "grid": "#E7ECF2",
        "blue": "#1E40AF",
        "blue_soft": "#DBEAFE",
        "green": "#065F46",
        "green_soft": "#DDF7EB",
        "amber": "#9A6700",
        "amber_soft": "#FFF3C4",
        "red": "#CF222E",
        "red_soft": "#FFEBE9",
        "purple": "#8250DF",
        "purple_soft": "#EDE4FF",
        "shadow": "#1F29370F",
    },
    "dark": {
        "background": "#0D1117",
        "panel": "#161B22",
        "panel_alt": "#1C2128",
        "text": "#F0F6FC",
        "muted": "#A6B1BE",
        "faint": "#7D8590",
        "border": "#30363D",
        "grid": "#252C35",
        "blue": "#58A6FF",
        "blue_soft": "#152A45",
        "green": "#3FB950",
        "green_soft": "#173D2B",
        "amber": "#D29922",
        "amber_soft": "#473715",
        "red": "#FF7B72",
        "red_soft": "#472025",
        "purple": "#BC8CFF",
        "purple_soft": "#34234F",
        "shadow": "#00000035",
    },
}


class SVG:
    """Small deterministic SVG writer with XML escaping."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    @staticmethod
    def _attrs(attributes: dict[str, Any]) -> str:
        def xml_name(python_name: str) -> str:
            if python_name.endswith("_"):
                python_name = python_name[:-1]
            return python_name.replace("_", "-")

        return " ".join(
            f'{xml_name(name)}="{escape(_number(value) if isinstance(value, float) else str(value), quote=True)}"'
            for name, value in attributes.items()
            if value is not None
        )

    def raw(self, value: str) -> None:
        self.lines.append("  " * self.depth + value)

    def open(self, tag: str, **attributes: Any) -> None:
        rendered = self._attrs(attributes)
        self.raw(f"<{tag}{' ' if rendered else ''}{rendered}>")
        self.depth += 1

    def close(self, tag: str) -> None:
        self.depth -= 1
        self.raw(f"</{tag}>")

    def element(self, tag: str, text: str | None = None, **attributes: Any) -> None:
        rendered = self._attrs(attributes)
        prefix = f"<{tag}{' ' if rendered else ''}{rendered}"
        if text is None:
            self.raw(prefix + "/>")
        else:
            self.raw(prefix + f">{escape(text)}</{tag}>")

    def text(self, x: float, y: float, value: str, class_name: str, **attributes: Any) -> None:
        self.element(
            "text",
            value,
            x=_number(x),
            y=_number(y),
            class_=class_name,
            **attributes,
        )

    def finish(self) -> str:
        if self.depth:
            raise RuntimeError("unclosed SVG element")
        return "\n".join(self.lines) + "\n"


def _number(value: float) -> str:
    rounded = round(value, 3)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def load_snapshot(path: Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    """Load and validate the processed public evidence snapshot."""
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(data, root=ROOT)
    return data


def _markdown_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    lines = text.splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError as error:
        raise ValueError(f"canonical source is missing section {marker!r}") from error
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Parse the first GitHub-flavored Markdown table under an H2 heading."""
    lines = _markdown_section(text, heading).splitlines()
    for index in range(len(lines) - 1):
        if not lines[index].lstrip().startswith("|"):
            continue
        header = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        separator = [
            cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")
        ]
        if len(header) != len(separator) or not all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separator
        ):
            continue
        rows: list[dict[str, str]] = []
        for raw_row in lines[index + 2 :]:
            if not raw_row.lstrip().startswith("|"):
                break
            cells = [cell.strip() for cell in raw_row.strip().strip("|").split("|")]
            if len(cells) != len(header):
                raise ValueError(f"malformed table row under {heading!r}: {raw_row}")
            rows.append(dict(zip(header, cells)))
        if not rows:
            raise ValueError(f"canonical table under {heading!r} is empty")
        return rows
    raise ValueError(f"canonical source has no table under {heading!r}")


def _source_metadata(text: str) -> dict[str, Any]:
    commit = re.search(r"(?m)^- Git commit: `([0-9a-f]+)`\.$", text)
    device = re.search(r"(?m)^- Device: (.+)\.$", text)
    direct = re.search(
        r"(?m)^- Rows: (\d+); (?:expected )?trials: (\d+)\.$",
        text,
    )
    paired = re.search(r"(?m)^- Rows: (\d+); paired trials: (\d+)\.$", text)
    if commit is None or device is None or (direct is None and paired is None):
        raise ValueError("canonical source validation metadata is incomplete")
    if direct is not None:
        return {
            "commit": commit.group(1),
            "device": device.group(1),
            "rows": int(direct.group(1)),
            "trials": int(direct.group(2)),
            "paired_trials": None,
        }
    return {
        "commit": commit.group(1),
        "device": device.group(1),
        "rows": int(paired.group(1)),
        "trials": None,
        "paired_trials": int(paired.group(2)),
    }


def _ratio(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)x", value)
    if match is None:
        raise ValueError(f"invalid ratio value in canonical table: {value!r}")
    return float(match.group(1))


def _ratio_range(value: str) -> tuple[float, float, float]:
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)x \[([0-9]+(?:\.[0-9]+)?),([0-9]+(?:\.[0-9]+)?)\]",
        value,
    )
    if match is None:
        raise ValueError(f"invalid ratio range in canonical table: {value!r}")
    return tuple(float(match.group(index)) for index in (1, 2, 3))


def _validate_provenance(
    evidence_id: str,
    section: dict[str, Any],
    source_text: str,
    *,
    expected: tuple[str, int, int],
) -> dict[str, Any]:
    observed = (
        section.get("evidence_commit"),
        section.get("rows"),
        section.get("trials"),
    )
    if observed != expected:
        raise ValueError(
            f"{evidence_id} provenance must be commit/rows/trials "
            f"{'/'.join(map(str, expected))}, got {observed!r}"
        )
    metadata = _source_metadata(source_text)
    if metadata["commit"] != expected[0] or metadata["rows"] != expected[1]:
        raise ValueError(
            f"{evidence_id} snapshot provenance differs from canonical source"
        )
    if metadata["device"] != "NVIDIA GeForce RTX 5070":
        raise ValueError(f"{evidence_id} canonical source has an unexpected device")
    if metadata["trials"] is not None and metadata["trials"] != expected[2]:
        raise ValueError(f"{evidence_id} trial count differs from canonical source")
    return metadata


def validate_snapshot(data: dict[str, Any], *, root: Path = ROOT) -> None:
    """Validate the snapshot by parsing canonical tables and outcome sections."""
    if data.get("schema_version") != 2:
        raise ValueError("snapshot schema_version must be 2")

    artifact = data.get("artifact", {})
    expected_artifact = {
        "device": "NVIDIA GeForce RTX 5070",
        "provenance": "Derived from validated canonical Markdown summaries.",
        "data_class": "Curated processed snapshot; not a raw benchmark dataset.",
    }
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            raise ValueError(f"artifact.{key} must be {expected!r}")
    if "not comparable across panels" not in artifact.get("ratio_boundary", ""):
        raise ValueError("artifact.ratio_boundary must forbid cross-panel comparison")

    scheduler = data["scheduler_progress"]
    shared_prefix = data["shared_prefix_capacity"]
    optimization = data["optimization_outcomes"]
    outcomes = optimization["entries"]
    integrated = data["integrated_lifecycle"]
    external_baseline = data["flashinfer_kernel_baseline"]
    sections = [scheduler, shared_prefix, *outcomes, integrated, external_baseline]
    source_texts: dict[str, str] = {}
    for section in sections:
        source = root / section["source"]
        if not source.is_file():
            raise ValueError(f"canonical source does not exist: {section['source']}")
        source_texts[section["source"]] = source.read_text(encoding="utf-8")

    if scheduler.get("kind") != "correctness_and_progress":
        raise ValueError(
            "scheduler progress must be represented as correctness/progress evidence"
        )
    scheduler_outcomes = {
        row["id"]: (row["completed"], row["cancelled"], row["deadlock"])
        for row in scheduler["policies"]
    }
    if scheduler_outcomes != {
        "lifetime_fifo_aging": (2, 0, False),
        "cancel_on_backpressure": (1, 1, False),
        "greedy_step_only": (0, 0, True),
    }:
        raise ValueError(
            "scheduler boundary-pressure outcomes do not match canonical evidence"
        )
    scheduler_text = source_texts[scheduler["source"]]
    _validate_provenance(
        "scheduler_progress",
        scheduler,
        scheduler_text,
        expected=("16de9d4", 36, 3),
    )
    scheduler_table = _markdown_table(scheduler_text, "Cross-trial Medians")
    observed_scheduler = {
        (
            row["dtype"],
            row["policy"],
            row["completion"],
            row["cancellations"],
            row["deadlocks"],
        )
        for row in scheduler_table
        if row["case"] == "boundary_deadlock"
    }
    expected_scheduler = {
        (dtype, policy, completion, cancellations, deadlocks)
        for dtype in ("float16", "bfloat16")
        for policy, completion, cancellations, deadlocks in (
            ("lifetime_fifo_aging", "1.000", "0", "0"),
            ("cancel_on_backpressure", "0.500", "1", "0"),
            ("greedy_step_only", "0.000", "0", "1"),
        )
    }
    if observed_scheduler != expected_scheduler:
        raise ValueError("scheduler canonical boundary-pressure table changed")

    if shared_prefix.get("kind") != "kv_pool_capacity_not_process_vram":
        raise ValueError(
            "shared-prefix evidence must distinguish KV-pool capacity from process VRAM"
        )
    expected_shared_prefix = [
        (0, 64, 9, 0.0),
        (25, 52, 12, 1.5),
        (50, 36, 15, 3.5),
        (75, 20, 16, 5.5),
    ]
    snapshot_shared_prefix = [
        (
            row["hit_rate_percent"],
            row["physical_context_blocks"],
            row["admitted_requests"],
            row["saved_kv_capacity_mib"],
        )
        for row in shared_prefix["points"]
    ]
    if snapshot_shared_prefix != expected_shared_prefix:
        raise ValueError(
            "shared-prefix capacity/admission points do not match canonical evidence"
        )
    shared_prefix_text = source_texts[shared_prefix["source"]]
    _validate_provenance(
        "shared_prefix_capacity",
        shared_prefix,
        shared_prefix_text,
        expected=("fe72e27", 64, 8),
    )
    canonical_shared_prefix = set()
    for row in _markdown_table(shared_prefix_text, "Cross-trial Medians"):
        admitted, admitted_total = map(int, row["admitted"].split("/"))
        physical, logical = map(
            int,
            row["context physical/logical blocks"].split("/"),
        )
        if admitted_total != 16 or logical != 64:
            raise ValueError("shared-prefix canonical capacity denominator changed")
        canonical_shared_prefix.add(
            (
                row["dtype"],
                int(row["hit rate"].rstrip("%")),
                physical,
                admitted,
                float(row["saved KV-capacity MiB"]),
            )
        )
    expected_canonical_shared_prefix = {
        (dtype, hit_rate, physical, admitted, saved)
        for dtype in ("float16", "bfloat16")
        for hit_rate, physical, admitted, saved in expected_shared_prefix
    }
    if canonical_shared_prefix != expected_canonical_shared_prefix:
        raise ValueError("shared-prefix canonical capacity table changed")

    expected_outcomes = [
        (
            "fused_append",
            "Fused append path",
            1.2101,
            20,
            24,
            "observed",
            "stability_observation",
        ),
        (
            "trusted_transaction",
            "Trusted transaction path",
            1.7307,
            16,
            16,
            "accepted",
            "scoped_acceptance",
        ),
        (
            "persistent_metadata",
            "Persistent metadata",
            1.2493,
            13,
            16,
            "rejected_and_rolled_back",
            "pre_registered_keep_gate",
        ),
    ]
    observed_outcomes = [
        (
            row["id"],
            row["feature"],
            row["p50_ratio"],
            row["stable_groups"],
            row["total_groups"],
            row["outcome"],
            row["outcome_basis"],
        )
        for row in outcomes
    ]
    if observed_outcomes != expected_outcomes:
        raise ValueError(
            "transaction-optimization scorecard does not match canonical evidence"
        )

    technical_boundary = optimization.get("technical_boundary", "")
    if (
        "different workloads and timing boundaries" not in technical_boundary
        or "pre-registered keep gate" not in technical_boundary
    ):
        raise ValueError(
            "optimization_outcomes.technical_boundary must state the comparison boundary"
        )

    outcome_rules = {
        "fused_append": ("fa0f89a", 144, 3, "fused_faster", 20, 24, 1.2101),
        "trusted_transaction": (
            "4018449",
            160,
            5,
            "trusted_faster",
            16,
            16,
            1.7307,
        ),
        "persistent_metadata": (
            "8047a9c",
            160,
            5,
            "persistent_faster",
            13,
            16,
            1.2493,
        ),
    }
    for row in outcomes:
        commit, total_rows, trials, direction, stable, groups, p50 = outcome_rules[
            row["id"]
        ]
        source_text = source_texts[row["source"]]
        metadata = _validate_provenance(
            row["id"],
            row,
            source_text,
            expected=(commit, total_rows, trials),
        )
        case_table = _markdown_table(source_text, "Cross-trial Cases")
        if len(case_table) != groups:
            raise ValueError(f"{row['id']} canonical group count changed")
        if metadata["paired_trials"] != groups * trials:
            raise ValueError(f"{row['id']} paired-trial count changed")
        if sum(case["direction"] == direction for case in case_table) != stable:
            raise ValueError(f"{row['id']} canonical stable-group count changed")
        overall = _markdown_table(source_text, "Overall Geometric Mean")
        metric_row = next(
            (item for item in overall if item["metric"] == "complete-token p50"),
            None,
        )
        if metric_row is None:
            raise ValueError(f"{row['id']} canonical p50 metric is missing")
        ratio_column = next(key for key in metric_row if key != "metric")
        if _ratio(metric_row[ratio_column]) != p50:
            raise ValueError(f"{row['id']} canonical p50 ratio changed")
    persistent_metadata_text = source_texts[outcomes[2]["source"]]
    if "- Keep decision: `not adopted`." not in _markdown_section(
        persistent_metadata_text,
        "Pre-registered Decision Rule",
    ):
        raise ValueError("persistent-metadata decision rule changed")

    if integrated.get("kind") != "integrated_correctness_and_lifecycle" or integrated.get(
        "status"
    ) != "passed":
        raise ValueError("integrated workload must be represented as a passed lifecycle gate")
    integrated_text = source_texts[integrated["source"]]
    _validate_provenance(
        "integrated_lifecycle",
        integrated,
        integrated_text,
        expected=("6912894", 24, 3),
    )
    if len(_markdown_table(integrated_text, "Cross-trial Absolute Results")) != 8:
        raise ValueError("integrated-workload canonical matrix shape changed")
    if "24 行均通过 strict validator" not in _markdown_section(
        integrated_text,
        "Interpretation",
    ):
        raise ValueError("integrated-workload lifecycle validation changed")

    if external_baseline.get("kind") != "descriptive_external_kernel_baseline":
        raise ValueError("external baseline must be descriptive kernel-only evidence")
    if external_baseline.get("performance_gate") is not False:
        raise ValueError("external baseline must not define a performance gate")
    if "above 1 favor FlashInfer" not in external_baseline.get(
        "ratio_direction", ""
    ):
        raise ValueError("external baseline must state that ratios above 1 favor FlashInfer")
    external_baseline_text = source_texts[external_baseline["source"]]
    _validate_provenance(
        "flashinfer_kernel_baseline",
        external_baseline,
        external_baseline_text,
        expected=("d7d4feb", 72, 3),
    )
    external_baseline_table = _markdown_table(
        external_baseline_text,
        "Paired Cross-trial Results",
    )
    if len(external_baseline_table) != 16:
        raise ValueError("external-baseline matrix must contain 16 backend comparisons")
    by_backend: dict[str, list[float]] = {}
    above_one = 0
    for row in external_baseline_table:
        median, minimum, _maximum = _ratio_range(
            row["p50 ratio FlashDec/external"]
        )
        by_backend.setdefault(row["external backend"], []).append(median)
        above_one += minimum > 1.0
    canonical_external_baseline = [
        (backend["id"], backend["p50_ratio_geometric_mean"])
        for backend in external_baseline["backends"]
    ]
    observed_external_baseline = [
        (
            backend,
            round(math.exp(sum(math.log(value) for value in values) / len(values)), 4),
        )
        for backend, values in sorted(by_backend.items())
    ]
    if observed_external_baseline != sorted(canonical_external_baseline):
        raise ValueError("external-baseline canonical p50 geometric means changed")
    range_counts = (
        external_baseline.get("observed_p50_ranges_above_one"),
        external_baseline.get("total_p50_ranges"),
    )
    if range_counts != (above_one, len(external_baseline_table)) or range_counts != (
        16,
        16,
    ):
        raise ValueError("external-baseline observed p50 range counts changed")


def _style(palette: dict[str, str]) -> str:
    values = dict(palette)
    return """
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif; }
      .eyebrow { fill: %(blue)s; font-size: 17px; font-weight: 750; letter-spacing: 1.6px; }
      .title { fill: %(text)s; font-size: 42px; font-weight: 760; letter-spacing: -0.8px; }
      .subtitle { fill: %(muted)s; font-size: 19px; font-weight: 450; }
      .panel-title { fill: %(text)s; font-size: 24px; font-weight: 730; }
      .panel-subtitle { fill: %(muted)s; font-size: 16px; font-weight: 450; }
      .label { fill: %(text)s; font-size: 17px; font-weight: 650; }
      .small { fill: %(muted)s; font-size: 17px; font-weight: 500; }
      .tiny { fill: %(muted)s; font-size: 16px; font-weight: 500; }
      .value { fill: %(text)s; font-size: 20px; font-weight: 760; font-variant-numeric: tabular-nums; }
      .value-green { fill: %(green)s; font-size: 20px; font-weight: 760; font-variant-numeric: tabular-nums; }
      .value-red { fill: %(red)s; font-size: 18px; font-weight: 780; }
      .bar-label { fill: %(text)s; font-size: 15px; font-weight: 750; text-anchor: middle; }
      .footer { fill: %(muted)s; font-size: 16px; font-weight: 500; text-anchor: middle; }
      .panel { fill: %(panel)s; stroke: %(border)s; stroke-width: 1; }
      .row { fill: %(panel_alt)s; stroke: %(border)s; stroke-width: 1; }
      .grid { stroke: %(grid)s; stroke-width: 1; }
      .axis { stroke: %(faint)s; stroke-width: 1; }
      .complete { fill: %(green_soft)s; stroke: %(green)s; stroke-width: 1.5; }
      .cancelled { fill: url(#cancel-pattern); stroke: %(amber)s; stroke-width: 1.5; }
      .deadlock { fill: %(red_soft)s; stroke: %(red)s; stroke-width: 2; }
      .context-bar { fill: %(blue_soft)s; stroke: %(blue)s; stroke-width: 1.5; }
      .admission-line { fill: none; stroke: %(green)s; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
      .admission-dot { fill: %(panel)s; stroke: %(green)s; stroke-width: 3; }
      .ratio-line { stroke: %(purple)s; stroke-width: 7; stroke-linecap: round; }
      .ratio-dot { fill: %(panel)s; stroke: %(purple)s; stroke-width: 5; }
    """ % values


def _panel(svg: SVG, x: int, y: int, width: int, height: int, panel_id: str, label: str) -> None:
    svg.open("g", id=panel_id, role="group", aria_label=label)
    svg.element("rect", x=x, y=y + 7, width=width, height=height, rx=20, fill="url(#panel-shadow)")
    svg.element("rect", class_="panel", x=x, y=y, width=width, height=height, rx=20)


def _panel_end(svg: SVG) -> None:
    svg.close("g")


def _render_scheduler_progress(
    svg: SVG,
    data: dict[str, Any],
    p: dict[str, str],
) -> None:
    x, y, width, height = 64, 178, 674, 382
    _panel(
        svg,
        x,
        y,
        width,
        height,
        "scheduler-progress-panel",
        "Scheduler progress outcomes",
    )
    svg.text(x + 30, y + 43, "Scheduler progress", "panel-title")
    svg.text(x + 30, y + 69, "Boundary-pressure case · 2 requests · FP16/BF16 agree", "panel-subtitle")

    bar_x = x + 266
    segment_width = 90
    segment_gap = 7
    rows = data["policies"]
    for index, row in enumerate(rows):
        row_y = y + 112 + index * 76
        svg.text(x + 30, row_y + 18, row["label"], "label")
        if row["id"] == "lifetime_fifo_aging":
            states = ["complete", "complete"]
            status = "2/2 COMPLETE"
            status_class = "value-green"
        elif row["id"] == "cancel_on_backpressure":
            states = ["complete", "cancelled"]
            status = "1/2 COMPLETE"
            status_class = "label"
        else:
            states = ["deadlock", "deadlock"]
            status = "DEADLOCK"
            status_class = "value-red"
        for request_index, state in enumerate(states):
            segment_x = bar_x + request_index * (segment_width + segment_gap)
            svg.element(
                "rect",
                class_=state,
                x=segment_x,
                y=row_y,
                width=segment_width,
                height=30,
                rx=8,
            )
            glyph = "✓ complete" if state == "complete" else "× cancelled" if state == "cancelled" else "! blocked"
            svg.text(segment_x + segment_width / 2, row_y + 20, glyph, "bar-label")
        svg.text(x + width - 30, row_y + 20, status, status_class, text_anchor="end")

    legend_y = y + height - 32
    svg.element("circle", cx=x + 33, cy=legend_y - 4, r=5, fill=p["green"])
    svg.text(x + 45, legend_y, "completed", "tiny")
    svg.element("circle", cx=x + 132, cy=legend_y - 4, r=5, fill=p["amber"])
    svg.text(x + 144, legend_y, "cancelled", "tiny")
    svg.element("circle", cx=x + 232, cy=legend_y - 4, r=5, fill=p["red"])
    svg.text(x + 244, legend_y, "deadlock", "tiny")
    svg.text(x + width - 30, legend_y, "Correctness/progress evidence — not latency", "tiny", text_anchor="end")
    _panel_end(svg)


def _render_shared_prefix_capacity(
    svg: SVG,
    data: dict[str, Any],
    p: dict[str, str],
) -> None:
    x, y, width, height = 762, 178, 674, 382
    _panel(
        svg,
        x,
        y,
        width,
        height,
        "shared-prefix-capacity-panel",
        "Shared-prefix capacity and admission",
    )
    svg.text(x + 30, y + 43, "Shared-prefix capacity", "panel-title")
    svg.text(x + 30, y + 69, "48-block admission pool · 64 logical context blocks", "panel-subtitle")

    legend_y = y + 101
    svg.element("rect", class_="context-bar", x=x + 31, y=legend_y - 12, width=17, height=12, rx=3)
    svg.text(x + 56, legend_y, "physical context blocks", "tiny")
    svg.element("line", class_="admission-line", x1=x + 238, y1=legend_y - 6, x2=x + 270, y2=legend_y - 6)
    svg.element("circle", class_="admission-dot", cx=x + 254, cy=legend_y - 6, r=5)
    svg.text(x + 280, legend_y, "admitted requests / 16", "tiny")
    svg.text(x + width - 30, legend_y, "saved KV-pool MiB below", "tiny", text_anchor="end")

    chart_left = x + 70
    chart_right = x + width - 40
    chart_top = y + 135
    chart_bottom = y + 291
    for fraction in (0.0, 0.5, 1.0):
        line_y = chart_bottom - fraction * (chart_bottom - chart_top)
        svg.element("line", class_="grid", x1=chart_left, y1=line_y, x2=chart_right, y2=line_y)
        svg.text(chart_left - 12, line_y + 4, str(int(64 * fraction)), "tiny", text_anchor="end")

    points = data["points"]
    x_positions = [chart_left + 52 + index * 132 for index in range(4)]
    admission_points = []
    for point, point_x in zip(points, x_positions):
        bar_height = point["physical_context_blocks"] / 64 * (chart_bottom - chart_top)
        svg.element(
            "rect",
            class_="context-bar",
            x=point_x - 28,
            y=chart_bottom - bar_height,
            width=56,
            height=bar_height,
            rx=8,
        )
        svg.text(point_x - 16, chart_bottom - bar_height - 9, str(point["physical_context_blocks"]), "bar-label")
        admission_y = chart_bottom - point["admitted_requests"] / 16 * (chart_bottom - chart_top)
        admission_points.append((point_x, admission_y, point["admitted_requests"]))

    svg.element(
        "polyline",
        class_="admission-line",
        points=" ".join(f"{_number(px)},{_number(py)}" for px, py, _ in admission_points),
    )
    for (point_x, point_y, admitted), point in zip(admission_points, points):
        svg.element("circle", class_="admission-dot", cx=point_x, cy=point_y, r=7)
        svg.text(point_x + 23, point_y - 13, f"{admitted}/16", "bar-label")
        svg.text(point_x, chart_bottom + 22, f"{point['hit_rate_percent']}% hit", "small", text_anchor="middle")
        saved = point["saved_kv_capacity_mib"]
        svg.text(point_x, chart_bottom + 43, f"+{saved:.1f} MiB", "tiny", text_anchor="middle")

    svg.text(
        x + width / 2,
        y + height - 19,
        "KV-pool capacity—not process VRAM · 75% hit: 20/64 blocks, 16/16 admitted",
        "tiny",
        text_anchor="middle",
    )
    _panel_end(svg)


def _render_scorecard(
    svg: SVG,
    data: dict[str, Any],
    integrated_lifecycle: dict[str, Any],
    p: dict[str, str],
) -> None:
    x, y, width, height = 64, 584, 674, 416
    _panel(svg, x, y, width, height, "scorecard-panel", "Transaction optimization outcomes")
    svg.text(x + 30, y + 43, "Transaction optimization", "panel-title")
    svg.text(x + 30, y + 69, "p50 ratio · stable groups · evidence disposition", "panel-subtitle")

    svg.text(x + 31, y + 104, "PATH / FEATURE", "tiny")
    svg.text(x + 392, y + 104, "P50", "tiny", text_anchor="end")
    svg.text(x + 475, y + 104, "STABLE", "tiny", text_anchor="middle")
    svg.text(x + width - 31, y + 104, "OUTCOME", "tiny", text_anchor="end")

    outcome_labels = {
        "observed": "OBSERVED",
        "accepted": "ADOPTED",
        "rejected_and_rolled_back": "NOT ADOPTED",
    }
    for index, row in enumerate(data["entries"]):
        row_y = y + 122 + index * 82
        svg.element("rect", class_="row", x=x + 22, y=row_y, width=width - 44, height=68, rx=13)
        svg.text(x + 38, row_y + 25, row["feature"], "label")
        svg.text(x + 38, row_y + 47, row["ratio_direction"], "tiny")
        svg.text(x + 392, row_y + 30, f"{row['p50_ratio']:.4f}×", "value", text_anchor="end")
        svg.text(x + 475, row_y + 29, f"{row['stable_groups']}/{row['total_groups']}", "value", text_anchor="middle")
        svg.text(x + 475, row_y + 48, "min > 1", "tiny", text_anchor="middle")
        rejected = row["outcome"] == "rejected_and_rolled_back"
        pill_width = 112 if rejected else 104
        pill_x = x + width - 38 - pill_width
        svg.element(
            "rect",
            x=pill_x,
            y=row_y + 17,
            width=pill_width,
            height=32,
            rx=16,
            fill=p["red_soft"] if rejected else p["green_soft"],
            stroke=p["red"] if rejected else p["green"],
        )
        svg.text(
            pill_x + pill_width / 2,
            row_y + 38,
            outcome_labels[row["outcome"]],
            "value-red" if rejected else "value-green",
            text_anchor="middle",
        )

    svg.text(
        x + 30,
        y + height - 20,
        "Integrated lifecycle · "
        f"{integrated_lifecycle['rows']} validated rows / "
        f"{integrated_lifecycle['trials']} trials",
        "tiny",
    )
    svg.text(x + width - 30, y + height - 20, "Persistent metadata was not adopted", "tiny", text_anchor="end")
    _panel_end(svg)


def _render_flashinfer_kernel_baseline(
    svg: SVG,
    data: dict[str, Any],
    p: dict[str, str],
) -> None:
    x, y, width, height = 762, 584, 674, 416
    _panel(
        svg,
        x,
        y,
        width,
        height,
        "flashinfer-kernel-baseline-panel",
        "FlashInfer kernel-only baseline",
    )
    svg.text(x + 30, y + 43, "External kernel baseline", "panel-title")
    svg.text(x + 30, y + 69, "Common-shape paged decode · geometric mean across 8 groups", "panel-subtitle")

    plot_left = x + 254
    plot_right = x + width - 56
    plot_top = y + 140
    plot_bottom = y + 275
    for tick in (1.0, 1.1, 1.2, 1.3):
        tick_x = plot_left + (tick - 1.0) / 0.3 * (plot_right - plot_left)
        svg.element("line", class_="grid", x1=tick_x, y1=plot_top - 12, x2=tick_x, y2=plot_bottom)
        svg.text(tick_x, plot_top - 21, f"{tick:.1f}×", "tiny", text_anchor="middle")
    svg.text(plot_right, plot_top - 44, "higher → FlashInfer lower p50", "tiny", text_anchor="end")

    for index, backend in enumerate(data["backends"]):
        row_y = plot_top + 30 + index * 82
        value = backend["p50_ratio_geometric_mean"]
        value_x = plot_left + (value - 1.0) / 0.3 * (plot_right - plot_left)
        svg.text(x + 30, row_y + 5, backend["label"], "label")
        svg.text(x + 30, row_y + 25, "FlashDec / FlashInfer p50", "tiny")
        svg.element("line", class_="ratio-line", x1=plot_left, y1=row_y, x2=value_x, y2=row_y)
        svg.element("circle", class_="ratio-dot", cx=value_x, cy=row_y, r=9)
        svg.text(x + width - 30, row_y + 6, f"{value:.4f}×", "value", text_anchor="end")

    note_y = y + 292
    svg.element("rect", x=x + 24, y=note_y, width=width - 48, height=91, rx=14, fill=p["purple_soft"], stroke=p["purple"])
    svg.text(x + 41, note_y + 27, ">1 favors FlashInfer · descriptive p50 evidence only", "label")
    svg.text(
        x + 41,
        note_y + 51,
        f"{data['observed_p50_ranges_above_one']}/{data['total_p50_ranges']} observed three-trial ranges are above 1; this is descriptive kernel evidence.",
        "small",
    )
    svg.text(x + 41, note_y + 74, "KERNEL-ONLY · excludes scheduler, KV ownership and end-to-end serving", "tiny")
    _panel_end(svg)


def render_svg(data: dict[str, Any], theme: str) -> str:
    """Render one deterministic, standalone light or dark SVG."""
    if theme not in PALETTES:
        raise ValueError(f"unknown theme: {theme}")
    p = PALETTES[theme]
    svg = SVG()
    svg.raw('<?xml version="1.0" encoding="UTF-8"?>')
    svg.open(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width="1500",
        height="1080",
        viewBox="0 0 1500 1080",
        role="img",
        aria_labelledby="chart-title chart-description",
        data_theme=theme,
    )
    svg.element("title", "FlashDec auditable research evidence overview", id="chart-title")
    svg.element(
        "desc",
        "Four panels summarize scheduler progress, shared-prefix KV-pool capacity, "
        "transaction optimization outcomes including a negative result and integrated lifecycle validation, "
        "and the FlashInfer kernel-only p50 comparison. Data was derived from validated canonical "
        "Markdown summaries on an NVIDIA GeForce RTX 5070 and is not a raw dataset.",
        id="chart-description",
    )
    outcomes_by_id = {
        row["id"]: row for row in data["optimization_outcomes"]["entries"]
    }
    evidence_sections = [
        ("scheduler_progress", data["scheduler_progress"]),
        ("fused_append", outcomes_by_id["fused_append"]),
        ("shared_prefix_capacity", data["shared_prefix_capacity"]),
        ("trusted_transaction", outcomes_by_id["trusted_transaction"]),
        ("persistent_metadata", outcomes_by_id["persistent_metadata"]),
        ("integrated_lifecycle", data["integrated_lifecycle"]),
        ("flashinfer_kernel_baseline", data["flashinfer_kernel_baseline"]),
    ]
    metadata = {
        "device": data["artifact"]["device"],
        "evidence": [
            {
                "id": evidence_id,
                "commit": section["evidence_commit"],
                "rows": section["rows"],
                "trials": section["trials"],
            }
            for evidence_id, section in evidence_sections
        ],
        "snapshot": "benchmarks/results/public_results_snapshot.json",
        "sources": sorted(
            {
                data["scheduler_progress"]["source"],
                data["shared_prefix_capacity"]["source"],
                data["flashinfer_kernel_baseline"]["source"],
                *(row["source"] for row in data["optimization_outcomes"]["entries"]),
                data["integrated_lifecycle"]["source"],
            }
        ),
    }
    svg.element("metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")), id="evidence-metadata")
    svg.open("defs")
    svg.open("linearGradient", id="header-glow", x1="0", y1="0", x2="1", y2="1")
    svg.element("stop", offset="0%", stop_color=p["blue"], stop_opacity="0.22")
    svg.element("stop", offset="100%", stop_color=p["purple"], stop_opacity="0.03")
    svg.close("linearGradient")
    svg.open("linearGradient", id="panel-shadow", x1="0", y1="0", x2="0", y2="1")
    svg.element("stop", offset="0%", stop_color=p["shadow"], stop_opacity="0")
    svg.element("stop", offset="100%", stop_color=p["shadow"], stop_opacity="1")
    svg.close("linearGradient")
    svg.open("pattern", id="cancel-pattern", width="8", height="8", patternUnits="userSpaceOnUse", patternTransform="rotate(45)")
    svg.element("rect", width="8", height="8", fill=p["amber_soft"])
    svg.element("line", x1="0", y1="0", x2="0", y2="8", stroke=p["amber"], stroke_width="3", stroke_opacity="0.45")
    svg.close("pattern")
    svg.open("style")
    svg.raw(_style(p).strip())
    svg.close("style")
    svg.close("defs")

    svg.element("rect", width="1500", height="1080", fill=p["background"])
    svg.element("ellipse", cx="1270", cy="20", rx="370", ry="215", fill="url(#header-glow)", aria_hidden="true")
    svg.element("rect", x="64", y="48", width="308", height="31", rx="15.5", fill=p["blue_soft"], stroke=p["blue"], aria_hidden="true")
    svg.text(218, 69, "AUDITABLE EVIDENCE SNAPSHOT", "eyebrow", text_anchor="middle")
    svg.text(64, 126, data["artifact"]["title"], "title")
    svg.text(64, 157, "RTX 5070 · derived from validated canonical summaries · processed snapshot, not a raw dataset", "subtitle")

    _render_scheduler_progress(svg, data["scheduler_progress"], p)
    _render_shared_prefix_capacity(svg, data["shared_prefix_capacity"], p)
    _render_scorecard(
        svg,
        data["optimization_outcomes"],
        data["integrated_lifecycle"],
        p,
    )
    _render_flashinfer_kernel_baseline(
        svg,
        data["flashinfer_kernel_baseline"],
        p,
    )

    svg.text(750, 1037, "Ratios retain each source's direction and are not comparable across panels.", "footer")
    svg.text(750, 1061, "Canonical Markdown remains authoritative · theme-specific SVGs are deterministic", "footer")
    svg.close("svg")
    return svg.finish()


def expected_outputs(data: dict[str, Any]) -> dict[str, str]:
    return {theme: render_svg(data, theme) for theme in ("light", "dark")}


def write_outputs(data: dict[str, Any], outputs: dict[str, Path] = DEFAULT_OUTPUTS) -> None:
    rendered = expected_outputs(data)
    for theme, destination in outputs.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered[theme], encoding="utf-8")
        print(f"Wrote {theme} chart: {destination.relative_to(ROOT)}")


def check_outputs(data: dict[str, Any], outputs: dict[str, Path] = DEFAULT_OUTPUTS) -> list[str]:
    """Return tracked outputs that are missing or differ from deterministic renders."""
    rendered = expected_outputs(data)
    problems = []
    for theme, destination in outputs.items():
        if not destination.is_file():
            problems.append(f"missing {theme} chart: {destination}")
            continue
        if destination.read_text(encoding="utf-8") != rendered[theme]:
            problems.append(f"stale {theme} chart: {destination}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that both tracked SVGs exactly match the deterministic render",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = load_snapshot(args.snapshot)
        if args.check:
            problems = check_outputs(data)
            if problems:
                print("Public results chart check: FAIL")
                for problem in problems:
                    print(f"- {problem}")
                return 1
            print("Public results chart check: PASS (light + dark)")
            return 0
        write_outputs(data)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Public results chart: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
