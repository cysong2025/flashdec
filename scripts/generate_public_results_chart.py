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
    if data.get("schema_version") != 3:
        raise ValueError("snapshot schema_version must be 3")

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

    model_end_to_end = data["vllm_model_end_to_end"]
    vllm_attention = data["vllm_attention"]
    scheduler = data["scheduler_progress"]
    shared_prefix = data["shared_prefix_capacity"]
    optimization = data["optimization_outcomes"]
    outcomes = optimization["entries"]
    integrated = data["integrated_lifecycle"]
    external_baseline = data["flashinfer_kernel_baseline"]
    sections = [
        model_end_to_end,
        vllm_attention,
        scheduler,
        shared_prefix,
        *outcomes,
        integrated,
        external_baseline,
    ]
    source_texts: dict[str, str] = {}
    for section in sections:
        source = root / section["source"]
        if not source.is_file():
            raise ValueError(f"canonical source does not exist: {section['source']}")
        source_texts[section["source"]] = source.read_text(encoding="utf-8")

    if model_end_to_end.get("kind") != "offline_fixed_batch_llm_generate":
        raise ValueError("model result must retain its offline fixed-batch boundary")
    if model_end_to_end.get("baseline") != "vLLM 0.25.1 TRITON_ATTN":
        raise ValueError("model result must name the explicit vLLM baseline")
    model_text = source_texts[model_end_to_end["source"]]
    model_validation = re.search(
        r"(?m)^- Rows: (\d+); paired backend process pairs: \d+ "
        r"\((\d+) trials per case\)\.$",
        model_text,
    )
    model_commit = re.search(r"(?m)^- Git commit: `([0-9a-f]+)`;", model_text)
    model_device = re.search(r"(?m)^- Device: (.+)\.$", model_text)
    if model_validation is None or model_commit is None or model_device is None:
        raise ValueError("model canonical source validation metadata is incomplete")
    if (
        int(model_validation.group(1)) != model_end_to_end["rows"]
        or int(model_validation.group(2)) != model_end_to_end["trials"]
        or model_commit.group(1) != model_end_to_end["evidence_commit"]
        or model_device.group(1) != artifact["device"]
    ):
        raise ValueError("model snapshot provenance differs from canonical source")
    model_rows = {
        row["case"]: row for row in _markdown_table(model_text, "Paired Results")
    }
    target = model_rows.get(model_end_to_end["target_case"])
    guard = model_rows.get(model_end_to_end["guard_case"])
    if target is None or guard is None:
        raise ValueError("model canonical target or guard case is missing")
    target_ratio = _ratio_range(target["ratio [min,max]"])
    guard_ratio = _ratio_range(guard["ratio [min,max]"])[0]
    if (
        target_ratio
        != (
            model_end_to_end["latency_ratio"],
            *model_end_to_end["latency_range"],
        )
        or float(target["latency reduction"].rstrip("%"))
        != model_end_to_end["latency_reduction_percent"]
        or float(target["output TPS uplift"].rstrip("%"))
        != model_end_to_end["output_tps_uplift_percent"]
        or guard_ratio != model_end_to_end["guard_latency_ratio"]
    ):
        raise ValueError("model snapshot results differ from canonical source")

    if vllm_attention.get("kind") != "single_token_decode_attention_kernel":
        raise ValueError("vLLM attention result must remain kernel-only")
    attention_text = source_texts[vllm_attention["source"]]
    attention_rows_meta = re.search(
        r"(?m)^- Rows: (\d+); paired trials: (\d+)\.$",
        attention_text,
    )
    attention_commit = re.search(
        r"(?m)^- Git commit: `([0-9a-f]+)`;",
        attention_text,
    )
    attention_device = re.search(r"(?m)^- Device: (.+)\.$", attention_text)
    if (
        attention_rows_meta is None
        or attention_commit is None
        or attention_device is None
    ):
        raise ValueError("vLLM attention canonical metadata is incomplete")
    if (
        attention_commit.group(1) != vllm_attention["evidence_commit"]
        or int(attention_rows_meta.group(1)) != vllm_attention["rows"]
        or int(attention_rows_meta.group(2))
        != len(_markdown_table(attention_text, "Paired Results"))
        * vllm_attention["trials"]
        or attention_device.group(1) != artifact["device"]
    ):
        raise ValueError("vLLM attention snapshot provenance differs from canonical source")
    attention_rows = {
        row["case"]: row
        for row in _markdown_table(attention_text, "Paired Results")
    }
    for case in vllm_attention["cases"]:
        canonical = attention_rows.get(case["id"])
        if canonical is None:
            raise ValueError(f"missing canonical attention case: {case['id']}")
        ratio = _ratio_range(canonical["ratio [min,max]"])[0]
        if ratio != case["latency_ratio"] or round((1.0 - ratio) * 100, 2) != case[
            "latency_reduction_percent"
        ]:
            raise ValueError(f"attention snapshot result changed: {case['id']}")

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


def _style(p: dict[str, str]) -> str:
    return f"""
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif; }}
      .eyebrow {{ fill: {p['purple']}; font-size: 15px; font-weight: 760; letter-spacing: 1.5px; }}
      .title {{ fill: {p['text']}; font-size: 38px; font-weight: 780; letter-spacing: -0.7px; }}
      .subtitle {{ fill: {p['muted']}; font-size: 16px; font-weight: 480; }}
      .panel-title {{ fill: {p['text']}; font-size: 21px; font-weight: 740; }}
      .panel-copy {{ fill: {p['muted']}; font-size: 14px; font-weight: 500; }}
      .metric {{ fill: {p['green']}; font-size: 48px; font-weight: 800; font-variant-numeric: tabular-nums; }}
      .metric-small {{ fill: {p['text']}; font-size: 29px; font-weight: 780; font-variant-numeric: tabular-nums; }}
      .metric-label {{ fill: {p['muted']}; font-size: 14px; font-weight: 650; }}
      .label {{ fill: {p['text']}; font-size: 15px; font-weight: 680; }}
      .tiny {{ fill: {p['muted']}; font-size: 12px; font-weight: 520; }}
      .footer {{ fill: {p['muted']}; font-size: 12px; font-weight: 520; text-anchor: middle; }}
      .panel {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.1; }}
      .card {{ fill: {p['panel_alt']}; stroke: {p['border']}; stroke-width: 1; }}
      .baseline {{ fill: {p['grid']}; }}
      .flashdec {{ fill: {p['purple']}; }}
      .good {{ fill: {p['green']}; }}
      .bar-label {{ fill: {p['text']}; font-size: 12px; font-weight: 700; }}
    """.strip()


def _panel(svg: SVG, x: int, y: int, width: int, height: int, panel_id: str, label: str) -> None:
    svg.open("g", id=panel_id, role="group", aria_label=label)
    svg.element("rect", x=x, y=y + 6, width=width, height=height, rx=20, fill="url(#panel-shadow)")
    svg.element("rect", class_="panel", x=x, y=y, width=width, height=height, rx=20)


def _panel_end(svg: SVG) -> None:
    svg.close("g")


def _render_model_panel(svg: SVG, data: dict[str, Any], p: dict[str, str]) -> None:
    x, y, width, height = 50, 148, 630, 268
    _panel(svg, x, y, width, height, "model-panel", "Qwen end-to-end generation result")
    svg.element("rect", x=x + 24, y=y + 22, width=116, height=24, rx=12, fill=p["green_soft"], stroke=p["green"])
    svg.text(x + 82, y + 39, "END-TO-END", "eyebrow", text_anchor="middle")
    svg.text(x + 24, y + 78, "Qwen2.5-3B long-context generation", "panel-title")
    svg.text(x + 24, y + 101, "B8 · input 8192 · output 4096 · BF16 · 4 paired trials", "panel-copy")
    svg.text(x + 24, y + 166, f"−{data['latency_reduction_percent']:.2f}%", "metric")
    svg.text(x + 30, y + 191, "LLM.generate latency", "metric-label")
    svg.text(x + 328, y + 166, f"+{data['output_tps_uplift_percent']:.2f}%", "metric")
    svg.text(x + 334, y + 191, "output tokens / second", "metric-label")
    bar_x, bar_y, bar_width = x + 24, y + 218, width - 48
    svg.element("rect", x=bar_x, y=bar_y, width=bar_width, height=14, rx=7, class_="baseline")
    svg.element("rect", x=bar_x, y=bar_y, width=bar_width * data["latency_ratio"], height=14, rx=7, class_="flashdec")
    svg.text(bar_x, bar_y + 34, "FlashDec", "bar-label")
    svg.text(bar_x + bar_width, bar_y + 34, f"{data['latency_ratio']:.4f}× vs {data['baseline']}", "bar-label", text_anchor="end")
    _panel_end(svg)


def _render_attention_panel(svg: SVG, data: dict[str, Any], p: dict[str, str]) -> None:
    x, y, width, height = 720, 148, 630, 268
    _panel(svg, x, y, width, height, "attention-panel", "vLLM attention kernel result")
    svg.element("rect", x=x + 24, y=y + 22, width=112, height=24, rx=12, fill=p["purple_soft"], stroke=p["purple"])
    svg.text(x + 80, y + 39, "KERNEL", "eyebrow", text_anchor="middle")
    svg.text(x + 24, y + 78, "vLLM decode-attention p50", "panel-title")
    svg.text(x + 24, y + 101, "Qwen grouped-GQA · B8 · FlashDec / vLLM Triton", "panel-copy")
    chart_x, chart_width = x + 190, width - 230
    for index, case in enumerate(data["cases"]):
        row_y = y + 145 + index * 70
        svg.text(x + 24, row_y + 5, f"context {case['context_tokens']}", "label")
        svg.text(x + 24, row_y + 25, f"−{case['latency_reduction_percent']:.2f}%", "metric-label")
        svg.element("rect", x=chart_x, y=row_y - 8, width=chart_width, height=16, rx=8, class_="baseline")
        svg.element("rect", x=chart_x, y=row_y - 8, width=chart_width * case["latency_ratio"], height=16, rx=8, class_="flashdec")
        svg.text(chart_x + chart_width, row_y + 29, f"{case['latency_ratio']:.4f}×", "bar-label", text_anchor="end")
    _panel_end(svg)


def _metric_card(
    svg: SVG,
    *,
    x: int,
    title: str,
    value: str,
    copy: str,
    boundary: str,
    color: str,
    soft: str,
) -> None:
    y, width, height = 448, 400, 184
    svg.element("rect", x=x, y=y + 5, width=width, height=height, rx=18, fill="url(#panel-shadow)")
    svg.element("rect", x=x, y=y, width=width, height=height, rx=18, fill=soft, stroke=color, stroke_width="1.2")
    svg.element("circle", cx=x + 30, cy=y + 31, r=8, fill=color)
    svg.text(x + 50, y + 37, title, "panel-title")
    svg.text(x + 24, y + 97, value, "metric-small")
    svg.text(x + 24, y + 124, copy, "label")
    svg.text(x + 24, y + 155, boundary, "tiny")


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
        width="1400",
        height="760",
        viewBox="0 0 1400 760",
        role="img",
        aria_labelledby="chart-title chart-description",
        data_theme=theme,
    )
    svg.element("title", "FlashDec performance at a glance", id="chart-title")
    svg.element(
        "desc",
        "Selected validated results for Qwen end-to-end generation, vLLM attention, shared-prefix capacity, transaction dispatch and scheduler progress. A compact note retains the FlashInfer kernel-only negative comparison.",
        id="chart-description",
    )
    outcomes = {
        row["id"]: row for row in data["optimization_outcomes"]["entries"]
    }
    evidence_sections = [
        ("vllm_model_end_to_end", data["vllm_model_end_to_end"]),
        ("vllm_attention", data["vllm_attention"]),
        ("shared_prefix_capacity", data["shared_prefix_capacity"]),
        ("trusted_transaction", outcomes["trusted_transaction"]),
        ("scheduler_progress", data["scheduler_progress"]),
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
        "sources": sorted({section["source"] for _, section in evidence_sections}),
    }
    svg.element("metadata", json.dumps(metadata, sort_keys=True, separators=(",", ":")), id="evidence-metadata")
    svg.open("defs")
    svg.open("linearGradient", id="background", x1="0", y1="0", x2="1", y2="1")
    svg.element("stop", offset="0%", stop_color=p["background"])
    svg.element("stop", offset="100%", stop_color=p["purple_soft"], stop_opacity="0.35")
    svg.close("linearGradient")
    svg.open("linearGradient", id="panel-shadow", x1="0", y1="0", x2="0", y2="1")
    svg.element("stop", offset="0%", stop_color=p["shadow"], stop_opacity="0")
    svg.element("stop", offset="100%", stop_color=p["shadow"], stop_opacity="1")
    svg.close("linearGradient")
    svg.open("style")
    svg.raw(_style(p))
    svg.close("style")
    svg.close("defs")

    svg.element("rect", width="1400", height="760", rx="26", fill="url(#background)", stroke=p["border"])
    svg.text(50, 48, "VALIDATED PERFORMANCE SNAPSHOT", "eyebrow")
    svg.text(50, 91, data["artifact"]["title"], "title")
    svg.text(50, 120, "RTX 5070 · Qwen2.5-3B + vLLM results first · internal mechanisms kept in scope", "subtitle")

    _render_model_panel(svg, data["vllm_model_end_to_end"], p)
    _render_attention_panel(svg, data["vllm_attention"], p)

    prefix = data["shared_prefix_capacity"]["points"][-1]
    trusted = outcomes["trusted_transaction"]
    lifetime = next(
        row
        for row in data["scheduler_progress"]["policies"]
        if row["id"] == "lifetime_fifo_aging"
    )
    greedy = next(
        row
        for row in data["scheduler_progress"]["policies"]
        if row["id"] == "greedy_step_only"
    )
    _metric_card(
        svg,
        x=50,
        title="KV capacity",
        value=f"−{prefix['physical_context_blocks'] / data['shared_prefix_capacity']['logical_context_blocks'] * -100 + 100:.1f}%",
        copy=f"physical context blocks · admission {prefix['admitted_requests']}/16",
        boundary="75% prefix hit · fixed KV pool · not process VRAM",
        color=p["blue"],
        soft=p["blue_soft"],
    )
    _metric_card(
        svg,
        x=500,
        title="Transaction dispatch",
        value=f"{trusted['p50_ratio']:.4f}×",
        copy="complete-token p50 · trusted vs checked",
        boundary=f"{trusted['stable_groups']}/{trusted['total_groups']} groups · cache-owned metadata only",
        color=p["purple"],
        soft=p["purple_soft"],
    )
    _metric_card(
        svg,
        x=950,
        title="Scheduler progress",
        value=f"{lifetime['completed'] / 2 * 100:.0f}% vs {greedy['completed'] / 2 * 100:.0f}%",
        copy="boundary-case request completion",
        boundary="lifetime commitment vs greedy · progress, not latency",
        color=p["green"],
        soft=p["green_soft"],
    )

    flashinfer = data["flashinfer_kernel_baseline"]
    values = " / ".join(
        f"{row['p50_ratio_geometric_mean']:.4f}×" for row in flashinfer["backends"]
    )
    svg.element("rect", x="50", y="655", width="1300", height="54", rx="16", fill=p["panel"], stroke=p["border"])
    svg.text(72, 679, "External reality check", "label")
    svg.text(232, 679, f"FlashDec / FlashInfer p50 = {values}; FlashInfer is lower-latency for this kernel-only matrix.", "panel-copy")
    svg.text(72, 699, "This result does not include scheduler, KV ownership, prefix cache or end-to-end model execution.", "tiny")
    svg.text(700, 738, "Canonical Markdown is authoritative · ratio directions differ across panels and are not composable", "footer")
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
