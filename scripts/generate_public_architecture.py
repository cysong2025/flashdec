#!/usr/bin/env python3
"""Generate the light and dark FlashDec README architecture diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from scripts.generate_public_results_chart import PALETTES, SVG
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from generate_public_results_chart import PALETTES, SVG


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = {
    "light": ROOT / "docs" / "assets" / "flashdec-architecture-light.svg",
    "dark": ROOT / "docs" / "assets" / "flashdec-architecture-dark.svg",
}


def _style(p: dict[str, str]) -> str:
    return f"""
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Arial, sans-serif; }}
      .eyebrow {{ fill: {p['purple']}; font-size: 15px; font-weight: 760; letter-spacing: 1.5px; }}
      .title {{ fill: {p['text']}; font-size: 38px; font-weight: 780; letter-spacing: -0.7px; }}
      .subtitle {{ fill: {p['muted']}; font-size: 17px; font-weight: 480; }}
      .lane {{ fill: {p['faint']}; font-size: 13px; font-weight: 760; letter-spacing: 1.3px; }}
      .card-title {{ fill: {p['text']}; font-size: 20px; font-weight: 740; }}
      .card-copy {{ fill: {p['muted']}; font-size: 14px; font-weight: 500; }}
      .chip {{ fill: {p['text']}; font-size: 13px; font-weight: 650; }}
      .edge-label {{ fill: {p['muted']}; font-size: 13px; font-weight: 650; }}
      .scope {{ fill: {p['muted']}; font-size: 13px; font-weight: 520; text-anchor: middle; }}
      .panel {{ fill: {p['panel']}; stroke: {p['border']}; stroke-width: 1.2; }}
      .soft {{ fill: {p['panel_alt']}; stroke: {p['border']}; stroke-width: 1.1; }}
      .arrow {{ fill: none; stroke: {p['faint']}; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }}
      .arrow-purple {{ fill: none; stroke: {p['purple']}; stroke-width: 2.8; stroke-linecap: round; stroke-linejoin: round; }}
      .arrow-amber {{ fill: none; stroke: {p['amber']}; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 7 6; }}
    """.strip()


def _card(
    svg: SVG,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    copy: str,
    fill: str,
    stroke: str,
    icon: str,
) -> None:
    svg.element("rect", x=x, y=y + 5, width=width, height=height, rx=18, fill="url(#shadow)")
    svg.element("rect", x=x, y=y, width=width, height=height, rx=18, fill=fill, stroke=stroke, stroke_width="1.5")
    svg.element("circle", cx=x + 32, cy=y + 31, r=18, fill=stroke, fill_opacity="0.15")
    svg.text(x + 32, y + 37, icon, "card-title", text_anchor="middle")
    svg.text(x + 58, y + 28, title, "card-title")
    svg.text(x + 58, y + 51, copy, "card-copy")


def render_svg(theme: str) -> str:
    if theme not in PALETTES:
        raise ValueError(f"unknown theme: {theme}")
    p = PALETTES[theme]
    svg = SVG()
    svg.raw('<?xml version="1.0" encoding="UTF-8"?>')
    svg.open(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width="1400",
        height="720",
        viewBox="0 0 1400 720",
        role="img",
        aria_labelledby="architecture-title architecture-description",
        data_theme=theme,
    )
    svg.element("title", "FlashDec integration architecture", id="architecture-title")
    svg.element(
        "desc",
        "vLLM owns model execution, prefill, scheduling and sampling. Its custom attention router sends eligible single-token decode to the transactional FlashDec runtime and unsupported work to the native Triton fallback.",
        id="architecture-description",
    )
    svg.open("defs")
    svg.open("linearGradient", id="background", x1="0", y1="0", x2="1", y2="1")
    svg.element("stop", offset="0%", stop_color=p["background"])
    svg.element("stop", offset="100%", stop_color=p["purple_soft"], stop_opacity="0.4")
    svg.close("linearGradient")
    svg.open("linearGradient", id="shadow", x1="0", y1="0", x2="0", y2="1")
    svg.element("stop", offset="0%", stop_color=p["shadow"], stop_opacity="0")
    svg.element("stop", offset="100%", stop_color=p["shadow"], stop_opacity="1")
    svg.close("linearGradient")
    for marker_id, color in (
        ("arrow", p["faint"]),
        ("arrow-purple", p["purple"]),
        ("arrow-amber", p["amber"]),
    ):
        svg.open("marker", id=marker_id, markerWidth="8", markerHeight="8", refX="7", refY="4", orient="auto", markerUnits="strokeWidth")
        svg.element("path", d="M0,0 L8,4 L0,8 Z", fill=color)
        svg.close("marker")
    svg.open("style")
    svg.raw(_style(p))
    svg.close("style")
    svg.close("defs")

    svg.element("rect", width="1400", height="720", rx="26", fill="url(#background)", stroke=p["border"])
    svg.element("circle", cx="70", cy="67", r="26", fill=p["purple"])
    svg.text(70, 76, "⚡", "card-title", text_anchor="middle")
    svg.text(112, 59, "FLASHDEC ARCHITECTURE", "eyebrow")
    svg.text(112, 95, "A focused decode path inside vLLM", "title")
    svg.text(112, 124, "Keep the engine; replace eligible single-token attention with an auditable transactional path.", "subtitle")

    svg.text(52, 180, "VLLM CONTROL PLANE", "lane")
    svg.element("line", x1="232", y1="175", x2="1348", y2="175", stroke=p["border"])
    _card(svg, x=52, y=200, width=250, height=84, title="Request / Qwen", copy="prompt · model · generation", fill=p["blue_soft"], stroke=p["blue"], icon="1")
    _card(svg, x=350, y=200, width=250, height=84, title="vLLM Engine", copy="prefill · scheduler · KV layout", fill=p["panel"], stroke=p["faint"], icon="2")
    _card(svg, x=648, y=190, width=300, height=104, title="FlashDec plugin router", copy="CUSTOM backend · strict eligibility", fill=p["purple_soft"], stroke=p["purple"], icon="3")
    _card(svg, x=1058, y=200, width=290, height=84, title="Sampling / serving", copy="still owned by vLLM", fill=p["blue_soft"], stroke=p["blue"], icon="4")
    svg.element("path", d="M302 242H350", class_="arrow", marker_end="url(#arrow)")
    svg.element("path", d="M600 242H648", class_="arrow-purple", marker_end="url(#arrow-purple)")
    svg.element("path", d="M948 242H1058", class_="arrow", marker_end="url(#arrow)")

    svg.text(52, 356, "FLASHDEC DECODE PATH", "lane")
    svg.element("line", x1="250", y1="351", x2="1348", y2="351", stroke=p["border"])
    svg.element("path", d="M798 294V379", class_="arrow-purple", marker_end="url(#arrow-purple)")
    svg.text(812, 337, "eligible decode", "edge-label")
    _card(svg, x=150, y=382, width=318, height=100, title="DecodeEngine", copy="begin · step layers · commit / abort", fill=p["purple_soft"], stroke=p["purple"], icon="A")
    _card(svg, x=510, y=382, width=380, height=100, title="Transactional PagedKVCache", copy="blocks · seq_len · refcount · rollback", fill=p["green_soft"], stroke=p["green"], icon="B")
    _card(svg, x=932, y=382, width=318, height=100, title="Policy + prefix", copy="admission · FIFO aging · LRU", fill=p["green_soft"], stroke=p["green"], icon="C")
    svg.element("path", d="M468 432H510", class_="arrow-purple", marker_end="url(#arrow-purple)")
    svg.element("path", d="M890 432H932", class_="arrow-purple", marker_end="url(#arrow-purple)")

    svg.element("path", d="M948 224H1012V330H1262", class_="arrow-amber", marker_end="url(#arrow-amber)")
    svg.element("rect", x="1046", y="306", width="302", height="50", rx="15", fill=p["amber_soft"], stroke=p["amber"])
    svg.text(1064, 328, "Unsupported path → vLLM Triton", "card-title")
    svg.text(1064, 347, "prefill · mixed batch · unsupported shape", "card-copy")

    svg.text(52, 544, "GPU DATA PLANE", "lane")
    svg.element("line", x1="198", y1="539", x2="1348", y2="539", stroke=p["border"])
    svg.element("path", d="M700 482V566", class_="arrow-purple", marker_end="url(#arrow-purple)")
    _card(svg, x=220, y=566, width=400, height=88, title="Fused RoPE + KV append", copy="CUDA extension · checked PyTorch fallback", fill=p["amber_soft"], stroke=p["amber"], icon="K")
    _card(svg, x=780, y=566, width=400, height=88, title="Split-KV PagedAttention", copy="Triton · grouped GQA · query-head reducer", fill=p["blue_soft"], stroke=p["blue"], icon="Q")
    svg.element("path", d="M620 610H780", class_="arrow-purple", marker_end="url(#arrow-purple)")

    svg.element("rect", x="270", y="674", width="860", height="30", rx="15", fill=p["panel"], stroke=p["border"])
    svg.text(700, 695, "PyTorch reference  ·  cross-backend parity  ·  capture attestation  ·  commit-bound evidence", "scope")
    svg.close("svg")
    return svg.finish()


def check_outputs() -> list[str]:
    problems: list[str] = []
    for theme, path in DEFAULT_OUTPUTS.items():
        expected = render_svg(theme)
        if not path.is_file():
            problems.append(f"missing {theme} architecture: {path}")
        elif path.read_text(encoding="utf-8") != expected:
            problems.append(f"stale {theme} architecture: {path}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        problems = check_outputs()
        if problems:
            print("Public architecture check: FAIL")
            for problem in problems:
                print(f"- {problem}")
            return 1
        print("Public architecture check: PASS (light + dark)")
        return 0
    for theme, path in DEFAULT_OUTPUTS.items():
        path.write_text(render_svg(theme), encoding="utf-8")
        print(f"Wrote {theme} architecture: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
