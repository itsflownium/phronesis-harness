#!/usr/bin/env python3
"""Render the Phronesis harness benchmark as a dependency-free SVG chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


WIDTH = 1200
HEIGHT = 720


def validate(data: dict) -> None:
    problems = data["problems"]
    if not problems:
        raise ValueError("benchmark must contain at least one problem")

    for problem in problems:
        for field in ("manual_score", "automated_score"):
            score = problem[field]
            if not isinstance(score, int) or not 0 <= score <= 10:
                raise ValueError(f"{problem['id']} has invalid {field}: {score}")

    declared = {
        "manual_score": data["manual_official_style_score"],
        "automated_score": data["automated_harness_score"],
    }
    calculated = {
        "manual_score": sum(row["manual_score"] for row in problems),
        "automated_score": sum(row["automated_score"] for row in problems),
    }
    if declared != calculated:
        raise ValueError(f"declared totals {declared} do not match problem totals {calculated}")


def svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 18,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(str(value))}</text>'
    )


def render(data: dict) -> str:
    problems = data["problems"]
    max_total = data["max_score"]
    manual_total = data["manual_official_style_score"]
    automated_total = data["automated_harness_score"]

    left = 86
    right = 1120
    gauge_y = 134
    gauge_width = right - left
    manual_width = gauge_width * manual_total / max_total
    automated_x = left + gauge_width * automated_total / max_total

    chart_top = 286
    chart_bottom = 610
    chart_height = chart_bottom - chart_top
    slot = (right - left) / len(problems)
    bar_width = 42

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">Putnam 2021 dev benchmark</title>',
        '<desc id="desc">Qwen3.5-35B-A3B with the Phronesis solver harness scored '
        f'{manual_total} out of {max_total} by manual official-style review. '
        f'The automated harness score was {automated_total}.</desc>',
        '<rect width="1200" height="720" rx="8" fill="#f8fafc"/>',
        svg_text(left, 58, "Putnam 2021 Dev Benchmark", size=32, weight=500),
        svg_text(left, 89, "Qwen3.5-35B-A3B + Phronesis full solver harness", size=17, fill="#526071"),
        svg_text(left, 121, f"Manual official-style audit  {manual_total}/{max_total}", size=22, weight=500),
        f'<rect x="{left}" y="{gauge_y}" width="{gauge_width}" height="34" rx="6" fill="#dbe3ec"/>',
        f'<rect x="{left}" y="{gauge_y}" width="{manual_width:.1f}" height="34" rx="6" fill="#0f766e"/>',
        f'<line x1="{automated_x:.1f}" y1="{gauge_y - 8}" x2="{automated_x:.1f}" '
        f'y2="{gauge_y + 42}" stroke="#d97706" stroke-width="4"/>',
        svg_text(automated_x + 9, gauge_y + 58, f"Harness {automated_total}", size=14, fill="#9a5705"),
        svg_text(right, gauge_y + 25, str(max_total), size=15, anchor="end", fill="#526071"),
        svg_text(left, 229, "Per-problem score", size=22, weight=500),
        '<rect x="86" y="286" width="1034" height="324" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    for tick in range(0, 11, 2):
        y = chart_bottom - chart_height * tick / 10
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 14, y + 5, tick, size=13, anchor="end", fill="#526071"))

    for index, problem in enumerate(problems):
        center = left + slot * (index + 0.5)
        score = problem["manual_score"]
        automated = problem["automated_score"]
        bar_height = chart_height * score / 10
        bar_y = chart_bottom - bar_height
        parts.append(
            f'<rect x="{center - bar_width / 2:.1f}" y="{bar_y:.1f}" width="{bar_width}" '
            f'height="{bar_height:.1f}" rx="4" fill="#1d4ed8"/>'
        )
        marker_y = chart_bottom - chart_height * automated / 10
        parts.append(
            f'<line x1="{center - 29:.1f}" y1="{marker_y:.1f}" x2="{center + 29:.1f}" '
            f'y2="{marker_y:.1f}" stroke="#d97706" stroke-width="4"/>'
        )
        label_y = bar_y - 10 if score else chart_bottom - 10
        parts.append(svg_text(center, label_y, score, size=14, weight=500, anchor="middle"))
        parts.append(svg_text(center, chart_bottom + 29, problem["id"], size=14, weight=500, anchor="middle"))

    parts.extend(
        [
            svg_text(28, (chart_top + chart_bottom) / 2, "Score", size=14, anchor="middle", fill="#526071"),
            '<rect x="86" y="644" width="15" height="15" rx="3" fill="#1d4ed8"/>',
            svg_text(110, 657, "Manual audit", size=14, fill="#526071"),
            '<line x1="232" y1="651" x2="260" y2="651" stroke="#d97706" stroke-width="4"/>',
            svg_text(270, 657, "Automated harness", size=14, fill="#526071"),
            svg_text(
                right,
                657,
                f"{len(problems)}/{len(problems)} problems | {data['model_calls']} model calls",
                size=14,
                anchor="end",
                fill="#526071",
            ),
            svg_text(
                left,
                696,
                "Manual score is an independent official-style review, not an MAA-adjudicated result.",
                size=13,
                fill="#64748b",
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    validate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")


if __name__ == "__main__":
    main()
