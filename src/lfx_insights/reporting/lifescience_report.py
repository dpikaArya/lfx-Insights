"""Markdown renderers for parameter-driven life-science artifacts (stats, protocols)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx_insights.models import Protocol, StatRecommendation


def render_stat(rec: StatRecommendation) -> str:
    lines = [
        f"# Statistical Recommendation â€” {rec.method}",
        "",
        f"- Design: {rec.design}",
        f"- Method: {rec.method}" + (f" ({rec.stato_term})" if rec.stato_term else ""),
        f"- alpha: {rec.alpha}",
    ]
    if rec.power is not None:
        lines.append(f"- power: {rec.power}")
    if rec.effect_size is not None:
        lines.append(f"- effect size: {rec.effect_size}")
    if rec.n_per_group is not None:
        lines.append(f"- n per group: {rec.n_per_group}")
    if rec.total_n is not None:
        lines.append(f"- total n: {rec.total_n}")
    if rec.notes:
        lines.append(f"- Notes: {rec.notes}")
    return "\n".join(lines) + "\n"


def render_protocol(protocol: Protocol) -> str:
    lines = [f"# Protocol â€” {protocol.name}", "", f"_Kind: {protocol.kind}_", "", "## Steps", ""]
    for i, step in enumerate(protocol.steps, start=1):
        lines.append(f"{i}. {step}")
    lines.append("")
    lines.append("## QC checklist")
    lines.append("")
    for item in protocol.qc_checklist:
        lines.append(f"- [ ] {item}")
    if protocol.notes:
        lines.append("")
        lines.append(f"> {protocol.notes}")
    return "\n".join(lines) + "\n"
