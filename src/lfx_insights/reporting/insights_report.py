"""Render a list of Insights as Markdown (scores shown with components)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx_insights.models import Insight


def render_insights(title: str, insights: list[Insight]) -> str:
    lines: list[str] = [f"# {title}", ""]
    if not insights:
        lines.append("_No insights produced._")
        return "\n".join(lines) + "\n"
    for ins in insights:
        lines.append(f"## {ins.statement}")
        lines.append("")
        if ins.score is not None:
            s = ins.score
            unc = f" Â± {s.uncertainty:.2f}" if s.uncertainty is not None else ""
            lines.append(f"- Score: **{s.value:.2f}** ({s.interpretation}){unc} â€” {s.method}")
            for c in s.components:
                lines.append(f"  - {c.name}: {c.value:.2f} (w={c.weight})")
        if ins.reasoning:
            lines.append(f"- Reasoning: {ins.reasoning}")
        if ins.tags:
            lines.append(f"- Tags: {', '.join(ins.tags)}")
        if ins.evidence:
            refs = ", ".join(e.paper_id for e in ins.evidence)
            lines.append(f"- Evidence: {refs}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
