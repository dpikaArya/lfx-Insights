"""Aggregation renderers: dashboard, research brief, explainability trace."""

from __future__ import annotations

from typing import Any


def render_dashboard(run: dict[str, Any]) -> str:
    lines = ["# Research Dashboard", ""]
    astra = run["astra"]
    if astra:
        lines.append("| Stage | Insights |")
        lines.append("|---|---|")
        for stage, coll in sorted(astra.items()):
            lines.append(f"| {stage} | {len(coll.get('insights', []))} |")
        lines.append("")
    if run["indicium"]:
        for stage, doc in sorted(run["indicium"].items()):
            lines.append(f"- {stage}: {len(doc.get('claims', []))} claim(s)")
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name in run["markdown"]:
        lines.append(f"- {name}")
    return "\n".join(lines).rstrip() + "\n"


def render_brief(run: dict[str, Any]) -> str:
    lines = ["# Research Brief", ""]
    for stage, coll in sorted(run["astra"].items()):
        insights = coll.get("insights", [])
        if not insights:
            continue
        lines.append(f"## {coll.get('title', stage)}")
        lines.append("")
        for ins in insights[:3]:
            lines.append(f"- {ins.get('statement', '')}")
        lines.append("")
    for stage, doc in sorted(run["indicium"].items()):
        claims = doc.get("claims", {})
        claim_list = list(claims.values()) if isinstance(claims, dict) else list(claims)
        if not claim_list:
            continue
        lines.append(f"## {stage} (draft claims)")
        lines.append("")
        for c in claim_list[:3]:
            if not isinstance(c, dict):
                continue
            text = c.get("statement") or " — ".join(
                p
                for p in (
                    c.get("subject"),
                    c.get("relation") or c.get("qualifier"),
                    c.get("object"),
                )
                if p
            )
            lines.append(f"- {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_explainability(run: dict[str, Any]) -> str:
    """Trace each finding to the corpus evidence it is grounded in."""
    lines = ["# Explainability Trace", ""]
    for stage, coll in sorted(run["astra"].items()):
        insights = coll.get("insights", [])
        if not insights:
            continue
        lines.append(f"## {coll.get('title', stage)}")
        lines.append("")
        for ins in insights:
            refs = ", ".join(e.get("paper_id", "?") for e in ins.get("evidence", [])) or "—"
            lines.append(f"- **{ins.get('statement', '')}** ← evidence: {refs}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
