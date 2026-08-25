"""Render a themes report as Markdown (no fabricated scores)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from consilium.models import Corpus, Theme


def render_themes_report(themes: list[Theme], corpus: Corpus) -> str:
    lines: list[str] = ["# Theme Discovery", ""]
    lines.append(f"Corpus: **{len(corpus)} papers** in KB `{corpus.kb_id}`.")
    lines.append(f"Discovered **{len(themes)} themes**.")
    lines.append("")
    for theme in themes:
        label = theme.label or f"Theme {theme.id}"
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Papers: {theme.size()}")
        if theme.keywords:
            lines.append(f"- Keywords: {', '.join(theme.keywords)}")
        if theme.rationale:
            lines.append(f"- Rationale: {theme.rationale}")
        lines.append("")
        for pid in theme.paper_ids:
            paper = corpus.by_id(pid)
            if paper:
                year = f" ({paper.year})" if paper.year else ""
                lines.append(f"  - {paper.title}{year}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
