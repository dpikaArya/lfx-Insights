"""Markdown renderers for generation artifacts (hypotheses, questions, drafts, review)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx_insights.generation.common import disambiguation_suffixes, format_apa

if TYPE_CHECKING:
    from lfx_insights.models import (
        Corpus,
        GeneratedSection,
        Hypothesis,
        ResearchQuestion,
        ReviewComment,
    )


def render_hypotheses(hypotheses: list[Hypothesis]) -> str:
    lines: list[str] = ["# Hypotheses", ""]
    if not hypotheses:
        return "# Hypotheses\n\n_No hypotheses generated._\n"
    for h in hypotheses:
        lines.append(f"## {h.statement}")
        lines.append("")
        lines.append(
            f"- Relation (Bucur SuperPattern): `{h.subject}` â€” **{h.qualifier}** â€” `{h.object}`"
        )
        if h.independent_var or h.dependent_var:
            lines.append(f"- IV â†’ DV: {h.independent_var or '?'} â†’ {h.dependent_var or '?'}")
        if h.methodology:
            lines.append(f"- Methodology: {h.methodology}")
        if h.rationale:
            lines.append(f"- Rationale: {h.rationale}")
        if h.evidence:
            lines.append(f"- Grounded in: {', '.join(e.paper_id for e in h.evidence)}")
        lines.append(f"- Status: {h.status}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_questions(questions: list[ResearchQuestion]) -> str:
    lines: list[str] = ["# Research Questions", ""]
    if not questions:
        return "# Research Questions\n\n_No questions generated._\n"
    for q in questions:
        lines.append(f"## {q.question}")
        lines.append("")
        if q.score is not None:
            comps = ", ".join(f"{c.name} {c.value:.2f}" for c in q.score.components)
            lines.append(f"- Score: **{q.score.value:.2f}** ({q.score.interpretation}) â€” {comps}")
        if q.rationale:
            lines.append(f"- Rationale: {q.rationale}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_sections(title: str, sections: list[GeneratedSection], corpus: Corpus) -> str:
    lines: list[str] = [f"# {title}", ""]
    cited: list[str] = []
    for sec in sections:
        lines.append(f"## {sec.name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(sec.text)
        lines.append("")
        for c in sec.citations:
            if c not in cited:
                cited.append(c)
    if cited:
        papers = []
        for ref in cited:
            paper = corpus.by_id(ref) or next(
                (p for p in corpus.papers if (p.doi or "") == ref), None
            )
            if paper is not None:
                papers.append(paper)
        # Same disambiguation map as the in-text citations baked into sec.text.
        suffixes = disambiguation_suffixes(papers)
        lines.append("## References")
        lines.append("")
        for paper in papers:
            lines.append(f"- {format_apa(paper, suffixes.get(paper.id, ''))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_review(comments: list[ReviewComment]) -> str:
    lines: list[str] = ["# Reviewer Simulation", ""]
    if not comments:
        return "# Reviewer Simulation\n\n_No review comments._\n"
    for c in comments:
        lines.append(f"## [{c.severity}] {c.section}")
        lines.append("")
        lines.append(f"- {c.comment}")
        if c.suggestion:
            lines.append(f"- Suggestion: {c.suggestion}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
