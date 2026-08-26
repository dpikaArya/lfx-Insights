"""LLM-backed reviewer simulation over drafted sections.

Given a set of :class:`~consilium.models.GeneratedSection` drafts, ask the LLM to
play a critical peer reviewer and surface SPECIFIC issues (unsupported claims,
missing citations, methodological gaps) â€” not a template that always fires. The
corpus is listed in the prompt so the reviewer can reason about which papers the
sections could legitimately cite.

Guardrails (the exact failures of the old tool, avoided here):
- Severity comes from the LLM and is normalized to the closed vocabulary
  ``{"major", "minor", "praise"}`` (unknown -> "minor"); it is never random.
- Any review whose comment text triggers :func:`has_output_leak` (an unrendered
  ``{placeholder}`` or ``NaN``) is dropped rather than emitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from lfx_insights.generation.common import format_apa, has_output_leak
from lfx_insights.models import ReviewComment

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus, GeneratedSection

# Closed severity vocabulary; anything else collapses to the neutral default.
_SEVERITIES: frozenset[str] = frozenset({"major", "minor", "praise"})
_DEFAULT_SEVERITY = "minor"


class ReviewItem(BaseModel):
    """One reviewer remark, as returned by the LLM."""

    severity: str
    section: str
    comment: str
    suggestion: str | None = None


class ReviewBatch(BaseModel):
    """Batch wrapper so all sections are reviewed in a single LLM call."""

    items: list[ReviewItem]


def _normalize_severity(severity: str) -> str:
    """Map a free-text severity onto the closed vocabulary (unknown -> 'minor')."""
    norm = severity.strip().lower()
    return norm if norm in _SEVERITIES else _DEFAULT_SEVERITY


def _build_prompt(sections: list[GeneratedSection], corpus: Corpus) -> str:
    """Compose a reviewer prompt listing the sections and the citable corpus."""
    corpus_lines = "\n".join(f"{p.id}: {p.title}" for p in corpus.papers)
    section_blocks = "\n\n".join(f"### Section: {s.name}\n{s.text}" for s in sections)
    references = "\n".join(format_apa(p) for p in corpus.papers)
    return (
        "You are a rigorous but fair peer reviewer. Review the manuscript sections "
        "below and produce SPECIFIC, actionable critiques. Do NOT use a generic "
        "template that fires on every section. Only raise a point when the text "
        "actually warrants it. Look for:\n"
        "- claims asserted without support,\n"
        "- statements that should cite the literature but do not,\n"
        "- methodological gaps, overreach, or unclear reasoning.\n"
        "Praise is allowed (severity 'praise') when a section is genuinely strong. "
        "For each remark set 'severity' to one of {major, minor, praise}, 'section' "
        "to the section name it refers to, a concrete 'comment', and an optional "
        "'suggestion' for how to fix it.\n\n"
        "Corpus papers you may reference (cite ONLY these ids):\n"
        f"{corpus_lines}\n\n"
        "Full references:\n"
        f"{references}\n\n"
        "Sections to review:\n"
        f"{section_blocks}\n"
    )


def simulate_review(
    sections: list[GeneratedSection], corpus: Corpus, llm: LLMClient
) -> list[ReviewComment]:
    """Simulate peer review of ``sections``, grounded in ``corpus``.

    Issues a single batched LLM call. Each returned item becomes a
    :class:`ReviewComment` with a normalized severity; items whose comment text
    leaks an unrendered placeholder or ``NaN`` are dropped.
    """
    if not sections:
        return []

    prompt = _build_prompt(sections, corpus)
    batch = llm.complete_structured(prompt, ReviewBatch)

    comments: list[ReviewComment] = []
    for item in batch.items:
        if has_output_leak(item.comment):
            continue
        comments.append(
            ReviewComment(
                severity=_normalize_severity(item.severity),
                section=item.section,
                comment=item.comment,
                suggestion=item.suggestion,
            )
        )
    return comments
