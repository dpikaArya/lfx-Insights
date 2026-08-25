"""Funding-priority alignment scoring (deterministic, keyword overlap).

For each theme we build a term set from its keywords and the tokenized titles of
its papers, then measure overlap against a set of funder priority areas. The best
matching priority gives an honest, components-backed alignment Score — no bare
magic numbers, and no embedder/network required.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from consilium.models import EvidenceRef, Insight, ScoreComponent
from consilium.scoring.common import clamp01, make_score

if TYPE_CHECKING:
    from consilium.models import Corpus, Theme

_EVIDENCE_CAP = 10
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class PriorityArea(BaseModel):
    """A funder priority area described by a handful of lowercase keywords."""

    name: str
    keywords: list[str]


DEFAULT_PRIORITIES: list[PriorityArea] = [
    PriorityArea(
        name="Climate & Sustainability",
        keywords=["climate", "carbon", "emissions", "sustainability", "warming"],
    ),
    PriorityArea(
        name="Human Health",
        keywords=["health", "disease", "clinical", "patient", "therapy"],
    ),
    PriorityArea(
        name="AI & Computing",
        keywords=["machine", "learning", "neural", "algorithm", "computing"],
    ),
    PriorityArea(
        name="Food & Agriculture",
        keywords=["food", "crop", "agriculture", "soil", "farming"],
    ),
    PriorityArea(
        name="Energy",
        keywords=["energy", "battery", "solar", "renewable", "grid"],
    ),
]


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from ``text``."""
    return set(_TOKEN_RE.findall(text.lower()))


def _theme_terms(theme: Theme, corpus: Corpus) -> set[str]:
    """Term set for a theme: its keywords plus tokenized paper titles."""
    terms: set[str] = set()
    for kw in theme.keywords:
        terms |= _tokenize(kw)
    for paper_id in theme.paper_ids:
        paper = corpus.by_id(paper_id)
        if paper is not None:
            terms |= _tokenize(paper.title)
    return terms


def _overlap(theme_terms: set[str], priority: PriorityArea) -> tuple[float, list[str]]:
    """Fraction of a priority's keywords present in the theme, and the matches.

    A (possibly multi-word) priority keyword counts as matched iff *all* of its
    tokens are present in the theme term set (token-subset test), so multi-word
    priorities like ``"machine learning"`` match a theme containing both tokens.
    """
    kws = [k.lower() for k in priority.keywords]
    if not kws:
        return 0.0, []
    matched = sorted({k for k in kws if (toks := _tokenize(k)) and toks <= theme_terms})
    return len(matched) / len(kws), matched


def align_funding(
    themes: list[Theme],
    corpus: Corpus,
    priorities: list[PriorityArea] | None = None,
) -> list[Insight]:
    """Score each theme's alignment to the best-matching funder priority area.

    Deterministic keyword overlap; no embedder. Returns ``[]`` for an empty
    corpus, empty theme list, or empty priority list.
    """
    used = DEFAULT_PRIORITIES if priorities is None else priorities
    if not themes or len(corpus) == 0 or not used:
        return []

    insights: list[Insight] = []
    for theme in themes:
        terms = _theme_terms(theme, corpus)

        scored = [(p, *_overlap(terms, p)) for p in used]
        # Best priority = max overlap; deterministic tie-break on input order.
        best_priority, best_overlap, matched = max(scored, key=lambda s: s[1])
        n_with_overlap = sum(1 for _, ov, _ in scored if ov > 0.0)

        components = [
            ScoreComponent(name="alignment", value=clamp01(best_overlap), weight=0.7),
            ScoreComponent(
                name="breadth",
                value=clamp01(n_with_overlap / len(used)),
                weight=0.3,
            ),
        ]
        score = make_score(components)

        label = theme.label or f"theme {theme.id}"
        statement = (
            f"Theme '{label}' aligns with funding priority "
            f"'{best_priority.name}' ({score.interpretation})."
        )
        if matched:
            reasoning = "Matched terms: " + ", ".join(matched) + "."
        else:
            reasoning = "No priority keywords matched this theme's terms."

        evidence = [EvidenceRef(paper_id=pid) for pid in theme.paper_ids[:_EVIDENCE_CAP]]

        insights.append(
            Insight(
                statement=statement,
                evidence=evidence,
                is_synthesized=True,
                reasoning=reasoning,
                tags=["funding", best_priority.name],
                score=score,
            )
        )
    return insights
