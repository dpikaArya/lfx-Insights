"""Novelty scoring for discovered themes.

A theme is "novel" / emerging when its papers are recent, its yearly publication
counts are trending up, and the theme is comparatively small (underexplored).

Honest by construction: every :class:`~consilium.models.Insight` carries a
:class:`~consilium.models.Score` built from explicit, weighted components via the
shared :func:`~consilium.scoring.common.make_score` kernel — never a bare number.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from consilium.corpus_features import max_year, theme_years
from consilium.models import EvidenceRef, Insight, ScoreComponent
from consilium.scoring.common import clamp01, make_score, minmax_normalize, sample_uncertainty

if TYPE_CHECKING:
    from consilium.models import Corpus, Theme

_RECENCY_WINDOW = 2
_EVIDENCE_CAP = 10

_W_RECENCY = 0.4
_W_GROWTH = 0.3
_W_SCARCITY = 0.3

_NEUTRAL = 0.5

# Honest disclosure: this score is a publication-dynamics proxy, not a measure of
# intrinsic originality. Surfaced on every Insight via ``reasoning``.
_PROXY_DISCLOSURE = (
    "Reflects publication recency/emergence (recency_share, growth, scarcity); "
    "not a measure of intrinsic originality."
)


def _recency_share(years: list[int], max_year: int | None) -> float:
    """Fraction of dated papers within ``_RECENCY_WINDOW`` years of ``max_year``.

    Neutral 0.5 when there is no dating signal (no years), matching the project
    convention and :func:`consilium.scoring.opportunity._recency_share`.
    """
    if not years or max_year is None:
        return _NEUTRAL
    cutoff = max_year - _RECENCY_WINDOW
    recent = sum(1 for y in years if y >= cutoff)
    return clamp01(recent / len(years))


def _growth(years: list[int]) -> float:
    """Map the *relative* change in yearly paper counts to 0..1.

    With fewer than two distinct years there is no trend to read, so we return a
    neutral 0.5. Otherwise we take the relative growth of counts from the first
    to the last year, ``(last - first) / max(1, first)``, and squash it through
    ``tanh`` so a positive (rising) trend pushes above 0.5 and a negative
    (declining) trend below. Using the *relative* change (rather than the raw
    slope) keeps trivial absolute rises honest: a 1->2 doubling no longer scores
    the same as a 1->50 surge.
    """
    distinct = sorted(set(years))
    if len(distinct) < 2:
        return _NEUTRAL
    counts = Counter(years)
    first, last = distinct[0], distinct[-1]
    relative = (counts[last] - counts[first]) / max(1, counts[first])
    return clamp01(_NEUTRAL + 0.5 * math.tanh(relative))


def score_novelty(themes: list[Theme], corpus: Corpus) -> list[Insight]:
    """Score each theme's novelty and return one :class:`Insight` per theme.

    Components (weighted mean):
      * ``recency_share`` (w=0.4): share of dated papers within the recency window.
      * ``growth`` (w=0.3): normalized slope of yearly counts (0.5 neutral).
      * ``scarcity`` (w=0.3): ``1 - minmax(size)`` across themes (smaller = novel).

    Higher value means more emerging/novel. Deterministic. Empty input -> ``[]``.
    """
    if not themes or len(corpus) == 0:
        return []

    corpus_max_year = max_year(corpus)
    sizes = [float(t.size()) for t in themes]
    size_norm = minmax_normalize(sizes)

    insights: list[Insight] = []
    for theme, sn in zip(themes, size_norm, strict=True):
        years = theme_years(theme, corpus)
        components = [
            ScoreComponent(
                name="recency_share",
                value=_recency_share(years, corpus_max_year),
                weight=_W_RECENCY,
            ),
            ScoreComponent(name="growth", value=_growth(years), weight=_W_GROWTH),
            ScoreComponent(name="scarcity", value=clamp01(1.0 - sn), weight=_W_SCARCITY),
        ]
        score = make_score(components, uncertainty=sample_uncertainty(theme.size()))
        label = theme.label or f"theme {theme.id}"
        evidence = [EvidenceRef(paper_id=pid) for pid in theme.paper_ids[:_EVIDENCE_CAP]]
        insights.append(
            Insight(
                statement=f"Novelty of theme '{label}' is {score.interpretation}.",
                evidence=evidence,
                is_synthesized=True,
                reasoning=_PROXY_DISCLOSURE,
                tags=["novelty"],
                score=score,
            )
        )
    return insights
