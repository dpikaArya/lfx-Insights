"""Opportunity scoring: rank themes by how attractive they are to pursue.

Self-contained and deterministic. Each theme yields one honest
:class:`~consilium.models.Insight` whose :class:`~consilium.models.Score` carries
three components:

- ``emergence`` â€” fraction of theme papers published in the most recent window
  (``year >= maxYear - 2``), anchored to the *corpus* max year. Weight 0.4.
- ``scarcity`` â€” ``1 - minmax_normalize(theme size)``: small (under-explored)
  themes are more of an opportunity. Weight 0.3.
- ``momentum`` â€” sign-aware transform of the theme's *own* raw yearly-count
  slope, anchored at 0 (declining < 0.5, flat = 0.5, rising > 0.5), independent
  of other themes. Weight 0.3.

Missing publication years are treated as a neutral 0.5 (never a crash), and the
neutral-0.5 normalization rule (no spread â†’ 0.5) is inherited from
:func:`consilium.scoring.common.minmax_normalize`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from lfx_insights.corpus_features import max_year, theme_papers, theme_years
from lfx_insights.models import EvidenceRef, Insight, ScoreComponent
from lfx_insights.scoring.common import (
    clamp01,
    make_score,
    minmax_normalize,
    sample_uncertainty,
)

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, Theme

_EVIDENCE_CAP = 10
_RECENCY_WINDOW = 2

_EMERGENCE_WEIGHT = 0.4
_SCARCITY_WEIGHT = 0.3
_MOMENTUM_WEIGHT = 0.3

# Slopes are papers/year; scale before the tanh so a modest trend (~1 paper/year
# of acceleration) produces a clearly off-neutral momentum without saturating.
_MOMENTUM_SLOPE_SCALE = 1.0


def _recency_share(years: list[int], corpus_max_year: int | None) -> float:
    """Fraction of papers in the recent window; neutral 0.5 when years are unknown."""
    if not years or corpus_max_year is None:
        return 0.5
    cutoff = corpus_max_year - _RECENCY_WINDOW
    recent = sum(1 for y in years if y >= cutoff)
    return recent / len(years)


def _slope(years: list[int]) -> float:
    """Least-squares slope of yearly publication counts within a theme.

    A positive slope means the theme is accelerating. Returns 0.0 when there is
    too little signal to estimate a trend (fewer than two distinct years).
    """
    if not years:
        return 0.0
    counts: dict[int, int] = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    distinct = sorted(counts)
    if len(distinct) < 2:
        return 0.0
    xs = [float(y) for y in distinct]
    ys = [float(counts[y]) for y in distinct]
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return num / denom


def _momentum(slope: float) -> float:
    """Sign-aware momentum from a theme's *own* raw slope, anchored at 0.

    Maps the raw papers/year slope to 0..1 independently of other themes:
    a declining theme (slope < 0) scores below 0.5, a flat or no-signal theme
    (slope == 0) scores the neutral 0.5, and a rising theme (slope > 0) scores
    above 0.5. Deterministic and monotonic in the slope.
    """
    return clamp01(0.5 + 0.5 * math.tanh(_MOMENTUM_SLOPE_SCALE * slope))


def _evidence_for(theme: Theme, corpus: Corpus) -> list[EvidenceRef]:
    """Up to ``_EVIDENCE_CAP`` evidence refs, one per supporting paper present in corpus."""
    refs: list[EvidenceRef] = []
    for paper in theme_papers(theme, corpus):
        refs.append(EvidenceRef(paper_id=paper.id, location="theme"))
        if len(refs) >= _EVIDENCE_CAP:
            break
    return refs


def rank_opportunities(themes: list[Theme], corpus: Corpus) -> list[Insight]:
    """Rank ``themes`` by research-opportunity score (descending, stable).

    Returns one :class:`Insight` per theme, each carrying an honest composite
    :class:`Score` (emergence + scarcity + momentum). Empty themes or an empty
    corpus return ``[]``.
    """
    if not themes or len(corpus) == 0:
        return []

    corpus_max_year = max_year(corpus)

    sizes = [float(t.size()) for t in themes]
    size_norm = minmax_normalize(sizes)

    insights: list[Insight] = []
    for theme, size_n in zip(themes, size_norm, strict=True):
        years = theme_years(theme, corpus)
        emergence = _recency_share(years, corpus_max_year)
        scarcity = 1.0 - size_n
        momentum = _momentum(_slope(years))

        components = [
            ScoreComponent(name="emergence", value=emergence, weight=_EMERGENCE_WEIGHT),
            ScoreComponent(name="scarcity", value=scarcity, weight=_SCARCITY_WEIGHT),
            ScoreComponent(name="momentum", value=momentum, weight=_MOMENTUM_WEIGHT),
        ]
        score = make_score(components, uncertainty=sample_uncertainty(theme.size()))

        label = theme.label or f"#{theme.id}"
        insights.append(
            Insight(
                statement=f"Opportunity in theme '{label}' is {score.interpretation}.",
                evidence=_evidence_for(theme, corpus),
                is_synthesized=True,
                reasoning=(
                    "Composite of emergence (recent publication share), scarcity "
                    "(inverse normalized theme size), and momentum (sign-aware "
                    "transform of the theme's own yearly-count slope)."
                ),
                tags=["opportunity"],
                score=score,
            )
        )

    insights.sort(key=lambda i: i.score.value if i.score is not None else 0.0, reverse=True)
    return insights
