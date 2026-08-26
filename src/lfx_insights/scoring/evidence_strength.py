"""Evidence-strength scoring: how solid is the evidence base under each theme?

Honest by construction: every theme's strength is a :class:`consilium.models.Score`
built from explicit components (study count, recency, source diversity) via the
shared kernel â€” never a bare magic number. Recency is anchored to the most recent
year present in the corpus, so a "recent" paper is judged relative to the corpus,
not to a hard-coded calendar year.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx_insights.corpus_features import max_year, theme_papers
from lfx_insights.models import EvidenceRef, Insight, ScoreComponent
from lfx_insights.scoring.common import (
    clamp01,
    make_score,
    minmax_normalize,
    sample_uncertainty,
)

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, Theme

_EVIDENCE_EVIDENCE_CAP = 10
_RECENCY_WINDOW = 10.0
# Number of distinct sources at which source diversity is considered saturated.
# Size-independent: a theme with this many distinct sources scores 1.0 regardless
# of how many papers it holds (so a large same-source theme is not penalized).
_TARGET_SOURCES = 3
# Uncertainty at/above this threshold flags the headline band as low-confidence.
# sample_uncertainty(n) = 1/sqrt(n): n=1 -> 1.0, n=2 -> 0.707, n=3 -> 0.577,
# n=4 -> 0.5. At 0.57 the caveat fires for any theme of <= 3 papers, so a tiny-n
# theme is never read as an authoritative band.
_HIGH_UNCERTAINTY = 0.57


def _recency(theme: Theme, corpus: Corpus, corpus_max_year: int | None) -> float:
    """Mean recency over the theme's papers.

    Each paper contributes ``clamp01((year - (maxYear - 10)) / 10)``; a paper with
    a missing year (or a corpus with no years at all) contributes a neutral 0.5.
    """
    if not theme.paper_ids:
        return 0.5
    scores: list[float] = []
    for pid in theme.paper_ids:
        paper = corpus.by_id(pid)
        if paper is None or paper.year is None or corpus_max_year is None:
            scores.append(0.5)
            continue
        scores.append(clamp01((paper.year - (corpus_max_year - _RECENCY_WINDOW)) / _RECENCY_WINDOW))
    return sum(scores) / len(scores)


def _source_diversity(theme: Theme, corpus: Corpus) -> float:
    """Distinct sources, normalized against a target, clamped to 0..1.

    Size-independent: it counts how many distinct sources back the theme relative
    to ``_TARGET_SOURCES``, not the fraction of papers that are distinct. A large
    theme drawing on one source is no worse off than a small one â€” adding more
    same-source papers neither helps nor hurts diversity.
    """
    sources = {paper.source for paper in theme_papers(theme, corpus) if paper.source is not None}
    return clamp01(len(sources) / _TARGET_SOURCES)


def score_evidence_strength(themes: list[Theme], corpus: Corpus) -> list[Insight]:
    """Score the strength of the evidence base under each theme.

    Per theme the Score combines three components:

    - ``study_count`` (weight 0.5): the theme size, min-max normalized across all
      themes (more papers -> stronger);
    - ``recency`` (weight 0.3): mean recency of the theme's papers, anchored to the
      max year present in the corpus (missing years -> neutral 0.5);
    - ``source_diversity`` (weight 0.2): distinct sources normalized against
      ``_TARGET_SOURCES`` (size-independent â€” a large same-source theme is not
      penalized for its size).

    Returns one :class:`Insight` per theme (in input order), each carrying its
    Score and one :class:`EvidenceRef` per supporting paper (capped at 10). When a
    theme's uncertainty is high (a single-paper theme), its statement is annotated
    ``(low confidence, n=...)`` so the band is not read as authoritative. Returns
    ``[]`` for an empty theme list.
    """
    if not themes:
        return []

    corpus_max_year = max_year(corpus)

    study_counts = minmax_normalize([float(t.size()) for t in themes])

    insights: list[Insight] = []
    for theme, study_count in zip(themes, study_counts, strict=True):
        components = [
            ScoreComponent(name="study_count", value=clamp01(study_count), weight=0.5),
            ScoreComponent(
                name="recency", value=_recency(theme, corpus, corpus_max_year), weight=0.3
            ),
            ScoreComponent(
                name="source_diversity",
                value=_source_diversity(theme, corpus),
                weight=0.2,
            ),
        ]
        score = make_score(components, uncertainty=sample_uncertainty(theme.size()))
        label = theme.label or f"Theme {theme.id}"
        evidence = [EvidenceRef(paper_id=pid) for pid in theme.paper_ids[:_EVIDENCE_EVIDENCE_CAP]]
        statement = f"Evidence base for theme '{label}' is {score.interpretation}"
        # Fold uncertainty into the headline: a tiny-n theme can land in a high band
        # purely on a couple of components, which reads as authoritative. Annotate so
        # the band is not mistaken for a well-supported conclusion.
        if score.uncertainty is not None and score.uncertainty >= _HIGH_UNCERTAINTY:
            statement += f" (low confidence, n={theme.size()})"
        statement += "."
        insights.append(
            Insight(
                statement=statement,
                evidence=evidence,
                is_synthesized=True,
                tags=["evidence_strength"],
                score=score,
            )
        )
    return insights
