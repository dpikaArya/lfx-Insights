"""Heuristic meta-analysis readiness scoring per theme.

Honest by construction: this is a *heuristic* estimate. Method/outcome
comparability across studies has not been extracted â€” readiness is approximated
from study count, keyword homogeneity, and recency. Every :class:`Insight`
carries a :class:`Score` built via the shared kernel (components + band +
uncertainty), never a bare magic number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx_insights.corpus_features import keyword_homogeneity, max_year, theme_papers
from lfx_insights.models import EvidenceRef, Insight, ScoreComponent
from lfx_insights.scoring.common import clamp01, make_score, sample_uncertainty

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, Paper, Theme

_HEURISTIC_CAVEAT = (
    "Heuristic estimate: method/outcome comparability across studies has not been "
    "extracted; readiness approximated from study count, keyword homogeneity, and recency."
)

_TARGET_STUDIES = 5.0
_RECENCY_WINDOW = 10.0
_EVIDENCE_CAP = 10


def _n_studies_component(theme: Theme) -> ScoreComponent:
    return ScoreComponent(
        name="n_studies",
        value=clamp01(theme.size() / _TARGET_STUDIES),
        weight=0.4,
    )


def _homogeneity_component(theme: Theme, corpus: Corpus) -> ScoreComponent:
    # Word-boundary keyword share (neutral 0.5 with no keywords/papers) lives in
    # the shared corpus_features helper so it cannot drift across modules.
    return ScoreComponent(
        name="homogeneity",
        value=keyword_homogeneity(theme, corpus),
        weight=0.4,
    )


def _recency_component(papers: list[Paper], corpus_max_year: int | None) -> ScoreComponent:
    if corpus_max_year is None or not papers:
        value = 0.5
    else:
        floor = corpus_max_year - _RECENCY_WINDOW
        scores: list[float] = []
        for paper in papers:
            if paper.year is None:
                scores.append(0.5)
            else:
                scores.append(clamp01((paper.year - floor) / _RECENCY_WINDOW))
        value = clamp01(sum(scores) / len(scores))
    return ScoreComponent(name="recency", value=value, weight=0.2)


def assess_meta_readiness(themes: list[Theme], corpus: Corpus) -> list[Insight]:
    """Assess each theme's readiness for a quantitative meta-analysis.

    Deterministic. Returns one :class:`Insight` per theme (tagged
    ``meta_analysis``) whose :class:`Score` blends study count, keyword
    homogeneity, and recency. Returns ``[]`` for empty input.
    """
    if not themes or len(corpus) == 0:
        return []

    corpus_max_year = max_year(corpus)
    insights: list[Insight] = []
    for theme in themes:
        papers = theme_papers(theme, corpus)
        components = [
            _n_studies_component(theme),
            _homogeneity_component(theme, corpus),
            _recency_component(papers, corpus_max_year),
        ]
        score = make_score(components, uncertainty=sample_uncertainty(theme.size()))
        label = theme.label or f"theme {theme.id}"
        statement = f"Meta-analysis readiness for theme '{label}' is {score.interpretation}."
        reasoning = f"{theme.size()} studies in theme '{label}'. {_HEURISTIC_CAVEAT}"
        evidence = [EvidenceRef(paper_id=p.id) for p in papers[:_EVIDENCE_CAP]]
        insights.append(
            Insight(
                statement=statement,
                evidence=evidence,
                is_synthesized=True,
                reasoning=reasoning,
                tags=["meta_analysis"],
                score=score,
            )
        )
    return insights
