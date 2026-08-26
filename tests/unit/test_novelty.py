from __future__ import annotations

import pytest

from lfx_insights.models import Corpus, Paper, Theme
from lfx_insights.scoring.novelty import score_novelty

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            # Recent / emerging cluster (near max year 2024, rising counts).
            Paper(id="R1", title="recent one", year=2023),
            Paper(id="R2", title="recent two", year=2024),
            Paper(id="R3", title="recent three", year=2024),
            # Old / mature cluster (far from max year, declining).
            Paper(id="O1", title="old one", year=2010),
            Paper(id="O2", title="old two", year=2010),
            Paper(id="O3", title="old three", year=2011),
        ],
    )


def _recent_theme() -> Theme:
    return Theme(id=0, label="Recent", paper_ids=["R1", "R2", "R3"])


def _old_theme() -> Theme:
    return Theme(id=1, label="Old", paper_ids=["O1", "O2", "O3"])


def test_recent_theme_more_novel_than_old() -> None:
    corpus = _corpus()
    insights = score_novelty([_recent_theme(), _old_theme()], corpus)
    assert len(insights) == 2
    recent, old = insights[0], insights[1]
    assert recent.score is not None
    assert old.score is not None
    # Same size -> scarcity neutral for both; recency+growth decide the ordering.
    assert recent.score.value > old.score.value


def test_insight_shape_tags_evidence_and_components() -> None:
    corpus = _corpus()
    insights = score_novelty([_recent_theme()], corpus)
    ins = insights[0]
    assert ins.tags == ["novelty"]
    assert ins.is_synthesized is True
    assert (
        ins.statement
        == "Novelty of theme 'Recent' is "
        + str(ins.score.interpretation if ins.score else "")
        + "."
    )
    assert ins.score is not None
    names = [c.name for c in ins.score.components]
    assert names == ["recency_share", "growth", "scarcity"]
    weights = {c.name: c.weight for c in ins.score.components}
    assert weights == {"recency_share": 0.4, "growth": 0.3, "scarcity": 0.3}
    assert ins.score.uncertainty is not None
    # One EvidenceRef per supporting paper.
    assert [e.paper_id for e in ins.evidence] == ["R1", "R2", "R3"]


def test_recency_share_component_values() -> None:
    corpus = _corpus()
    insights = score_novelty([_recent_theme(), _old_theme()], corpus)
    recent_comps = {c.name: c.value for c in insights[0].score.components}  # type: ignore[union-attr]
    old_comps = {c.name: c.value for c in insights[1].score.components}  # type: ignore[union-attr]
    # max_year=2024, cutoff=2022 -> all 3 recent papers in window.
    assert recent_comps["recency_share"] == 1.0
    # Old papers (2010, 2010, 2011) are all before the cutoff.
    assert old_comps["recency_share"] == 0.0


def test_scarcity_favors_smaller_theme() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="B1", title="b1", year=2024),
            Paper(id="B2", title="b2", year=2024),
            Paper(id="B3", title="b3", year=2024),
            Paper(id="S1", title="s1", year=2024),
        ],
    )
    big = Theme(id=0, label="Big", paper_ids=["B1", "B2", "B3"])
    small = Theme(id=1, label="Small", paper_ids=["S1"])
    insights = score_novelty([big, small], corpus)
    big_scarcity = next(
        c.value
        for c in insights[0].score.components
        if c.name == "scarcity"  # type: ignore[union-attr]
    )
    small_scarcity = next(
        c.value
        for c in insights[1].score.components
        if c.name == "scarcity"  # type: ignore[union-attr]
    )
    # Smaller theme -> higher scarcity -> more novel.
    assert small_scarcity > big_scarcity


def test_missing_years_neutral_no_crash() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="P1", title="no year a"),
            Paper(id="P2", title="no year b"),
        ],
    )
    theme = Theme(id=0, label="Undated", paper_ids=["P1", "P2"])
    insights = score_novelty([theme], corpus)
    assert len(insights) == 1
    comps = {c.name: c.value for c in insights[0].score.components}  # type: ignore[union-attr]
    # No dating signal -> recency NEUTRAL 0.5 (project convention), growth neutral.
    assert comps["recency_share"] == 0.5
    assert comps["growth"] == 0.5


def test_growth_neutral_with_single_distinct_year() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="X1", title="x1", year=2024),
            Paper(id="X2", title="x2", year=2024),
        ],
    )
    theme = Theme(id=0, label="Flat", paper_ids=["X1", "X2"])
    insights = score_novelty([theme], corpus)
    growth = next(
        c.value
        for c in insights[0].score.components
        if c.name == "growth"  # type: ignore[union-attr]
    )
    assert growth == 0.5


def test_growth_is_relative_not_saturating_on_trivial_rise() -> None:
    # A trivial absolute rise (1 paper in 2023 -> 2 in 2024) is a 100% relative
    # increase but must NOT max out: a far larger surge must score strictly higher.
    small_corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="A0", title="a0", year=2023),
            Paper(id="A1", title="a1", year=2024),
            Paper(id="A2", title="a2", year=2024),
        ],
    )
    small = Theme(id=0, label="SmallRise", paper_ids=["A0", "A1", "A2"])
    small_growth = next(
        c.value
        for c in score_novelty([small], small_corpus)[0].score.components  # type: ignore[union-attr]
        if c.name == "growth"
    )
    # 1 -> 2 is a relative growth of 1.0; squashed through tanh it sits well below 1.
    assert 0.5 < small_growth < 1.0

    big_papers = [Paper(id="B0", title="b0", year=2023)]
    big_papers += [Paper(id=f"C{i}", title=f"c{i}", year=2024) for i in range(50)]
    big_corpus = Corpus(kb_id="kb", papers=big_papers)
    big = Theme(id=0, label="BigSurge", paper_ids=[p.id for p in big_papers])
    big_growth = next(
        c.value
        for c in score_novelty([big], big_corpus)[0].score.components  # type: ignore[union-attr]
        if c.name == "growth"
    )
    # A 1 -> 50 surge is a far larger relative growth -> strictly more growth.
    assert big_growth > small_growth


def test_decline_scores_below_neutral_growth() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="D0", title="d0", year=2023),
            Paper(id="D1", title="d1", year=2023),
            Paper(id="D2", title="d2", year=2023),
            Paper(id="D3", title="d3", year=2024),
        ],
    )
    declining = Theme(id=0, label="Declining", paper_ids=["D0", "D1", "D2", "D3"])
    growth = next(
        c.value
        for c in score_novelty([declining], corpus)[0].score.components  # type: ignore[union-attr]
        if c.name == "growth"
    )
    # 3 -> 1 is a negative relative trend -> growth pushed below the neutral 0.5.
    assert growth < 0.5


def test_reasoning_discloses_proxy_nature() -> None:
    corpus = _corpus()
    insights = score_novelty([_recent_theme()], corpus)
    reasoning = insights[0].reasoning
    assert reasoning is not None
    # The disclosure must name the proxy and disclaim intrinsic originality.
    assert "recency" in reasoning.lower()
    assert "originality" in reasoning.lower()
    assert "not a measure of intrinsic originality" in reasoning.lower()


def test_evidence_capped_at_ten() -> None:
    papers = [Paper(id=f"P{i}", title=f"p{i}", year=2024) for i in range(15)]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(id=0, label="Big", paper_ids=[p.id for p in papers])
    insights = score_novelty([theme], corpus)
    assert len(insights[0].evidence) == 10


def test_deterministic() -> None:
    corpus = _corpus()
    themes = [_recent_theme(), _old_theme()]
    first = score_novelty(themes, corpus)
    second = score_novelty(themes, corpus)
    assert [i.score.value for i in first if i.score] == [i.score.value for i in second if i.score]


def test_empty_returns_empty() -> None:
    assert score_novelty([], _corpus()) == []
    assert score_novelty([_recent_theme()], Corpus(kb_id="kb", papers=[])) == []
