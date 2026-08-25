from __future__ import annotations

import pytest

from consilium.models import Corpus, Paper, Theme
from consilium.scoring.evidence_strength import score_evidence_strength

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            # Theme A: many, recent, multi-source.
            Paper(id="A1", title="A one", year=2024, source="biorxiv"),
            Paper(id="A2", title="A two", year=2023, source="pubmed"),
            Paper(id="A3", title="A three", year=2024, source="arxiv"),
            Paper(id="A4", title="A four", year=2022, source="biorxiv"),
            # Theme B: few, old, single-source.
            Paper(id="B1", title="B one", year=2005, source="legacy"),
            Paper(id="B2", title="B two", year=2006, source="legacy"),
        ],
    )


def _themes() -> list[Theme]:
    return [
        Theme(id=0, label="Strong", paper_ids=["A1", "A2", "A3", "A4"]),
        Theme(id=1, label="Weak", paper_ids=["B1", "B2"]),
    ]


def test_returns_one_insight_per_theme_with_tags_and_components() -> None:
    insights = score_evidence_strength(_themes(), _corpus())
    assert len(insights) == 2
    for ins in insights:
        assert ins.tags == ["evidence_strength"]
        assert ins.is_synthesized is True
        assert ins.score is not None
        names = [c.name for c in ins.score.components]
        assert names == ["study_count", "recency", "source_diversity"]
        # Honest score: band derived, uncertainty attached, no bare magic number.
        assert ins.score.uncertainty is not None
        assert ins.score.interpretation in ins.statement


def test_stronger_theme_scores_higher() -> None:
    insights = score_evidence_strength(_themes(), _corpus())
    strong, weak = insights[0], insights[1]
    assert strong.score is not None
    assert weak.score is not None
    assert strong.score.value > weak.score.value
    # Statement carries the theme label and the interpretation band.
    assert "Strong" in strong.statement
    assert strong.score.interpretation in strong.statement


def test_study_count_component_favors_larger_theme() -> None:
    insights = score_evidence_strength(_themes(), _corpus())
    strong_sc = next(c for c in insights[0].score.components if c.name == "study_count")  # type: ignore[union-attr]
    weak_sc = next(c for c in insights[1].score.components if c.name == "study_count")  # type: ignore[union-attr]
    # Larger theme normalizes to 1.0, smaller to 0.0.
    assert strong_sc.value == 1.0
    assert weak_sc.value == 0.0


def test_recency_anchored_to_max_year_and_handles_missing() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="P1", title="recent", year=2024, source="s"),
            Paper(id="P2", title="missing year", year=None, source="s"),
        ],
    )
    themes = [Theme(id=0, label="T", paper_ids=["P1", "P2"])]
    insights = score_evidence_strength(themes, corpus)
    recency = next(c for c in insights[0].score.components if c.name == "recency")  # type: ignore[union-attr]
    # max_year = 2024; P1 = clamp01((2024-2014)/10)=1.0, P2 missing -> 0.5; mean=0.75.
    assert recency.value == pytest.approx(0.75)


def test_source_diversity_component() -> None:
    insights = score_evidence_strength(_themes(), _corpus())
    strong_div = next(c for c in insights[0].score.components if c.name == "source_diversity")  # type: ignore[union-attr]
    weak_div = next(c for c in insights[1].score.components if c.name == "source_diversity")  # type: ignore[union-attr]
    # Size-independent: distinct sources / TARGET_SOURCES (=3), clamped.
    # Strong: 3 distinct sources -> 3/3 = 1.0; Weak: 1 distinct source -> 1/3.
    assert strong_div.value == pytest.approx(1.0)
    assert weak_div.value == pytest.approx(1.0 / 3.0)


def test_source_diversity_is_size_independent() -> None:
    # Two themes, same single source, different sizes. The larger one must NOT be
    # penalized for having more (same-source) papers: both score identically.
    corpus = Corpus(
        kb_id="kb",
        papers=[Paper(id=f"P{i}", title=f"p{i}", year=2020, source="solo") for i in range(8)],
    )
    small = Theme(id=0, label="small", paper_ids=["P0", "P1"])
    large = Theme(id=1, label="large", paper_ids=[f"P{i}" for i in range(8)])
    insights = score_evidence_strength([small, large], corpus)
    small_div = next(c for c in insights[0].score.components if c.name == "source_diversity")  # type: ignore[union-attr]
    large_div = next(c for c in insights[1].score.components if c.name == "source_diversity")  # type: ignore[union-attr]
    # Single source either way -> 1/3, regardless of theme size.
    assert small_div.value == pytest.approx(1.0 / 3.0)
    assert large_div.value == pytest.approx(1.0 / 3.0)
    assert small_div.value == large_div.value


def test_source_diversity_saturates_at_target() -> None:
    # More distinct sources than the target still clamps to 1.0 (no over-reward).
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="P1", title="p1", year=2020, source="a"),
            Paper(id="P2", title="p2", year=2020, source="b"),
            Paper(id="P3", title="p3", year=2020, source="c"),
            Paper(id="P4", title="p4", year=2020, source="d"),
        ],
    )
    themes = [Theme(id=0, label="T", paper_ids=["P1", "P2", "P3", "P4"])]
    insights = score_evidence_strength(themes, corpus)
    div = next(c for c in insights[0].score.components if c.name == "source_diversity")  # type: ignore[union-attr]
    assert div.value == pytest.approx(1.0)


def test_corpus_with_no_years_is_neutral_recency() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="X1", title="x1", year=None, source="s"),
            Paper(id="X2", title="x2", year=None, source="t"),
        ],
    )
    themes = [Theme(id=0, label="T", paper_ids=["X1", "X2"])]
    insights = score_evidence_strength(themes, corpus)
    recency = next(c for c in insights[0].score.components if c.name == "recency")  # type: ignore[union-attr]
    assert recency.value == pytest.approx(0.5)


def test_evidence_capped_at_ten() -> None:
    papers = [Paper(id=f"P{i}", title=f"p{i}", year=2020, source="s") for i in range(15)]
    corpus = Corpus(kb_id="kb", papers=papers)
    themes = [Theme(id=0, label="Big", paper_ids=[p.id for p in papers])]
    insights = score_evidence_strength(themes, corpus)
    assert len(insights[0].evidence) == 10
    assert insights[0].evidence[0].paper_id == "P0"


def test_label_fallback_when_unlabeled() -> None:
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="p", year=2020, source="s")])
    themes = [Theme(id=7, label="", paper_ids=["P1"])]
    insights = score_evidence_strength(themes, corpus)
    assert "Theme 7" in insights[0].statement


def test_missing_paper_in_corpus_does_not_crash() -> None:
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="p", year=2020, source="s")])
    # "GHOST" is referenced by the theme but absent from the corpus.
    themes = [Theme(id=0, label="T", paper_ids=["P1", "GHOST"])]
    insights = score_evidence_strength(themes, corpus)
    assert insights[0].score is not None
    recency = next(c for c in insights[0].score.components if c.name == "recency")
    # P1 -> 1.0, GHOST treated as neutral 0.5; mean = 0.75.
    assert recency.value == pytest.approx(0.75)


def test_single_paper_theme_statement_annotated_low_confidence() -> None:
    # A single-paper theme has max uncertainty (1/sqrt(1) == 1.0). Its headline band
    # must be flagged so it is not read as authoritative.
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="p", year=2024, source="s")])
    themes = [Theme(id=0, label="Lonely", paper_ids=["P1"])]
    insights = score_evidence_strength(themes, corpus)
    statement = insights[0].statement
    assert "(low confidence, n=1)" in statement
    # The interpretation band is still present, just qualified.
    assert insights[0].score is not None
    assert insights[0].score.interpretation in statement
    assert insights[0].score.uncertainty == pytest.approx(1.0)


def test_small_theme_caveated_large_theme_clean() -> None:
    # A theme of >= 4 papers keeps a clean headline; a tiny (<= 3 paper) theme is
    # annotated low-confidence so its band is not read as authoritative.
    insights = score_evidence_strength(_themes(), _corpus())
    strong = next(i for i in insights if "'Strong'" in i.statement)  # 4 papers
    weak = next(i for i in insights if "'Weak'" in i.statement)  # 2 papers
    assert "low confidence" not in strong.statement
    assert "low confidence, n=2" in weak.statement


def test_empty_themes_returns_empty() -> None:
    assert score_evidence_strength([], _corpus()) == []
