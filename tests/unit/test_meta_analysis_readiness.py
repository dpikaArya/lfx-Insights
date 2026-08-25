from __future__ import annotations

import pytest

from consilium.models import Corpus, Paper, Theme
from consilium.scoring.meta_analysis_readiness import assess_meta_readiness

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            # Homogeneous, recent cluster: every paper mentions "meta" and "trial".
            Paper(id="H1", title="meta trial alpha", abstract="meta trial design", year=2023),
            Paper(id="H2", title="meta trial beta", abstract="meta trial outcome", year=2022),
            Paper(id="H3", title="meta trial gamma", abstract="meta trial results", year=2023),
            Paper(id="H4", title="meta trial delta", abstract="meta trial cohort", year=2024),
            Paper(id="H5", title="meta trial epsilon", abstract="meta trial sample", year=2023),
            # Tiny, heterogeneous, old single-study theme.
            Paper(id="S1", title="unrelated topic", abstract="nothing matching", year=2005),
        ],
    )


def _homogeneous_theme() -> Theme:
    return Theme(
        id=0,
        label="Homogeneous Recent",
        paper_ids=["H1", "H2", "H3", "H4", "H5"],
        keywords=["meta", "trial"],
    )


def _tiny_theme() -> Theme:
    return Theme(
        id=1,
        label="Tiny Heterogeneous",
        paper_ids=["S1"],
        keywords=["meta", "trial"],
    )


def test_returns_one_insight_per_theme_with_tags_and_score() -> None:
    corpus = _corpus()
    insights = assess_meta_readiness([_homogeneous_theme(), _tiny_theme()], corpus)
    assert len(insights) == 2
    for ins in insights:
        assert ins.tags == ["meta_analysis"]
        # The readiness insight is a synthesized composite, not a single extracted fact.
        assert ins.is_synthesized is True
        assert ins.score is not None
        # Honest score: components present, no bare magic number.
        names = {c.name for c in ins.score.components}
        assert names == {"n_studies", "homogeneity", "recency"}
        assert ins.score.uncertainty is not None
        assert ins.reasoning is not None
        assert "comparability" in ins.reasoning  # heuristic caveat surfaced


def test_larger_homogeneous_recent_scores_higher_than_tiny_heterogeneous() -> None:
    corpus = _corpus()
    big = assess_meta_readiness([_homogeneous_theme()], corpus)[0]
    small = assess_meta_readiness([_tiny_theme()], corpus)[0]
    assert big.score is not None and small.score is not None
    assert big.score.value > small.score.value
    # The big theme is recent + homogeneous + well-powered → at least moderate.
    assert big.score.interpretation in {"moderate", "high", "very high"}


def test_homogeneity_component_reflects_keyword_overlap() -> None:
    corpus = _corpus()
    big = assess_meta_readiness([_homogeneous_theme()], corpus)[0]
    assert big.score is not None
    homo = next(c for c in big.score.components if c.name == "homogeneity")
    # Every paper contains both keywords → full homogeneity.
    assert homo.value == 1.0
    small = assess_meta_readiness([_tiny_theme()], corpus)[0]
    assert small.score is not None
    small_homo = next(c for c in small.score.components if c.name == "homogeneity")
    # The lone paper matches neither keyword → zero homogeneity.
    assert small_homo.value == 0.0


def test_statement_and_evidence_cap() -> None:
    # 12 matching papers → evidence must be capped at 10.
    papers = [
        Paper(id=f"P{i}", title="meta trial", abstract="meta trial body", year=2023)
        for i in range(12)
    ]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(
        id=0,
        label="Big",
        paper_ids=[p.id for p in papers],
        keywords=["meta", "trial"],
    )
    ins = assess_meta_readiness([theme], corpus)[0]
    assert len(ins.evidence) == 10
    assert ins.score is not None
    assert ins.statement == (
        f"Meta-analysis readiness for theme 'Big' is {ins.score.interpretation}."
    )


def test_missing_years_are_neutral_and_do_not_crash() -> None:
    papers = [
        Paper(id="N1", title="meta trial", abstract="meta trial", year=None),
        Paper(id="N2", title="meta trial", abstract="meta trial", year=None),
    ]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(id=0, label="NoYears", paper_ids=["N1", "N2"], keywords=["meta"])
    ins = assess_meta_readiness([theme], corpus)[0]
    assert ins.score is not None
    recency = next(c for c in ins.score.components if c.name == "recency")
    # No year anywhere in the corpus → neutral recency.
    assert recency.value == 0.5


def test_homogeneity_uses_word_boundary_not_substring() -> None:
    # The keyword "meta" is a substring of "metabolism" / "metaphor" but is not
    # present as a whole word in either paper, so homogeneity must be 0.0 — a
    # naive substring match would wrongly inflate it to 1.0.
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="metabolism study", abstract="metabolic pathways", year=2023),
            Paper(id="W2", title="a metaphor", abstract="parametric models", year=2023),
        ],
    )
    theme = Theme(id=0, label="Substring", paper_ids=["W1", "W2"], keywords=["meta"])
    ins = assess_meta_readiness([theme], corpus)[0]
    assert ins.score is not None
    homo = next(c for c in ins.score.components if c.name == "homogeneity")
    assert homo.value == 0.0


def test_homogeneity_word_boundary_matches_standalone_keyword() -> None:
    # Same keyword "meta" as a standalone word is a genuine match → full homogeneity,
    # confirming the word-boundary regex still recognises real occurrences.
    corpus = Corpus(
        kb_id="kb",
        papers=[Paper(id="M1", title="a meta analysis", abstract="meta review", year=2023)],
    )
    theme = Theme(id=0, label="Standalone", paper_ids=["M1"], keywords=["meta"])
    ins = assess_meta_readiness([theme], corpus)[0]
    assert ins.score is not None
    homo = next(c for c in ins.score.components if c.name == "homogeneity")
    assert homo.value == 1.0


def test_no_keywords_yields_neutral_homogeneity() -> None:
    corpus = _corpus()
    theme = Theme(id=0, label="NoKw", paper_ids=["H1", "H2"], keywords=[])
    ins = assess_meta_readiness([theme], corpus)[0]
    assert ins.score is not None
    homo = next(c for c in ins.score.components if c.name == "homogeneity")
    assert homo.value == 0.5


def test_empty_inputs_return_empty_list() -> None:
    corpus = _corpus()
    assert assess_meta_readiness([], corpus) == []
    assert assess_meta_readiness([_homogeneous_theme()], Corpus(kb_id="kb", papers=[])) == []
