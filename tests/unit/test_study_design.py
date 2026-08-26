from __future__ import annotations

import pytest

from lfx_insights.lifescience.study_design import recommend_designs
from lfx_insights.models import Corpus, Paper, Theme

pytestmark = pytest.mark.unit

_CONFIRMATORY = {"prospective_cohort", "randomized_controlled_trial"}
_EXPLORATORY = {"exploratory_observational", "cross_sectional"}
_ALL_DESIGNS = {
    "exploratory_observational",
    "cross_sectional",
    "case_control",
    "prospective_cohort",
    "randomized_controlled_trial",
}


def _corpus() -> Corpus:
    """A large, recent, keyword-consistent cluster + a tiny, stale, off-topic one."""
    papers: list[Paper] = []
    # Big mature cluster: 10 recent papers (2023/2024) all mentioning the keyword.
    for i in range(10):
        year = 2024 if i % 2 == 0 else 2023
        papers.append(
            Paper(
                id=f"M{i}",
                title=f"insulin resistance cohort study {i}",
                year=year,
                abstract="insulin resistance metabolism in a large population",
            )
        )
    # Tiny sparse cluster: 1 old paper, keyword absent from its text.
    papers.append(
        Paper(
            id="S0",
            title="a single old anecdotal note",
            year=2008,
            abstract="an isolated observation with no shared vocabulary",
        )
    )
    return Corpus(kb_id="kb", papers=papers)


def _mature_theme() -> Theme:
    return Theme(
        id=0,
        label="Insulin resistance",
        paper_ids=[f"M{i}" for i in range(10)],
        keywords=["insulin resistance"],
    )


def _sparse_theme() -> Theme:
    return Theme(
        id=1,
        label="Anecdote",
        paper_ids=["S0"],
        keywords=["insulin resistance"],
    )


def test_mature_theme_gets_confirmatory_design() -> None:
    corpus = _corpus()
    insights = recommend_designs([_mature_theme(), _sparse_theme()], corpus)
    assert len(insights) == 2
    mature = insights[0]
    assert mature.score is not None
    assert mature.score.value >= 0.6
    # Large + recent + consistent -> confirmatory rung.
    design = mature.statement.split(": ")[1].rstrip(".")
    assert design in _CONFIRMATORY


def test_sparse_theme_gets_exploratory_design() -> None:
    corpus = _corpus()
    insights = recommend_designs([_mature_theme(), _sparse_theme()], corpus)
    sparse = insights[1]
    assert sparse.score is not None
    assert sparse.score.value < 0.4
    design = sparse.statement.split(": ")[1].rstrip(".")
    assert design in _EXPLORATORY


def test_statement_format_and_design_in_vocabulary() -> None:
    corpus = _corpus()
    insights = recommend_designs([_mature_theme()], corpus)
    ins = insights[0]
    design = insights[0].statement.split(": ")[1].rstrip(".")
    assert ins.statement == f"Recommended design for theme 'Insulin resistance': {design}."
    assert design in _ALL_DESIGNS


def test_tags_synthesized_and_components() -> None:
    corpus = _corpus()
    ins = recommend_designs([_mature_theme()], corpus)[0]
    assert ins.tags == ["study_design"]
    assert ins.is_synthesized is True
    assert ins.score is not None
    names = [c.name for c in ins.score.components]
    assert names == ["breadth", "recency", "consistency"]
    weights = {c.name: c.weight for c in ins.score.components}
    assert weights == {"breadth": 0.4, "recency": 0.3, "consistency": 0.3}
    assert ins.score.uncertainty is not None


def test_reasoning_mentions_maturity_and_inputs() -> None:
    corpus = _corpus()
    ins = recommend_designs([_mature_theme()], corpus)[0]
    assert ins.reasoning is not None
    reasoning = ins.reasoning.lower()
    assert "maturity" in reasoning
    # The three explicit inputs must be named (transparent, conjunctive rule).
    assert "breadth" in reasoning
    assert "recency" in reasoning
    assert "consistency" in reasoning
    # OBI study-design term must be attached.
    assert "obi:" in reasoning


def test_rct_carries_its_specific_obi_term() -> None:
    corpus = _corpus()
    ins = recommend_designs([_mature_theme()], corpus)[0]
    assert ins.reasoning is not None
    design = ins.statement.split(": ")[1].rstrip(".")
    if design == "randomized_controlled_trial":
        assert "OBI:0000471" in ins.reasoning


def test_maturity_is_conjunctive_not_or() -> None:
    """A large but STALE field must not reach confirmatory on count alone."""
    papers = [
        Paper(
            id=f"O{i}",
            title=f"old broad topic paper {i}",
            year=2001,
            abstract="broad topic with stable vocabulary",
        )
        for i in range(12)
    ]
    # Add one recent paper so the corpus max year is modern; the theme is stale.
    papers.append(Paper(id="REF", title="recent reference", year=2024))
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(
        id=0,
        label="Stale but large",
        paper_ids=[f"O{i}" for i in range(12)],
        keywords=["broad topic"],
    )
    ref_theme = Theme(id=1, label="ref", paper_ids=["REF"])
    insights = recommend_designs([theme, ref_theme], corpus)
    stale = insights[0]
    assert stale.score is not None
    # High breadth + high consistency but recency=0 -> blended below confirmatory.
    design = stale.statement.split(": ")[1].rstrip(".")
    assert design not in _CONFIRMATORY


def test_undated_corpus_cannot_be_confirmatory() -> None:
    """Unknown recency (no dated papers) must fail the gate, never confirmatory.

    A large, keyword-consistent, but entirely UNDATED corpus has high breadth and
    high consistency; recency reads neutral (no signal). The recency gate must treat
    that unknown as failing, so the design can never reach a confirmatory rung.
    """
    papers = [
        Paper(
            id=f"U{i}",
            title=f"insulin resistance cohort study {i}",
            year=None,
            abstract="insulin resistance metabolism in a large population",
        )
        for i in range(12)
    ]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(
        id=0,
        label="Undated but large",
        paper_ids=[f"U{i}" for i in range(12)],
        keywords=["insulin resistance"],
    )
    ins = recommend_designs([theme], corpus)[0]
    design = ins.statement.split(": ")[1].rstrip(".")
    assert design not in _CONFIRMATORY
    # The gate note must fire and attribute the demotion to unknown recency.
    assert ins.reasoning is not None
    assert "unknown" in ins.reasoning.lower()


def test_no_spurious_demotion_note_on_case_control() -> None:
    """case_control reached on its own merits must NOT carry a demotion note.

    A theme whose un-gated maturity lands it on case_control (never confirmatory)
    must not claim it was demoted off a confirmatory rung â€” the note is reserved for
    real demotions only. Recency here is high (recent papers), so the gate never
    fires; the rung is genuine and there is nothing to demote.
    """
    # Big sibling pushes the mid theme's breadth rank down; the mid theme is RECENT
    # (gate passes) but only partly keyword-consistent, so maturity lands squarely in
    # the case_control band on its own merits.
    big = [Paper(id=f"B{i}", title=f"big {i}", year=2024, abstract="x") for i in range(20)]
    mid = [
        Paper(
            id=f"C{i}",
            title=f"note {i}",
            year=2024,
            abstract="insulin resistance" if i < 2 else "unrelated text",
        )
        for i in range(6)
    ]
    corpus = Corpus(kb_id="kb", papers=big + mid)
    big_theme = Theme(id=2, label="Big", paper_ids=[f"B{i}" for i in range(20)], keywords=["zzz"])
    mid_theme = Theme(
        id=0,
        label="Mid",
        paper_ids=[f"C{i}" for i in range(6)],
        keywords=["insulin resistance"],
    )
    ins = recommend_designs([mid_theme, big_theme], corpus)[0]
    design = ins.statement.split(": ")[1].rstrip(".")
    # This input deterministically lands on case_control without any demotion.
    assert design == "case_control"
    assert ins.reasoning is not None
    # No demotion language: the rung was reached on merit, not forced by the gate.
    assert "demoted" not in ins.reasoning.lower()
    assert "below the gate" not in ins.reasoning.lower()
    assert "fails the gate" not in ins.reasoning.lower()


def test_stale_demotion_note_attributes_to_recency() -> None:
    """A genuine recency-driven demotion fires the note citing the gate value."""
    papers = [
        Paper(
            id=f"O{i}",
            title=f"broad topic paper {i}",
            year=2001,
            abstract="broad topic with stable vocabulary",
        )
        for i in range(12)
    ]
    papers.append(Paper(id="REF", title="recent reference", year=2024))
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(
        id=0,
        label="Stale but large",
        paper_ids=[f"O{i}" for i in range(12)],
        keywords=["broad topic"],
    )
    ref_theme = Theme(id=1, label="ref", paper_ids=["REF"])
    stale = recommend_designs([theme, ref_theme], corpus)[0]
    design = stale.statement.split(": ")[1].rstrip(".")
    assert stale.reasoning is not None
    if design == "case_control":
        # Real demotion: the note must fire and cite the recency gate, not "unknown".
        assert "below the gate" in stale.reasoning.lower()
        assert "unknown" not in stale.reasoning.lower()


def test_homogeneity_uses_word_boundary_not_substring() -> None:
    """A keyword embedded in a larger word must not count as present (consistency).

    With substring matching, keyword "meta" would falsely match "metabolism" in
    every paper, inflating consistency. Word-boundary matching must yield a strictly
    lower consistency (here: 0) than a theme whose keyword truly appears as a word.
    """
    embedded = [
        Paper(
            id=f"E{i}",
            title=f"metabolism study {i}",
            year=2024,
            abstract="a study of metabolism and metabolic flux",
        )
        for i in range(6)
    ]
    standalone = [
        Paper(
            id=f"W{i}",
            title=f"meta analysis {i}",
            year=2024,
            abstract="a meta analysis of pooled studies",
        )
        for i in range(6)
    ]
    corpus = Corpus(kb_id="kb", papers=embedded + standalone)
    embedded_theme = Theme(
        id=0,
        label="Embedded",
        paper_ids=[p.id for p in embedded],
        keywords=["meta"],
    )
    standalone_theme = Theme(
        id=1,
        label="Standalone",
        paper_ids=[p.id for p in standalone],
        keywords=["meta"],
    )
    insights = recommend_designs([embedded_theme, standalone_theme], corpus)
    assert insights[0].score is not None
    assert insights[1].score is not None
    emb_consistency = next(c.value for c in insights[0].score.components if c.name == "consistency")
    std_consistency = next(c.value for c in insights[1].score.components if c.name == "consistency")
    # "meta" embedded in "metabolism"/"metabolic" must NOT count.
    assert emb_consistency == 0.0
    # "meta" as a standalone word IS present.
    assert std_consistency == 1.0


def test_evidence_capped_at_ten() -> None:
    papers = [Paper(id=f"P{i}", title=f"p{i}", year=2024) for i in range(15)]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(id=0, label="Big", paper_ids=[p.id for p in papers])
    ins = recommend_designs([theme], corpus)[0]
    assert len(ins.evidence) == 10


def test_unknown_paper_ids_dropped_no_crash() -> None:
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="p1", year=2024)])
    theme = Theme(id=0, label="Partial", paper_ids=["P1", "GHOST"])
    ins = recommend_designs([theme], corpus)[0]
    assert [e.paper_id for e in ins.evidence] == ["P1"]


def test_label_falls_back_to_theme_id() -> None:
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="p1", year=2024)])
    theme = Theme(id=7, label="", paper_ids=["P1"])
    ins = recommend_designs([theme], corpus)[0]
    assert "theme 7" in ins.statement


def test_deterministic() -> None:
    corpus = _corpus()
    themes = [_mature_theme(), _sparse_theme()]
    first = recommend_designs(themes, corpus)
    second = recommend_designs(themes, corpus)
    assert [i.statement for i in first] == [i.statement for i in second]
    assert [i.score.value for i in first if i.score] == [i.score.value for i in second if i.score]


def test_empty_returns_empty() -> None:
    assert recommend_designs([], _corpus()) == []
    assert recommend_designs([_mature_theme()], Corpus(kb_id="kb", papers=[])) == []
