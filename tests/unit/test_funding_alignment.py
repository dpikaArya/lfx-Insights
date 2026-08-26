from __future__ import annotations

import pytest

from lfx_insights.models import Corpus, Paper, Theme
from lfx_insights.scoring.funding_alignment import (
    DEFAULT_PRIORITIES,
    PriorityArea,
    align_funding,
)

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(id="P1", title="Carbon emissions and climate warming"),
            Paper(id="P2", title="Renewable energy and solar grids"),
            Paper(id="P3", title="Clinical therapy for patient disease"),
        ],
    )


def test_climate_theme_aligns_to_climate_priority() -> None:
    theme = Theme(
        id=0, label="Climate", paper_ids=["P1"], keywords=["climate", "carbon", "emissions"]
    )
    insights = align_funding([theme], _corpus())
    assert len(insights) == 1
    ins = insights[0]
    assert "Climate & Sustainability" in ins.statement
    assert ins.tags == ["funding", "Climate & Sustainability"]


def test_insight_is_marked_synthesized() -> None:
    theme = Theme(id=0, label="Climate", paper_ids=["P1"], keywords=["climate"])
    ins = align_funding([theme], _corpus())[0]
    assert ins.is_synthesized is True


def test_multi_word_priority_keyword_matches_by_token_subset() -> None:
    # The climate theme should match the multi-word priority keyword
    # "climate change" because both of its tokens are present in the theme terms.
    custom = [
        PriorityArea(
            name="Climate Action",
            keywords=["climate change", "global warming", "carbon"],
        )
    ]
    theme = Theme(
        id=0,
        label="Climate",
        paper_ids=["P1"],
        keywords=["climate", "change", "carbon"],
    )
    ins = align_funding([theme], _corpus(), priorities=custom)[0]
    assert ins.tags == ["funding", "Climate Action"]
    assert ins.reasoning is not None
    # "climate change" matched via token-subset; "global warming" did not
    # (no "global" token despite "warming" being in the P1 title).
    assert "climate change" in ins.reasoning
    assert "carbon" in ins.reasoning
    assert "global warming" not in ins.reasoning


def test_multi_word_keyword_requires_all_tokens() -> None:
    # Only one of the two tokens of "machine learning" is present, so it must
    # not match (the buggy substring behavior would have matched nothing at all).
    custom = [PriorityArea(name="AI", keywords=["machine learning"])]
    theme = Theme(id=0, label="ML", paper_ids=["P1"], keywords=["machine", "vision"])
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="machine vision systems")])
    ins = align_funding([theme], corpus, priorities=custom)[0]
    assert ins.score is not None
    alignment = next(c for c in ins.score.components if c.name == "alignment")
    assert alignment.value == pytest.approx(0.0)
    assert ins.reasoning is not None
    assert "machine learning" not in ins.reasoning


def test_score_has_components_and_alignment_drives_value() -> None:
    theme = Theme(
        id=0, label="Climate", paper_ids=["P1"], keywords=["climate", "carbon", "emissions"]
    )
    ins = align_funding([theme], _corpus())[0]
    assert ins.score is not None
    names = {c.name for c in ins.score.components}
    assert names == {"alignment", "breadth"}
    # 4 of 5 Climate keywords matched (climate, carbon, emissions, warming).
    alignment = next(c for c in ins.score.components if c.name == "alignment")
    assert alignment.value == pytest.approx(0.8)
    assert ins.score.interpretation in {"high", "very high", "moderate"}


def test_relative_magnitude_strong_beats_weak() -> None:
    corpus = _corpus()
    strong = Theme(
        id=0, label="Climate", paper_ids=["P1"], keywords=["climate", "carbon", "emissions"]
    )
    weak = Theme(id=1, label="Misc", paper_ids=["P2"], keywords=["solar"])
    insights = align_funding([strong, weak], corpus)
    strong_val = insights[0].score.value  # type: ignore[union-attr]
    weak_val = insights[1].score.value  # type: ignore[union-attr]
    assert strong_val > weak_val


def test_evidence_per_paper_capped_at_ten() -> None:
    papers = [Paper(id=f"P{i}", title="climate carbon study") for i in range(15)]
    corpus = Corpus(kb_id="kb", papers=papers)
    theme = Theme(id=0, label="Big", paper_ids=[p.id for p in papers], keywords=["climate"])
    ins = align_funding([theme], corpus)[0]
    assert len(ins.evidence) == 10
    assert all(e.paper_id.startswith("P") for e in ins.evidence)


def test_reasoning_lists_matched_terms() -> None:
    theme = Theme(id=0, label="Climate", paper_ids=["P1"], keywords=["climate"])
    ins = align_funding([theme], _corpus())[0]
    assert ins.reasoning is not None
    assert "climate" in ins.reasoning


def test_titles_contribute_terms() -> None:
    # Theme has no keywords; alignment must come from the paper title tokens.
    theme = Theme(id=0, label="Energy", paper_ids=["P2"], keywords=[])
    ins = align_funding([theme], _corpus())[0]
    assert "Energy" in ins.statement
    assert ins.tags == ["funding", "Energy"]


def test_custom_priorities_respected() -> None:
    custom = [PriorityArea(name="Quantum", keywords=["quantum", "qubit"])]
    theme = Theme(id=0, label="QC", paper_ids=["P1"], keywords=["quantum", "qubit"])
    corpus = Corpus(kb_id="kb", papers=[Paper(id="P1", title="Quantum computing")])
    ins = align_funding([theme], corpus, priorities=custom)[0]
    assert ins.tags == ["funding", "Quantum"]


def test_empty_inputs_return_empty() -> None:
    corpus = _corpus()
    theme = Theme(id=0, label="Climate", paper_ids=["P1"], keywords=["climate"])
    assert align_funding([], corpus) == []
    assert align_funding([theme], Corpus(kb_id="kb", papers=[])) == []
    assert align_funding([theme], corpus, priorities=[]) == []


def test_default_priorities_shape() -> None:
    assert len(DEFAULT_PRIORITIES) == 5
    assert all(p.keywords and all(k == k.lower() for k in p.keywords) for p in DEFAULT_PRIORITIES)
