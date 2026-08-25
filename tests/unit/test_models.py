from __future__ import annotations

import pytest
from pydantic import ValidationError

from consilium.models import (
    Author,
    Corpus,
    Hypothesis,
    Insight,
    Paper,
    Score,
    ScoreComponent,
    Theme,
)

pytestmark = pytest.mark.unit


def test_corpus_lookup() -> None:
    p = Paper(id="W1", title="T", doi="10.x/y", authors=[Author(name="A B")], year=2020)
    c = Corpus(kb_id="kb1", papers=[p])
    assert c.by_id("W1") is p
    assert c.by_id("nope") is None
    assert c.dois() == ["10.x/y"]
    assert len(c) == 1


def test_score_requires_band_and_rejects_out_of_range() -> None:
    s = Score(
        value=0.4,
        components=[ScoreComponent(name="recency", value=0.4, weight=0.5)],
        method="weighted_mean",
        interpretation="low",
        uncertainty=0.1,
    )
    assert s.value == 0.4
    with pytest.raises(ValidationError):
        Score(
            value=1.5,
            components=[ScoreComponent(name="recency", value=0.4, weight=0.5)],
            method="x",
            interpretation="y",
        )


def test_component_less_score_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Score(value=0.4, method="weighted_mean", interpretation="low")
    assert "magic number" in str(excinfo.value)


def test_qualifier_synonyms_map_to_bucur_terms() -> None:
    cases = {
        "upregulates": "increases",
        "up-regulates": "increases",
        "downregulates": "decreases",
        "suppresses": "inhibits",
        "modulates": "is_associated_with",
        "induces": "causes",
    }
    for raw, expected in cases.items():
        h = Hypothesis(subject="X", qualifier=raw, object="Y", statement="X r Y")
        assert h.qualifier == expected


def test_qualifier_keeps_known_term_and_falls_back_when_unknown() -> None:
    known = Hypothesis(subject="X", qualifier="Causes", object="Y", statement="s")
    assert known.qualifier == "causes"
    unknown = Hypothesis(subject="X", qualifier="flibbertigibbets", object="Y", statement="s")
    assert unknown.qualifier == "is_associated_with"


def test_theme_and_insight() -> None:
    t = Theme(id=0, label="L", paper_ids=["W1"], keywords=["k"])
    assert t.size() == 1
    ins = Insight(statement="s", is_synthesized=True)
    assert ins.provenance.generated_by == "consilium"
