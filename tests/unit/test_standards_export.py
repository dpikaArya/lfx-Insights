from __future__ import annotations

import re

import pytest

from consilium.models import Author, Corpus, EvidenceRef, Insight, Paper
from consilium.standards.astra_export import astra_available, insights_to_collection
from consilium.standards.indicium_export import indicium_available, sources_to_indicium

pytestmark = pytest.mark.unit


def test_availability_flags_are_bool() -> None:
    assert isinstance(indicium_available(), bool)
    assert isinstance(astra_available(), bool)


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[Paper(id="W1", title="T", doi="10.x/y", authors=[Author(name="A B")])],
    )


def test_sources_to_indicium_shape() -> None:
    out = sources_to_indicium(_corpus())
    assert out == [{"identifier": "10.x/y", "doi": "10.x/y", "title": "T", "authors": ["A B"]}]


def test_insights_to_collection_shape() -> None:
    ins = Insight(
        statement="A gap exists in X",
        evidence=[EvidenceRef(paper_id="W1", quote="q", location="abstract")],
        is_synthesized=True,
        tags=["gap"],
    )
    coll = insights_to_collection([ins], title="Findings")
    # ASTRA InsightCollection: insights + notes (no "title" slot); Insight uses
    # claim/derived (not statement/is_synthesized); evidence uses doi (not paper_id).
    assert coll["notes"] == "Findings"
    assert coll["insights"][0]["claim"] == "A gap exists in X"
    assert coll["insights"][0]["derived"] is True
    assert coll["insights"][0]["evidence"][0]["doi"] == "W1"


def test_astra_ids_match_linkml_id_pattern() -> None:
    # ASTRA LinkML id pattern: lowercase start, then [a-z0-9_] only (no hyphens).
    id_pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    ins = Insight(
        statement="A gap exists in X",
        evidence=[
            EvidenceRef(paper_id="W1", quote="q", location="abstract"),
            EvidenceRef(paper_id="W2", quote="r", location="intro"),
        ],
        is_synthesized=True,
        tags=["gap"],
    )
    coll = insights_to_collection([ins])
    insight = coll["insights"][0]
    assert insight["id"] == "insight_0"
    assert "-" not in insight["id"]
    assert id_pattern.match(insight["id"])
    for k, ev in enumerate(insight["evidence"]):
        assert ev["id"] == f"ev_0_{k}"
        assert "-" not in ev["id"]
        assert id_pattern.match(ev["id"])
