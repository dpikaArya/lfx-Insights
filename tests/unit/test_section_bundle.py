from __future__ import annotations

import pytest

from consilium.generation.common import build_section_bundle
from consilium.models import Corpus, GeneratedSection, Paper, SectionBundle

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="Alpha", year=2020),
            Paper(id="W2", title="Beta", year=2021),
            Paper(id="W3", title="Gamma uncited", year=2022),
        ],
    )


def test_bundle_dedups_in_first_citation_order() -> None:
    sections = [
        GeneratedSection(name="introduction", text="x", citations=["W2", "W1"]),
        GeneratedSection(name="discussion", text="y", citations=["W1", "W2"]),
    ]
    bundle = build_section_bundle("Manuscript Draft", sections, _corpus())
    assert isinstance(bundle, SectionBundle)
    assert bundle.title == "Manuscript Draft"
    assert bundle.sections == sections
    assert [p.id for p in bundle.references] == ["W2", "W1"]


def test_bundle_excludes_uncited_and_unresolvable_ids() -> None:
    sections = [GeneratedSection(name="methods", text="z", citations=["W1", "MISSING"])]
    bundle = build_section_bundle("Manuscript Draft", sections, _corpus())
    assert [p.id for p in bundle.references] == ["W1"]
