from __future__ import annotations

import pytest
from pydantic import BaseModel

from consilium.generation.grant import (
    DEFAULT_SECTIONS,
    CitedRef,
    SectionDraft,
    draft_grant,
)
from consilium.llm.client import MockLLM
from consilium.models import Author, Corpus, Paper

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecular property prediction",
                doi="10.1000/gnn",
                authors=[Author(name="Alice Smith")],
                year=2021,
                abstract="GNNs predict molecular properties.",
            ),
            Paper(
                id="W2",
                title="Deep generative models for de novo drug design",
                doi="10.1000/gen",
                authors=[Author(name="Bob Lee")],
                year=2022,
                abstract="Generative models propose novel molecules.",
            ),
        ],
    )


def _responder(text: str, cited: list[CitedRef]) -> MockLLM:
    def respond(prompt: str, response_model: type[BaseModel]) -> BaseModel:
        draft: BaseModel = SectionDraft(text=text, cited=list(cited))
        return draft

    return MockLLM(responder=respond)


def test_default_sections_drafted() -> None:
    corpus = _corpus()
    llm = _responder(
        "This proposal builds on prior work.",
        [
            CitedRef(paper_id="W1", quote="GNNs predict molecular properties."),
            CitedRef(paper_id="W2", quote="Generative models propose novel molecules."),
        ],
    )
    sections = draft_grant(corpus, llm)
    assert [s.name for s in sections] == list(DEFAULT_SECTIONS)
    assert all(s.text == "This proposal builds on prior work." for s in sections)
    assert all(s.provenance.generated_by == "consilium" for s in sections)
    assert all(s.provenance.model == "(see settings)" for s in sections)
    assert all(s.citations == ["W1", "W2"] for s in sections)


def test_custom_sections_drafted() -> None:
    corpus = _corpus()
    llm = _responder(
        "Specific aims here.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    sections = draft_grant(corpus, llm, sections=["specific_aims"])
    assert [s.name for s in sections] == ["specific_aims"]
    assert sections[0].citations == ["W1"]


def test_grounded_quote_survives() -> None:
    corpus = _corpus()
    llm = _responder(
        "Grounded prose.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    sections = draft_grant(corpus, llm, sections=["significance"])
    assert sections[0].citations == ["W1"]


def test_made_up_quote_and_missing_quote_dropped() -> None:
    corpus = _corpus()
    # W1: quote is a hallucinated paraphrase NOT present in the abstract -> drop.
    # W2: real corpus paper but NO supporting quote -> drop.
    llm = _responder(
        "Grounded prose.",
        [
            CitedRef(paper_id="W1", quote="Transformers dominate every benchmark."),
            CitedRef(paper_id="W2", quote=""),
        ],
    )
    sections = draft_grant(corpus, llm, sections=["significance"])
    assert sections[0].citations == []


def test_hallucinated_paper_id_dropped() -> None:
    corpus = _corpus()
    # W404 is not in the corpus -> must be dropped even with a quote.
    llm = _responder(
        "Grounded prose.",
        [
            CitedRef(paper_id="W1", quote="GNNs predict molecular properties."),
            CitedRef(paper_id="W404", quote="GNNs predict molecular properties."),
            CitedRef(paper_id="W2", quote="Generative models propose novel molecules."),
        ],
    )
    sections = draft_grant(corpus, llm, sections=["significance"])
    assert sections[0].citations == ["W1", "W2"]
    assert "W404" not in sections[0].citations


def test_leaked_placeholder_section_rejected() -> None:
    corpus = _corpus()
    llm = _responder(
        "Aim {top_theme} is great.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    sections = draft_grant(corpus, llm, sections=["specific_aims"])
    assert sections == []


def test_leaked_nan_section_rejected() -> None:
    corpus = _corpus()
    llm = _responder(
        "The effect size was NaN.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    sections = draft_grant(corpus, llm, sections=["approach"])
    assert sections == []


def test_doi_citation_grounded() -> None:
    corpus = _corpus()
    # Cite by DOI with a grounded quote -> resolved to the paper id.
    # A bogus DOI is dropped even though it carries a (foreign) quote.
    llm = _responder(
        "Cited by DOI.",
        [
            CitedRef(paper_id="10.1000/gnn", quote="GNNs predict molecular properties."),
            CitedRef(paper_id="10.1/bogus", quote="GNNs predict molecular properties."),
        ],
    )
    sections = draft_grant(corpus, llm, sections=["approach"])
    assert sections[0].citations == ["W1"]


def test_offline_minimal_response_yields_no_citations() -> None:
    corpus = _corpus()
    # No responder: MockLLM auto-builds a minimal SectionDraft (empty cited list).
    llm = MockLLM()
    sections = draft_grant(corpus, llm, sections=["significance"])
    assert len(sections) == 1
    assert sections[0].citations == []


def test_prompt_lists_corpus_ids() -> None:
    corpus = _corpus()
    llm = _responder(
        "Prose.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    draft_grant(corpus, llm, sections=["significance"])
    prompt = llm.calls[0]
    assert "W1: Graph neural networks for molecular property prediction" in prompt
    assert "W2: Deep generative models for de novo drug design" in prompt


def test_prompt_instructs_verbatim_quote() -> None:
    corpus = _corpus()
    llm = _responder(
        "Prose.",
        [CitedRef(paper_id="W1", quote="GNNs predict molecular properties.")],
    )
    draft_grant(corpus, llm, sections=["significance"])
    prompt = llm.calls[0]
    assert "verbatim" in prompt.lower()
