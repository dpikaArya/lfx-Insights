from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from consilium.generation.manuscript import (
    DEFAULT_SECTIONS,
    CitedRef,
    SectionDraft,
    draft_manuscript,
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
                title="Graph neural networks for molecules",
                doi="10.1/x",
                year=2020,
                authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
                abstract="Message passing networks predict molecular properties accurately.",
            ),
            Paper(
                id="W2",
                title="Generative models for de novo design",
                doi="10.1/y",
                year=2019,
                authors=[Author(name="Carol Lee")],
                abstract="Latent variable models generate novel drug-like molecules.",
            ),
        ],
    )


def _responder(*, leak_for: str | None = None) -> Callable[[str, type[BaseModel]], BaseModel]:
    def respond(prompt: str, model: type[BaseModel]) -> BaseModel:
        if leak_for is not None and leak_for in prompt:
            leaked: BaseModel = SectionDraft(
                text="Theme {top_theme} dominates [W1].",
                cited=[CitedRef(paper_id="W1", quote="Message passing networks")],
            )
            return leaked
        draft: BaseModel = SectionDraft(
            # W1 grounded -> rendered in-text; W2 ungrounded -> marker stripped.
            text="Recent work advances molecular modeling [W1]. Generative design helps [W2].",
            cited=[
                # Grounded: quote is verbatim in W1's abstract -> survives.
                CitedRef(paper_id="W1", quote="Message passing networks predict molecular"),
                # Real paper but made-up quote (not in W2's text) -> dropped.
                CitedRef(paper_id="W2", quote="Transformers dominate every benchmark"),
                # Real paper with no quote at all -> dropped.
                CitedRef(paper_id="W2", quote=""),
                # Unresolvable paper id -> dropped.
                CitedRef(paper_id="W404", quote="anything goes here"),
            ],
        )
        return draft

    return respond


def test_default_sections_and_quote_grounding() -> None:
    llm = MockLLM(responder=_responder())
    sections = draft_manuscript(_corpus(), llm)
    assert [s.name for s in sections] == list(DEFAULT_SECTIONS)
    for s in sections:
        # Only the grounded W1 citation survives; the made-up quote, the
        # quote-less cite, and the unresolvable id are all dropped.
        assert s.citations == ["W1"]
        assert "W2" not in s.citations
        assert "W404" not in s.citations
        # The grounded marker is rendered as an APA in-text citation; the
        # ungrounded W2 marker is stripped from the prose.
        assert s.text == (
            "Recent work advances molecular modeling (Smith & Jones, 2020). "
            "Generative design helps."
        )
        assert "[W1]" not in s.text and "[W2]" not in s.text
        assert s.provenance.model == "(see settings)"
        assert s.provenance.generated_by == "consilium"


def test_ungrounded_and_quoteless_citations_dropped() -> None:
    llm = MockLLM(responder=_responder())
    [section] = draft_manuscript(_corpus(), llm, sections=["introduction"])
    # W2 with a fabricated quote AND W2 with no quote are both dropped;
    # only the verbatim-grounded W1 remains, both in citations and in-text.
    assert section.citations == ["W1"]
    assert "(Smith & Jones, 2020)" in section.text


def test_crosssection_author_year_disambiguation() -> None:
    # Two papers by the same author in the same year, cited in DIFFERENT sections.
    # The document-level pass must letter them a/b consistently (APA orders the
    # colliding set by title: "Alpha study" -> a, "Beta study" -> b).
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="S1",
                title="Beta study",
                year=2020,
                authors=[Author(name="Alice Smith")],
                abstract="alpha findings here",
            ),
            Paper(
                id="S2",
                title="Alpha study",
                year=2020,
                authors=[Author(name="Alice Smith")],
                abstract="beta findings here",
            ),
        ],
    )

    def respond(prompt: str, model: type[BaseModel]) -> BaseModel:
        if "'introduction'" in prompt:
            return SectionDraft(
                text="Intro cites [S1].", cited=[CitedRef(paper_id="S1", quote="alpha findings")]
            )
        if "'methods'" in prompt:
            return SectionDraft(
                text="Methods cite [S2].", cited=[CitedRef(paper_id="S2", quote="beta findings")]
            )
        return SectionDraft(text="No cites.", cited=[])

    llm = MockLLM(responder=respond)
    sections = draft_manuscript(corpus, llm, sections=["introduction", "methods"])
    by_name = {s.name: s for s in sections}
    # S1 = "Beta study" -> 'b'; S2 = "Alpha study" -> 'a'; agreement across sections.
    assert "(Smith, 2020b)" in by_name["introduction"].text
    assert "(Smith, 2020a)" in by_name["methods"].text


def test_custom_sections_order_preserved() -> None:
    llm = MockLLM(responder=_responder())
    sections = draft_manuscript(_corpus(), llm, sections=["methods", "introduction"])
    assert [s.name for s in sections] == ["methods", "introduction"]


def test_leaked_section_is_skipped() -> None:
    # Only the 'discussion' section leaks an unformatted placeholder.
    llm = MockLLM(responder=_responder(leak_for="discussion"))
    sections = draft_manuscript(_corpus(), llm)
    names = [s.name for s in sections]
    assert "discussion" not in names
    assert names == ["introduction", "related_work", "methods"]


def test_nan_leak_is_skipped() -> None:
    def respond(prompt: str, model: type[BaseModel]) -> BaseModel:
        draft: BaseModel = SectionDraft(
            text="The effect size was NaN across studies.",
            cited=[CitedRef(paper_id="W1", quote="Message passing networks")],
        )
        return draft

    llm = MockLLM(responder=respond)
    sections = draft_manuscript(_corpus(), llm, sections=["methods"])
    assert sections == []


def test_offline_minimal_response_yields_no_citations() -> None:
    # No responder => MockLLM auto-builds a minimal SectionDraft (empty cited list).
    llm = MockLLM()
    sections = draft_manuscript(_corpus(), llm, sections=["introduction"])
    assert len(sections) == 1
    assert sections[0].citations == []


def test_prompt_lists_corpus_ids_and_titles_and_quote_instruction() -> None:
    llm = MockLLM(responder=_responder())
    draft_manuscript(_corpus(), llm, sections=["introduction"])
    prompt = llm.calls[0]
    assert "W1: Graph neural networks for molecules" in prompt
    assert "W2: Generative models for de novo design" in prompt
    assert "introduction" in prompt
    # Prompt instructs a verbatim quote per citation.
    assert "verbatim" in prompt.lower()
    # Prompt instructs inline square-bracket citation markers (not author/year prose).
    assert "square brackets" in prompt.lower()
    assert "[W1]" in prompt
