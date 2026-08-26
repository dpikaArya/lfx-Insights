from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from lfx_insights.eval.answer import AnswerDraft, CitedMarker, answer_question
from lfx_insights.llm.client import MockLLM
from lfx_insights.models import Corpus, Paper

pytestmark = pytest.mark.unit

# Verbatim substrings of each paper's text() ("title\n\nabstract") used as grounded
# quotes; keeping them as module constants makes the verbatim contract explicit.
_DOC0_QUOTE = "Message passing networks predict molecular properties accurately."
_DOC1_QUOTE = "Latent variable models generate novel drug-like molecules."


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecules",
                doi="10.1/x",
                abstract="Message passing networks predict molecular properties accurately.",
            ),
            Paper(
                id="W2",
                title="Generative models for de novo design",
                doi="10.1/y",
                abstract="Latent variable models generate novel drug-like molecules.",
            ),
        ],
    )


def _responder(draft: AnswerDraft) -> Callable[[str, type[BaseModel]], BaseModel]:
    def respond(prompt: str, model: type[BaseModel]) -> BaseModel:
        return draft

    return respond


def test_both_grounded_citations_kept() -> None:
    draft = AnswerDraft(
        text="Claim one [0]. Claim two [1].",
        cited=[
            CitedMarker(marker=0, paper_id="W1", quote=_DOC0_QUOTE),
            CitedMarker(marker=1, paper_id="W2", quote=_DOC1_QUOTE),
        ],
    )
    llm = MockLLM(responder=_responder(draft))
    ans = answer_question("What models design molecules?", _corpus(), llm)

    assert len(ans.docs) == 2
    assert [c.marker for c in ans.citations] == [0, 1]
    assert [c.doc_id for c in ans.citations] == ["D0", "D1"]
    # Both inline markers survive the grounding gate.
    assert "[0]" in ans.text
    assert "[1]" in ans.text
    assert ans.text == "Claim one [0]. Claim two [1]."
    assert ans.provenance.generated_by == "lfx-insights"
    assert ans.provenance.model == "(eval)"


def test_ungrounded_quote_dropped_and_marker_stripped() -> None:
    draft = AnswerDraft(
        text="Claim one [0]. Claim two [1].",
        cited=[
            CitedMarker(marker=0, paper_id="W1", quote=_DOC0_QUOTE),
            # Quote not present anywhere in doc1's text -> dropped.
            CitedMarker(marker=1, paper_id="W2", quote="Transformers dominate every benchmark"),
        ],
    )
    llm = MockLLM(responder=_responder(draft))
    ans = answer_question("q", _corpus(), llm)

    assert [c.marker for c in ans.citations] == [0]
    # The grounded marker stays; the ungrounded one is stripped from prose.
    assert "[0]" in ans.text
    assert "[1]" not in ans.text
    assert ans.text == "Claim one [0]. Claim two ."


def test_out_of_range_marker_dropped_and_stripped() -> None:
    draft = AnswerDraft(
        text="Claim one [0]. Spurious [5].",
        cited=[
            CitedMarker(marker=0, paper_id="W1", quote=_DOC0_QUOTE),
            # Marker 5 is beyond the 2-doc corpus, even with a real-looking quote.
            CitedMarker(marker=5, paper_id="W2", quote=_DOC1_QUOTE),
        ],
    )
    llm = MockLLM(responder=_responder(draft))
    ans = answer_question("q", _corpus(), llm)

    assert [c.marker for c in ans.citations] == [0]
    assert "[0]" in ans.text
    assert "[5]" not in ans.text
    assert ans.text == "Claim one [0]. Spurious ."


def test_empty_corpus_is_closed_book() -> None:
    draft = AnswerDraft(
        text="Closed book claim [0] with no sources [1].",
        cited=[CitedMarker(marker=0, paper_id="W1", quote="anything")],
    )
    llm = MockLLM(responder=_responder(draft))
    ans = answer_question("q", Corpus(kb_id="empty", papers=[]), llm)

    assert ans.docs == []
    assert ans.citations == []
    # No docs => every marker is out of range and stripped.
    assert "[0]" not in ans.text
    assert "[1]" not in ans.text
    assert ans.text == "Closed book claim with no sources ."


def test_max_docs_truncates_corpus() -> None:
    draft = AnswerDraft(
        text="Only first source [0].",
        cited=[CitedMarker(marker=0, paper_id="W1", quote=_DOC0_QUOTE)],
    )
    llm = MockLLM(responder=_responder(draft))
    ans = answer_question("q", _corpus(), llm, max_docs=1)

    assert len(ans.docs) == 1
    assert ans.docs[0].doc_id == "D0"
    assert [c.marker for c in ans.citations] == [0]


def test_offline_minimal_response_yields_no_citations() -> None:
    # No responder => MockLLM auto-builds a minimal AnswerDraft (empty text/cited).
    llm = MockLLM()
    ans = answer_question("q", _corpus(), llm)
    assert ans.citations == []
    assert len(ans.docs) == 2


def test_prompt_lists_numbered_sources_and_quote_instruction() -> None:
    draft = AnswerDraft(text="x", cited=[])
    llm = MockLLM(responder=_responder(draft))
    answer_question("How do GNNs work?", _corpus(), llm)
    prompt = llm.calls[0]

    assert "[0] Graph neural networks for molecules" in prompt
    assert "[1] Generative models for de novo design" in prompt
    assert "How do GNNs work?" in prompt
    assert "verbatim" in prompt.lower()


class _StubEntailer:
    """Entails a sentence only if the premise contains the token 'alpha'."""

    name = "stub"

    def entails(self, premise: str, hypothesis: str) -> bool:
        return "alpha" in premise.lower()


def test_entailment_gate_drops_unsupported_citation() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="Alpha doc", abstract="alpha supporting evidence here"),
            Paper(id="W2", title="Beta doc", abstract="unrelated beta content"),
        ],
    )

    def responder(prompt: str, model: type[BaseModel]) -> BaseModel:
        return AnswerDraft(
            text="This is the first grounded claim sentence about the topic [0]. "
            "This is a second claim sentence on the topic that is unsupported [1].",
            cited=[
                CitedMarker(marker=0, paper_id="W1", quote=""),
                CitedMarker(marker=1, paper_id="W2", quote=""),
            ],
        )

    answer = answer_question("q", corpus, MockLLM(responder=responder), entailer=_StubEntailer())
    markers = {c.marker for c in answer.citations}
    assert markers == {0}  # [0] entailed (doc has 'alpha'); [1] dropped
    assert "[0]" in answer.text and "[1]" not in answer.text


def test_cite_as_you_write_attributes_uncited_sentence() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="Alpha doc", abstract="alpha supporting evidence here"),
            Paper(id="W2", title="Beta doc", abstract="unrelated beta content"),
        ],
    )

    def responder(prompt: str, model: type[BaseModel]) -> BaseModel:
        # A long, clearly-scorable claim sentence with NO citation at all.
        return AnswerDraft(
            text="This is a sufficiently long uncited claim sentence about the topic at hand.",
            cited=[],
        )

    answer = answer_question("q", corpus, MockLLM(responder=responder), entailer=_StubEntailer())
    # The first entailing source (doc0, which contains 'alpha') is attributed.
    assert {c.marker for c in answer.citations} == {0}
    assert "[0]" in answer.text
