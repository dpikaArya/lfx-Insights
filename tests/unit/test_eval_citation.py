"""Unit tests for the AutoAIS/ALCE citation precision/recall/F1 metric."""

from __future__ import annotations

import re

import pytest

from consilium.eval.metrics.citation import compute_citation_prf
from consilium.eval.models import GeneratedAnswer, RetrievedDoc

pytestmark = pytest.mark.unit

_TOKEN = re.compile(r"[a-z0-9]+")


class StubEntailer:
    """Deterministic entailment judge for tests.

    By default ``entails(premise, hypothesis)`` is True iff the hypothesis token
    set is a subset of the premise token set (lexical containment). A ``canned``
    map of ``(premise, hypothesis) -> bool`` overrides specific pairs, letting a
    test force exact necessity/sufficiency decisions.
    """

    name = "stub"

    def __init__(self, canned: dict[tuple[str, str], bool] | None = None) -> None:
        self.canned = canned or {}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_TOKEN.findall(text.lower()))

    def entails(self, premise: str, hypothesis: str) -> bool:
        if (premise, hypothesis) in self.canned:
            return self.canned[(premise, hypothesis)]
        return self._tokens(hypothesis) <= self._tokens(premise)


def _doc(doc_id: str, text: str) -> RetrievedDoc:
    return RetrievedDoc(doc_id=doc_id, text=text)


def test_two_supported_sentences_recall_one() -> None:
    """Both long sentences are entailed by their single cited doc -> recall 1.0."""
    docs = [
        _doc("d0", "graph neural networks accurately predict molecular properties from structure"),
        _doc("d1", "transformer language models achieve strong performance on reasoning"),
    ]
    answer = GeneratedAnswer(
        text=(
            "Graph neural networks accurately predict molecular properties [0]. "
            "Transformer language models achieve strong performance on reasoning [1]."
        ),
        docs=docs,
    )
    score = compute_citation_prf(answer, StubEntailer())
    assert score.n_sentences == 2
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.f1 == 1.0
    assert score.n_citations == 2
    assert score.judge == "stub"


def test_short_sentence_is_skipped() -> None:
    """A <50-char (stripped) sentence is not scored or counted."""
    docs = [
        _doc("d0", "graph neural networks accurately predict molecular properties from structure")
    ]
    answer = GeneratedAnswer(
        text=(
            "See below [0]. "
            "Graph neural networks accurately predict molecular properties from structure [0]."
        ),
        docs=docs,
    )
    score = compute_citation_prf(answer, StubEntailer())
    # Only the long second sentence is scored.
    assert score.n_sentences == 1
    assert score.recall == 1.0
    assert score.n_citations == 1


def test_citationless_sentence_inherits_previous_markers() -> None:
    """A long sentence with no markers inherits the previous scored sentence's markers."""
    # d0's token set is a superset of BOTH sentences' tokens, so the inherited
    # marker can entail the second (citation-less) sentence under the lexical judge.
    docs = [
        _doc(
            "d0",
            "graph neural networks accurately predict molecular properties reliably and they "
            "also estimate binding affinity for candidate drug compounds in practice",
        )
    ]
    answer = GeneratedAnswer(
        text=(
            "Graph neural networks accurately predict molecular properties reliably [0]. "
            "They also estimate binding affinity for candidate drug compounds in practice."
        ),
        docs=docs,
    )
    score = compute_citation_prf(answer, StubEntailer())
    assert score.n_sentences == 2
    # Both scored: first cites [0]; second inherits [0] and is also entailed by d0.
    assert score.recall == 1.0
    # Inherited markers count toward total_citations.
    assert score.n_citations == 2


def test_out_of_range_marker_recall_zero() -> None:
    """A marker outside [0, len(docs)) yields joint_entail 0 for that sentence."""
    # d0 contains all tokens of the first sentence (entailed); the second sentence
    # cites an out-of-range marker, so it scores 0 regardless of content.
    docs = [
        _doc("d0", "graph neural networks accurately predict molecular properties reliably here")
    ]
    answer = GeneratedAnswer(
        text=(
            "Graph neural networks accurately predict molecular properties reliably here [0]. "
            "Transformer language models achieve strong performance on hard reasoning tasks [5]."
        ),
        docs=docs,
    )
    score = compute_citation_prf(answer, StubEntailer())
    assert score.n_sentences == 2
    # First sentence supported (1.0), second has an out-of-range marker (0.0).
    assert score.recall == 0.5
    # Only the in-range marker counts toward total_citations.
    assert score.n_citations == 1
    # The one in-range, supported, single citation is precise.
    assert score.precision == 1.0


def test_overcited_citation_not_counted_in_precision() -> None:
    """A sentence citing two docs where only one is necessary -> over-cited one dropped.

    Under the lexical token-subset judge:
      - d0's token set is a superset of the sentence -> d0 alone entails it.
      - d1 is an unrelated passage -> d1 alone does NOT entail it.
      - joint (d0 + d1) entails -> supported, so recall 1.0.
      - For d0: sufficient (alone entails) -> precise.
      - For d1: not sufficient (alone fails); the others (d0) DO entail, so d1 is
        not necessary -> over-cited (not precise).
    So precision = 1/2 while recall == 1.0.
    """
    # d0 covers every sentence token; d1 shares no token with the sentence.
    d0 = _doc("d0", "graph neural networks accurately predict molecular properties reliably here")
    d1 = _doc("d1", "unrelated boilerplate about gardening tools and weather forecasts")
    answer = GeneratedAnswer(
        text="Graph neural networks accurately predict molecular properties reliably here [0][1].",
        docs=[d0, d1],
    )
    score = compute_citation_prf(answer, StubEntailer())
    assert score.n_sentences == 1
    assert score.recall == 1.0
    assert score.n_citations == 2
    assert score.precision == 0.5
    # f1 harmonic mean of 0.5 and 1.0
    assert score.f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_no_citations_anywhere_all_zero() -> None:
    """An answer with no markers -> precision 0, recall 0, f1 0, n_citations 0."""
    docs = [_doc("d0", "graph neural networks accurately predict molecular properties reliably")]
    answer = GeneratedAnswer(
        text="Graph neural networks accurately predict molecular properties reliably in practice.",
        docs=docs,
    )
    score = compute_citation_prf(answer, StubEntailer())
    assert score.n_sentences == 1
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.n_citations == 0
