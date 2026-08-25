from __future__ import annotations

import pytest
from pydantic import BaseModel

from consilium.generation.questions import (
    CitedRef,
    QuestionBatch,
    QuestionDraft,
    generate_questions,
)
from consilium.llm.client import MockLLM
from consilium.models import Corpus, Paper

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecules",
                abstract="Graph neural networks predict molecular properties on a single assay.",
            ),
            Paper(
                id="W2",
                title="Generative models for drug design",
                abstract="Generative models propose novel candidate molecules for drug design.",
            ),
        ],
    )


def _batch() -> QuestionBatch:
    return QuestionBatch(
        items=[
            QuestionDraft(
                question="Can GNNs generalize across chemical space?",
                rationale="W1 is limited to one assay.",
                novelty=0.9,
                feasibility=0.6,
                impact=0.8,
                citations=[
                    # Grounded: quote is verbatim in W1's abstract -> survives.
                    CitedRef(paper_id="W1", quote="predict molecular properties on a single assay"),
                    # Real paper but NO quote -> dropped.
                    CitedRef(paper_id="W2", quote=""),
                    # Hallucinated paper id -> dropped.
                    CitedRef(paper_id="W404", quote="anything"),
                ],
            ),
            QuestionDraft(
                question="How does generative design scale with data?",
                rationale="Builds on W2.",
                novelty=0.4,
                feasibility=0.5,
                impact=0.4,
                citations=[
                    # Real paper but the quote is MADE UP (not verbatim) -> dropped.
                    CitedRef(paper_id="W2", quote="generative models that never appear here"),
                ],
            ),
            QuestionDraft(
                question="What about {top_theme}?",  # leaked placeholder -> skipped
                rationale="bad",
                novelty=0.99,
                feasibility=0.99,
                impact=0.99,
            ),
        ]
    )


def _responder(prompt: str, model: type[BaseModel]) -> BaseModel:
    return _batch()


def test_questions_built_with_scored_components() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm)

    # Leaked question dropped: 3 drafts -> 2 questions.
    assert len(out) == 2
    for q in out:
        assert q.score is not None
        names = [c.name for c in q.score.components]
        assert names == ["novelty", "feasibility", "impact"]
        assert q.score.method == "weighted_mean"


def test_questions_sorted_by_score_desc() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm)

    scores = [q.score.value for q in out if q.score is not None]
    assert scores == sorted(scores, reverse=True)
    # The high-novelty/impact GNN question ranks first.
    assert out[0].question.startswith("Can GNNs")


def test_grounded_quote_citation_survives() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm)

    gnn = next(q for q in out if q.question.startswith("Can GNNs"))
    cited = [e.paper_id for e in gnn.evidence]
    # Only W1 grounds: W2 had no quote, W404 is hallucinated.
    assert cited == ["W1"]
    assert "W2" not in cited
    assert "W404" not in cited


def test_ungrounded_and_quoteless_citations_dropped() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm)

    # Question 2 cited W2 with a made-up quote -> no surviving evidence.
    gen = next(q for q in out if q.question.startswith("How does generative"))
    assert gen.evidence == []


def test_leaked_question_rejected() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm)

    assert all("{top_theme}" not in q.question for q in out)


def test_n_caps_output() -> None:
    llm = MockLLM(responder=_responder)
    out = generate_questions(_corpus(), llm, n=1)
    assert len(out) == 1


def test_floats_clamped_into_components() -> None:
    def responder(prompt: str, model: type[BaseModel]) -> BaseModel:
        return QuestionBatch(
            items=[
                QuestionDraft(
                    question="Clamp me.",
                    novelty=5.0,  # out of range -> clamp to 1.0
                    feasibility=-2.0,  # -> 0.0
                    impact=0.5,
                )
            ]
        )

    out = generate_questions(_corpus(), MockLLM(responder=responder))
    assert len(out) == 1
    assert out[0].score is not None
    by_name = {c.name: c.value for c in out[0].score.components}
    assert by_name["novelty"] == 1.0
    assert by_name["feasibility"] == 0.0
    assert by_name["impact"] == 0.5


def test_offline_minimal_response_yields_zero_citations() -> None:
    # MockLLM with no responder auto-builds a minimal QuestionBatch (no items),
    # so a grounded run completes without error and produces no questions.
    out = generate_questions(_corpus(), MockLLM())
    assert out == []
