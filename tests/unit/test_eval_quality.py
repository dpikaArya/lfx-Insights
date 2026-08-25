"""Unit tests for the approximate-Prometheus answer-quality judge."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from consilium.eval.metrics.quality import QualityRubric, judge_quality
from consilium.eval.models import GeneratedAnswer
from consilium.llm.client import MockLLM

pytestmark = pytest.mark.unit


def _responder(rubric: QualityRubric):
    """Build a MockLLM responder that always returns ``rubric``."""

    def respond(prompt: str, model: type[BaseModel]) -> BaseModel:
        return rubric

    return respond


def test_normalizes_each_aspect_by_five() -> None:
    llm = MockLLM(responder=_responder(QualityRubric(organization=5, coverage=4, relevance=3)))
    answer = GeneratedAnswer(text="A structured, thorough, focused answer.")

    score = judge_quality(answer, reference="gold reference answer", llm=llm)

    assert score.organization == pytest.approx(1.0)
    assert score.coverage == pytest.approx(0.8)
    assert score.relevance == pytest.approx(0.6)
    assert score.judge == "llm"


def test_reference_none_path_works() -> None:
    llm = MockLLM(responder=_responder(QualityRubric(organization=3, coverage=3, relevance=5)))
    answer = GeneratedAnswer(text="An answer judged on its own merits.")

    score = judge_quality(answer, reference=None, llm=llm)

    assert score.organization == pytest.approx(0.6)
    assert score.coverage == pytest.approx(0.6)
    assert score.relevance == pytest.approx(1.0)
    assert score.judge == "llm"
    # With no reference the prompt must not advertise a Score-5 anchor.
    assert "REFERENCE answer (Score 5 anchor)" not in llm.calls[0]


def test_reference_present_prompt_carries_anchor() -> None:
    llm = MockLLM(responder=_responder(QualityRubric(organization=4, coverage=4, relevance=4)))
    answer = GeneratedAnswer(text="An answer with a gold reference available.")

    judge_quality(answer, reference="the gold standard", llm=llm)

    assert "REFERENCE answer (Score 5 anchor)" in llm.calls[0]
    assert "the gold standard" in llm.calls[0]


def test_out_of_range_high_score_is_clamped() -> None:
    llm = MockLLM(responder=_responder(QualityRubric(organization=7, coverage=5, relevance=5)))
    answer = GeneratedAnswer(text="Judge returned an over-range score.")

    score = judge_quality(answer, reference=None, llm=llm)

    # 7 clamps to 5 -> 1.0, not 1.4.
    assert score.organization == pytest.approx(1.0)
    assert score.coverage == pytest.approx(1.0)
    assert score.relevance == pytest.approx(1.0)


def test_out_of_range_low_score_is_clamped() -> None:
    llm = MockLLM(responder=_responder(QualityRubric(organization=0, coverage=-3, relevance=1)))
    answer = GeneratedAnswer(text="Judge returned an under-range score.")

    score = judge_quality(answer, reference=None, llm=llm)

    # 0 and -3 clamp to the 1.0 floor -> 0.2 (matching the /5 convention).
    assert score.organization == pytest.approx(0.2)
    assert score.coverage == pytest.approx(0.2)
    assert score.relevance == pytest.approx(0.2)
