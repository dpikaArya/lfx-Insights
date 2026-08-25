from __future__ import annotations

import pytest
from pydantic import BaseModel

from consilium.generation.reviewer_sim import (
    ReviewBatch,
    ReviewItem,
    simulate_review,
)
from consilium.llm.client import MockLLM
from consilium.models import Author, Corpus, GeneratedSection, Paper

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecules",
                doi="10.1/x",
                authors=[Author(name="Alice Smith")],
                year=2021,
                source="Journal of ML",
            ),
            Paper(
                id="W2",
                title="Generative models for drug design",
                doi="10.1/y",
                authors=[Author(name="Bob Lee")],
                year=2022,
            ),
        ],
    )


def _sections() -> list[GeneratedSection]:
    return [
        GeneratedSection(
            name="Introduction",
            text="Graph neural networks dominate molecular property prediction.",
            citations=["W1"],
        ),
        GeneratedSection(
            name="Methods",
            text="We trained a model on a dataset.",
            citations=[],
        ),
    ]


def _responder() -> MockLLM:
    batch = ReviewBatch(
        items=[
            ReviewItem(
                severity="MAJOR",
                section="Methods",
                comment="The dataset is never described; provenance is unclear.",
                suggestion="Specify the dataset, size, and splits.",
            ),
            ReviewItem(
                severity="praise",
                section="Introduction",
                comment="Clear framing of the problem.",
                suggestion=None,
            ),
            ReviewItem(
                severity="catastrophic",
                section="Introduction",
                comment="Overclaims dominance without a baseline comparison.",
                suggestion="Soften the claim or add evidence.",
            ),
            ReviewItem(
                severity="minor",
                section="Methods",
                comment="Reported accuracy is {accuracy}.",
                suggestion="Fill in the metric.",
            ),
        ]
    )
    return MockLLM(responder=lambda prompt, model: batch)


def test_builds_review_comments() -> None:
    comments = simulate_review(_sections(), _corpus(), _responder())
    # Four items returned, one dropped for leaked '{accuracy}' -> three comments.
    assert len(comments) == 3
    sections = {c.section for c in comments}
    assert sections == {"Methods", "Introduction"}


def test_severity_normalized() -> None:
    comments = simulate_review(_sections(), _corpus(), _responder())
    by_comment = {c.comment: c.severity for c in comments}
    # "MAJOR" -> "major" (cased + valid)
    assert by_comment["The dataset is never described; provenance is unclear."] == "major"
    # "praise" stays
    assert by_comment["Clear framing of the problem."] == "praise"
    # unknown "catastrophic" -> default "minor"
    assert by_comment["Overclaims dominance without a baseline comparison."] == "minor"
    assert all(c.severity in {"major", "minor", "praise"} for c in comments)


def test_leaked_comment_skipped() -> None:
    comments = simulate_review(_sections(), _corpus(), _responder())
    assert all("{accuracy}" not in c.comment for c in comments)
    assert all("{" not in c.comment for c in comments)


def test_suggestion_preserved() -> None:
    comments = simulate_review(_sections(), _corpus(), _responder())
    praise = next(c for c in comments if c.severity == "praise")
    assert praise.suggestion is None
    major = next(c for c in comments if c.severity == "major")
    assert major.suggestion == "Specify the dataset, size, and splits."


def test_prompt_lists_corpus_and_sections() -> None:
    llm = _responder()
    simulate_review(_sections(), _corpus(), llm)
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    # Corpus papers listed as "<id>: <title>" and instruction to cite only them.
    assert "W1: Graph neural networks for molecules" in prompt
    assert "W2: Generative models for drug design" in prompt
    assert "cite ONLY these ids" in prompt
    # Section names + text present.
    assert "Introduction" in prompt
    assert "Methods" in prompt
    assert "We trained a model on a dataset." in prompt


def test_empty_sections_returns_empty() -> None:
    called = {"n": 0}

    class _NoCallLLM:
        def complete_structured(
            self,
            prompt: str,
            response_model: type[BaseModel],
            *,
            temperature: float | None = None,
        ) -> BaseModel:
            called["n"] += 1
            return ReviewBatch(items=[])

    out = simulate_review([], _corpus(), _NoCallLLM())
    assert out == []
    # No sections -> no LLM call.
    assert called["n"] == 0
