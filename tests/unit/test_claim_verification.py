"""Tests for claim verification."""

from __future__ import annotations

import pytest

from lfx_insights.llm.client import MockLLM
from lfx_insights.models import Corpus, Paper
from lfx_insights.projects.claim_verification import (
    ClaimVerificationResult,
    VerificationPrompt,
    verify_claim,
    verify_paragraph,
)
from lfx_insights.themes.discover import SimpleEmbedder

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="test",
        papers=[
            Paper(id="p1", title="Gene X inhibits pathway Y in cancer", abstract="We found that gene X strongly inhibits pathway Y."),
            Paper(id="p2", title="Drug A treats disease B", abstract="Drug A showed therapeutic effects on disease B."),
            Paper(id="p3", title="Unrelated topic entirely", abstract="This paper discusses quantum computing."),
        ],
    )


def test_verify_claim_supported() -> None:
    corpus = _corpus()
    llm = MockLLM(responder=lambda p, rt: rt(
        status="SUPPORTED",
        confidence=0.85,
        supporting_points=["Paper p1 confirms gene X inhibits pathway Y"],
        contradictory_points=[],
        limitations=["Small sample size in cited studies"],
    ))
    embedder = SimpleEmbedder()
    result = verify_claim("Gene X inhibits pathway Y", corpus, llm, embedder)
    assert result.status == "SUPPORTED"
    assert result.confidence == 0.85
    assert len(result.supporting_evidence) > 0
    assert len(result.relevant_papers) > 0


def test_verify_claim_insufficient_evidence() -> None:
    corpus = _corpus()
    llm = MockLLM()
    embedder = SimpleEmbedder()
    # Create a corpus with no relevant papers
    empty_corpus = Corpus(kb_id="empty", papers=[])
    result = verify_claim("Some random claim", empty_corpus, llm, embedder)
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == 0.0


def test_verify_paragraph() -> None:
    corpus = _corpus()
    llm = MockLLM(responder=lambda p, rt: rt(
        status="PARTIALLY_SUPPORTED",
        confidence=0.6,
        supporting_points=[],
        contradictory_points=[],
        limitations=[],
    ))
    embedder = SimpleEmbedder()
    results = verify_paragraph(
        "Gene X shows inhibition of pathway Y in cancer cells.",
        corpus,
        llm,
        embedder,
    )
    assert len(results) >= 1
    assert all(isinstance(r, ClaimVerificationResult) for r in results)


def test_verification_prompt_model() -> None:
    vp = VerificationPrompt(
        status="SUPPORTED",
        confidence=0.9,
        supporting_points=["point1"],
        contradictory_points=[],
        limitations=["limit1"],
    )
    assert vp.status == "SUPPORTED"
    assert vp.confidence == 0.9
