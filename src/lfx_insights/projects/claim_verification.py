"""Claim verification — verify scientific claims against the corpus using retrieval + Ollama."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus, GeneratedSection
    from lfx_insights.themes.discover import Embedder


class ClaimVerificationResult(BaseModel):
    """Result of verifying a single claim against the corpus."""

    claim: str
    status: str  # SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    relevant_papers: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class VerificationPrompt(BaseModel):
    """Structured response from the LLM for claim verification."""

    status: str = "INSUFFICIENT_EVIDENCE"
    confidence: float = 0.0
    supporting_points: list[str] = Field(default_factory=list)
    contradictory_points: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def _retrieve_evidence(
    claim: str,
    corpus: Corpus,
    embedder: Embedder,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """Retrieve the most similar passages from the corpus. Returns (paper_id, text, similarity)."""
    import numpy as np

    from lfx_insights.scoring.common import cosine

    if not corpus.papers:
        return []

    claim_vec = embedder.encode([claim])[0]
    paper_vecs = embedder.encode([p.text() for p in corpus.papers])

    similarities = []
    for i, paper in enumerate(corpus.papers):
        sim = float(cosine(claim_vec.tolist(), paper_vecs[i].tolist()))
        similarities.append((paper.id, paper.text(), sim))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


def _build_verification_prompt(claim: str, evidence: list[tuple[str, str, float]]) -> str:
    """Build a prompt for the LLM to classify the claim-evidence relationship."""
    evidence_block = "\n\n".join(
        f"[Paper {pid}] {text[:500]}..." if len(text) > 500 else f"[Paper {pid}] {text}"
        for pid, text, sim in evidence
    )

    return f"""Verify the following scientific claim against the retrieved evidence.

CLAIM: {claim}

RETRIEVED EVIDENCE:
{evidence_block}

Classify the relationship between the claim and the evidence:
- SUPPORTED: strong evidence directly supports the claim
- PARTIALLY_SUPPORTED: some evidence supports but is incomplete or mixed
- UNSUPPORTED: evidence does not support the claim
- CONTRADICTED: evidence contradicts the claim
- INSUFFICIENT_EVIDENCE: not enough evidence to judge

Return a JSON object with:
- status: one of the 5 statuses above
- confidence: 0.0-1.0
- supporting_points: list of supporting evidence summaries
- contradictory_points: list of contradictory evidence summaries
- limitations: list of limitations or caveats"""


def verify_claim(
    claim: str,
    corpus: Corpus,
    llm: LLMClient,
    embedder: Embedder,
    *,
    top_k: int = 10,
) -> ClaimVerificationResult:
    """Verify a scientific claim against the corpus.

    Steps:
    1. Embed claim, retrieve top_k most similar passages
    2. Use LLM to classify: supported/contradicted/inconclusive/unverifiable
    3. Return structured verification with evidence and limitations
    """
    evidence = _retrieve_evidence(claim, corpus, embedder, top_k=top_k)

    if not evidence:
        return ClaimVerificationResult(
            claim=claim,
            status="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
        )

    prompt = _build_verification_prompt(claim, evidence)
    result = llm.complete_structured(prompt, VerificationPrompt)

    return ClaimVerificationResult(
        claim=claim,
        status=result.status,
        confidence=result.confidence,
        supporting_evidence=result.supporting_points,
        contradictory_evidence=result.contradictory_points,
        relevant_papers=[pid for pid, _, _ in evidence],
        limitations=result.limitations,
    )


def verify_paragraph(
    paragraph: str,
    corpus: Corpus,
    llm: LLMClient,
    embedder: Embedder,
) -> list[ClaimVerificationResult]:
    """Extract and verify all scientific claims from a paragraph."""
    # Extract claims using a simple sentence splitter (deterministic)
    sentences = [s.strip() for s in paragraph.replace(".", ".\n").split("\n") if s.strip()]

    # Filter to likely scientific claims (contain assertion verbs or domain terms)
    claim_markers = [
        "shows", "demonstrates", "suggests", "indicates", "reveals", "found",
        "observed", "identified", "increased", "decreased", "associated",
        "correlated", "caused", "inhibited", "activated", "regulated",
    ]
    claims = []
    for s in sentences:
        lower = s.lower()
        if any(marker in lower for marker in claim_markers) and len(s.split()) > 5:
            claims.append(s)

    if not claims:
        # If no claims detected, verify the whole paragraph as a single claim
        claims = [paragraph]

    return [verify_claim(c, corpus, llm, embedder) for c in claims]


def verify_manuscript(
    sections: list[GeneratedSection],
    corpus: Corpus,
    llm: LLMClient,
    embedder: Embedder,
) -> list[ClaimVerificationResult]:
    """Verify all claims in a manuscript draft. Only verifies scientific sections."""
    verify_sections = {"Introduction", "Literature Review", "Discussion", "Results"}
    results: list[ClaimVerificationResult] = []

    for section in sections:
        if section.name in verify_sections:
            results.extend(verify_paragraph(section.text, corpus, llm, embedder))

    return results
