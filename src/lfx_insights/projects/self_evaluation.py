"""Self-evaluation — evidence-grounded evaluation of research outputs.

Uses retrieval, claim verification, and citation inspection to produce
structured evaluation results. Task-specific profiles prevent unnecessary
LLM calls and reduce runtime.

Architecture position: Output → Evidence Verification → Self Evaluation → Human Approval
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus, GeneratedSection, Hypothesis
    from lfx_insights.themes.discover import Embedder


# ---------------------------------------------------------------------------
# Task-specific evaluation profiles — only run relevant dimensions
# ---------------------------------------------------------------------------

TASK_PROFILES: dict[str, list[str]] = {
    "literature_search": ["retrieval_quality", "source_quality", "coverage"],
    "research_gap": ["evidence_support", "novelty", "contradiction_handling"],
    "hypothesis": ["evidence_support", "feasibility", "novelty", "methodological_consistency"],
    "manuscript": [
        "claim_support", "citation_correctness",
        "contradiction_handling", "completeness",
    ],
    "claim_verification": ["evidence_support", "citation_correctness", "confidence_calibration"],
    "default": ["evidence_support", "completeness", "confidence_calibration"],
}


# ---------------------------------------------------------------------------
# Structured evaluation result
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Structured output of a self-evaluation."""

    evaluation_id: str
    task_id: str
    task_type: str
    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    identified_problems: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_warnings: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    recommended_action: str = "none"
    created_at: str


# ---------------------------------------------------------------------------
# Deterministic dimension scorers
# ---------------------------------------------------------------------------

def _score_evidence_support(
    verification_status: str,
    confidence: float,
    n_supporting: int,
    n_contradictory: int,
) -> float:
    """Score how well evidence supports the output claims."""
    status_map = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.6,
        "INSUFFICIENT_EVIDENCE": 0.3,
        "UNSUPPORTED": 0.1,
        "CONTRADICTED": 0.0,
    }
    base = status_map.get(verification_status, 0.3)
    evidence_ratio = n_supporting / max(n_supporting + n_contradictory, 1)
    return max(0.0, min(1.0, 0.5 * base + 0.3 * confidence + 0.2 * evidence_ratio))


def _score_citation_correctness(
    citations_in_text: int,
    citations_verified: int,
) -> float:
    """Score citation correctness from verification results."""
    if citations_in_text == 0:
        return 0.0
    return min(1.0, citations_verified / citations_in_text)


def _score_completeness(
    sections_present: set[str],
    sections_expected: set[str],
    has_citations: bool,
    has_evidence: bool,
) -> float:
    """Score completeness of a manuscript/section output."""
    if not sections_expected:
        return 1.0
    section_score = len(sections_present & sections_expected) / len(sections_expected)
    bonus = 0.1 if has_citations else 0.0
    bonus += 0.1 if has_evidence else 0.0
    return max(0.0, min(1.0, section_score + bonus))


def _score_retrieval_quality(
    n_papers_retrieved: int,
    n_papers_relevant: int,
    max_similarity: float,
) -> float:
    """Score retrieval quality from corpus metadata."""
    if n_papers_retrieved == 0:
        return 0.0
    relevance = n_papers_relevant / n_papers_retrieved
    return max(0.0, min(1.0, 0.5 * relevance + 0.5 * max_similarity))


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class SelfEvaluator:
    """Evidence-grounded evaluator for research outputs.

    Reuses claim_verification for evidence inspection.
    Returns structured EvaluationResult with dimension scores.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "evaluations.json"

    def evaluate_claim(
        self,
        claim: str,
        task_id: str,
        corpus: Corpus,
        llm: LLMClient,
        embedder: Embedder,
    ) -> EvaluationResult:
        """Evaluate a claim by verifying it against the corpus."""
        from lfx_insights.projects.claim_verification import verify_claim

        verification = verify_claim(claim, corpus, llm, embedder)

        scores = {
            "evidence_support": _score_evidence_support(
                verification.status,
                verification.confidence,
                len(verification.supporting_evidence),
                len(verification.contradictory_evidence),
            ),
            "citation_correctness": _score_citation_correctness(
                len(verification.relevant_papers),
                len(verification.relevant_papers) if verification.status == "SUPPORTED" else 0,
            ),
            "confidence_calibration": verification.confidence,
        }

        problems: list[str] = []
        unsupported: list[str] = []
        contradictions: list[str] = []

        if verification.status in ("UNSUPPORTED", "CONTRADICTED"):
            unsupported.append(claim)
            problems.append(f"Claim {verification.status.lower()} by evidence")
        if verification.contradictory_evidence:
            contradictions.extend(verification.contradictory_evidence)
            n_contra = len(verification.contradictory_evidence)
            problems.append(f"{n_contra} contradictory evidence points")
        if not verification.supporting_evidence:
            problems.append("No supporting evidence found")
        if verification.confidence < 0.3:
            problems.append("Low verification confidence")

        overall = _weighted_mean(scores)
        recommended = _recommend_action(overall, problems)

        result = EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:8],
            task_id=task_id,
            task_type="claim_verification",
            overall_score=overall,
            dimension_scores=scores,
            identified_problems=problems,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            confidence=verification.confidence,
            recommended_action=recommended,
            created_at=_now_iso(),
        )
        self._append(result)
        return result

    def evaluate_manuscript(
        self,
        sections: list[GeneratedSection],
        task_id: str,
        corpus: Corpus,
        llm: LLMClient,
        embedder: Embedder,
    ) -> EvaluationResult:
        """Evaluate a manuscript by inspecting claims and citations across sections."""
        from lfx_insights.projects.claim_verification import verify_manuscript

        verifications = verify_manuscript(sections, corpus, llm, embedder)

        total_citations = sum(len(s.citations) for s in sections)
        verified_citations = sum(
            1 for v in verifications if v.status in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        )

        scores: dict[str, float] = {
            "claim_support": _score_evidence_support(
                "SUPPORTED"
                if all(v.status == "SUPPORTED" for v in verifications)
                else "PARTIALLY_SUPPORTED",
                sum(v.confidence for v in verifications) / max(len(verifications), 1),
                sum(len(v.supporting_evidence) for v in verifications),
                sum(len(v.contradictory_evidence) for v in verifications),
            ),
            "citation_correctness": _score_citation_correctness(
                total_citations, verified_citations,
            ),
            "contradiction_handling": (
                1.0 if not any(
                    v.status == "CONTRADICTED" for v in verifications
                ) else 0.3
            ),
            "completeness": _score_completeness(
                {s.name for s in sections},
                {"Introduction", "Methods", "Results", "Discussion"},
                total_citations > 0,
                bool(verifications),
            ),
        }

        problems: list[str] = []
        unsupported_claims: list[str] = []
        contradictions: list[str] = []
        citation_warnings: list[str] = []
        missing_evidence: list[str] = []

        for v in verifications:
            if v.status in ("UNSUPPORTED", "CONTRADICTED"):
                unsupported_claims.append(v.claim)
                problems.append(f"Unsupported claim: {v.claim[:80]}")
            if v.contradictory_evidence:
                contradictions.extend(v.contradictory_evidence)
            if v.status == "INSUFFICIENT_EVIDENCE":
                missing_evidence.append(v.claim)

        if total_citations > 0 and verified_citations < total_citations * 0.5:
            citation_warnings.append(
                f"Only {verified_citations}/{total_citations}"
                " citations verified"
            )

        overall = _weighted_mean(scores)
        recommended = _recommend_action(overall, problems)

        result = EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:8],
            task_id=task_id,
            task_type="manuscript",
            overall_score=overall,
            dimension_scores=scores,
            identified_problems=problems,
            unsupported_claims=unsupported_claims,
            citation_warnings=citation_warnings,
            missing_evidence=missing_evidence,
            contradictions=contradictions,
            confidence=scores.get("confidence_calibration", 0.5),
            recommended_action=recommended,
            created_at=_now_iso(),
        )
        self._append(result)
        return result

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        task_id: str,
        corpus: Corpus,
        llm: LLMClient,
        embedder: Embedder,
    ) -> EvaluationResult:
        """Evaluate a hypothesis using evidence grounding and structural checks."""
        from lfx_insights.projects.claim_verification import verify_claim

        verification = verify_claim(hypothesis.statement, corpus, llm, embedder)

        has_method = 1.0 if hypothesis.methodology else 0.2
        iv_score = 1.0 if hypothesis.independent_var else 0.0
        dv_score = 1.0 if hypothesis.dependent_var else 0.0
        has_iv_dv = 0.5 * iv_score + 0.5 * dv_score

        scores: dict[str, float] = {
            "evidence_support": _score_evidence_support(
                verification.status,
                verification.confidence,
                len(verification.supporting_evidence),
                len(verification.contradictory_evidence),
            ),
            "feasibility": (has_method + has_iv_dv) / 2,
            "novelty": 1.0 if hypothesis.qualifier != "is_associated_with" else 0.4,
            "methodological_consistency": has_method,
        }

        problems: list[str] = []
        if scores["feasibility"] < 0.4:
            problems.append("Hypothesis lacks testable methodology")
        if scores["evidence_support"] < 0.3:
            problems.append("Weak evidence grounding")

        overall = _weighted_mean(scores)
        recommended = _recommend_action(overall, problems)

        result = EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:8],
            task_id=task_id,
            task_type="hypothesis",
            overall_score=overall,
            dimension_scores=scores,
            identified_problems=problems,
            unsupported_claims=(
                [hypothesis.statement]
                if verification.status == "UNSUPPORTED"
                else []
            ),
            contradictions=verification.contradictory_evidence,
            confidence=verification.confidence,
            recommended_action=recommended,
            created_at=_now_iso(),
        )
        self._append(result)
        return result

    def get_summary(self) -> dict[str, Any]:
        """Aggregate all past evaluations."""
        evals = self._load()
        if not evals:
            return {"total_evaluations": 0, "avg_score": 0.0, "pass_rate": 0.0}

        scores = [e.overall_score for e in evals]
        avg = sum(scores) / len(scores) if scores else 0
        passed = sum(1 for e in evals if e.overall_score >= 0.6)

        return {
            "total_evaluations": len(evals),
            "avg_score": round(avg, 3),
            "pass_rate": round(passed / len(evals), 3) if evals else 0.0,
        }

    def evaluate_citations(
        self,
        sections: list[GeneratedSection],
        task_id: str,
        corpus: Corpus,
    ) -> EvaluationResult:
        """Evaluate citation quality: accuracy, coverage, and metadata consistency.

        This is a lightweight, deterministic evaluation (no LLM calls) that checks
        whether citations exist in the KB, are deduplicated correctly, and that
        the reference list matches the in-text citations.
        """
        from lfx_insights.generation.common import (
            build_cited_reference_list,
            validate_manuscript_citations,
        )

        validation = validate_manuscript_citations(sections, corpus)
        build_cited_reference_list(sections, corpus)

        total_cited = validation["total_cited"]
        all_exist = validation["all_exist"]
        ref_count = validation["reference_count"]
        issues = validation["issues"]

        scores: dict[str, float] = {
            "citation_accuracy": 1.0 if all_exist else max(0.0, 1.0 - len(issues) * 0.2),
            "citation_coverage": min(1.0, ref_count / max(total_cited, 1)),
            "metadata_consistency": 1.0 if not issues else max(0.0, 1.0 - len(issues) * 0.15),
        }

        problems: list[str] = list(issues)
        citation_warnings: list[str] = []
        if not all_exist:
            citation_warnings.append(
                f"{len(issues)} citation(s) not found in corpus"
            )
        if total_cited == 0:
            problems.append("No citations found in generated text")

        overall = _weighted_mean(scores)
        recommended = _recommend_action(overall, problems)

        result = EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:8],
            task_id=task_id,
            task_type="citation_evaluation",
            overall_score=overall,
            dimension_scores=scores,
            identified_problems=problems,
            citation_warnings=citation_warnings,
            confidence=1.0 if all_exist else 0.5,
            recommended_action=recommended,
            created_at=_now_iso(),
        )
        self._append(result)
        return result

    def _load(self) -> list[EvaluationResult]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [
                    EvaluationResult.model_validate(r) for r in data
                ]
            return []
        except (json.JSONDecodeError, Exception):
            return []

    def _append(self, result: EvaluationResult) -> None:
        evals = self._load()
        evals.append(result)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([e.model_dump() for e in evals], indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _weighted_mean(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return max(0.0, min(1.0, sum(scores.values()) / len(scores)))


def _recommend_action(overall: float, problems: list[str]) -> str:
    if overall >= 0.7 and not problems:
        return "approve"
    if overall >= 0.4:
        return "review"
    return "revise"
