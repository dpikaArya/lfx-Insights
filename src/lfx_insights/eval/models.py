"""Frozen data contracts for the eval harness.

Every builder module (dataset, retrieval, answer, metrics, runner) targets these
models. They reuse the house :class:`~lfx_insights.models.Score` /
:class:`~lfx_insights.models.Provenance` for honest, provenance-carrying aggregates.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lfx_insights.models import Provenance, Score


class RetrievedDoc(BaseModel):
    """A context document the answer's inline ``[n]`` markers refer to.

    ``doc_id`` is stable within a case; the inline marker ``n`` indexes 0-based into
    the answer's ordered ``docs`` list. ``paper_id`` is set when the doc was produced
    by a retrieval backend (so it can be grounded against a corpus Paper).
    """

    doc_id: str
    title: str = ""
    text: str = ""
    paper_id: str | None = None


class EvalCase(BaseModel):
    """One normalised ScholarQABench example (any subset)."""

    id: str
    question: str  # ScholarQABench "input"
    reference_answer: str | None = None  # "output"/"answer" long-form reference
    gold_label: str | None = None  # SciFact "true"/"false", PubMedQA "yes"/"no"
    ctxs: list[RetrievedDoc] = Field(default_factory=list)  # provided datastore / tfidf pool
    # Identifiers of the gold/relevant documents for intrinsic retrieval scoring
    # (LitSearch): corpus ids, DOIs, and/or normalised titles. Empty = no retrieval eval.
    gold_docs: list[str] = Field(default_factory=list)
    subject: str | None = None  # cs | bio | physics | ...
    task: str = "long_form"  # long_form | short_form | retrieval
    # Default scorers for this case (subset of: citation, match, rouge, quality, retrieval).
    metrics: list[str] = Field(default_factory=lambda: ["citation"])


class Citation(BaseModel):
    """A parsed, in-range inline citation: marker ``[n]`` -> the doc it points to."""

    marker: int  # the n in [n], 0-indexed into the answer's docs
    doc_id: str


class GeneratedAnswer(BaseModel):
    """A long-form answer with inline ``[n]`` markers and the numbered docs they cite."""

    text: str
    docs: list[RetrievedDoc] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class CitationScore(BaseModel):
    """Per-case citation faithfulness (AutoAIS/ALCE scheme)."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    n_sentences: int = Field(ge=0)  # scored sentences (>= 50 chars)
    n_citations: int = Field(ge=0)
    judge: str  # the entailer used (lexical | grounding | llm | ...)


class CorrectnessScore(BaseModel):
    """Per-case reference-based correctness."""

    metric: str  # "match" | "rouge_l"
    value: float = Field(ge=0.0, le=1.0)


class QualityScore(BaseModel):
    """Per-case answer quality (approx. Prometheus 3-aspect, each 1-5 normalised /5)."""

    organization: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    judge: str


class RetrievalScore(BaseModel):
    """Per-case intrinsic retrieval quality (LitSearch-style: did we fetch the gold papers)."""

    recall: float = Field(ge=0.0, le=1.0)  # recall@k
    ndcg: float = Field(ge=0.0, le=1.0)  # nDCG@k, binary relevance
    k: int = Field(ge=1)
    n_gold: int = Field(ge=0)
    n_retrieved: int = Field(ge=0)


class CaseResult(BaseModel):
    """All scores for one (case, retrieval condition)."""

    case_id: str
    condition: str  # null | tfidf | perspicacite
    answer: GeneratedAnswer
    n_retrieved: int = Field(ge=0)
    citation: CitationScore | None = None
    correctness: CorrectnessScore | None = None
    quality: QualityScore | None = None
    retrieval: RetrievalScore | None = None


class ConditionReport(BaseModel):
    """Aggregate over all cases for one retrieval condition.

    Aggregates are honest :class:`~lfx_insights.models.Score`s (components/weights/
    method/interpretation/uncertainty), never bare means.
    """

    condition: str
    n_cases: int = Field(ge=0)
    citation: Score | None = None
    correctness: Score | None = None
    quality: Score | None = None
    retrieval: Score | None = None
    caveats: list[str] = Field(default_factory=list)


class AblationReport(BaseModel):
    """The full ablation: per-condition aggregates + per-case detail + the lift."""

    dataset: str
    conditions: list[ConditionReport] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    # Headline metric lift, e.g. {"citation_f1": perspicacite - tfidf}.
    lift: dict[str, float] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
