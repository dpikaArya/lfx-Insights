"""Typed schemas for research memory, workspace, evidence, claims, and audit."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectWorkspace(BaseModel):
    """A persistent research project linking all research artifacts."""

    project_id: str
    name: str
    description: str = ""
    topic: str
    status: str = "active"  # active | archived | paused
    created_at: str
    updated_at: str
    run_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    theme_ids: list[int] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    gap_ids: list[str] = Field(default_factory=list)
    manuscript_ids: list[str] = Field(default_factory=list)


class SessionRecord(BaseModel):
    """Compact record of a research session / pipeline run."""

    session_id: str
    project_id: str
    topic: str
    stages_run: list[str] = Field(default_factory=list)
    kb_id: str | None = None
    n_papers: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    summary: dict[str, object] = Field(default_factory=dict)
    researcher_decision: str | None = None
    created_at: str


class PaperSection(BaseModel):
    """An extracted section from a paper."""

    section_name: str
    page_start: int | None = None
    page_end: int | None = None
    paragraphs: list[str] = Field(default_factory=list)


class TextChunk(BaseModel):
    """A retrievable text chunk from a paper section."""

    chunk_id: str
    text: str
    page: int | None = None
    paragraph_index: int | None = None
    section_name: str | None = None


class PaperReference(BaseModel):
    """An extracted reference from a paper's bibliography."""

    ref_id: str
    text: str
    doi: str | None = None
    parsed_authors: list[str] = Field(default_factory=list)
    parsed_year: int | None = None
    parsed_title: str | None = None


class PaperRecord(BaseModel):
    """Extended paper record with full-text extraction metadata."""

    paper_id: str
    doi: str | None = None
    title: str
    year: int | None = None
    abstract: str | None = None
    full_text_path: str | None = None
    sections: list[PaperSection] = Field(default_factory=list)
    references: list[PaperReference] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    """A single evidence record linking a passage to a claim."""

    evidence_id: str
    claim_id: str
    document_id: str
    page: int | None = None
    section: str | None = None
    passage: str
    support_type: str  # supporting | contradictory | contextual | limitation
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimRecord(BaseModel):
    """A scientific claim tracked in the evidence ledger."""

    claim_id: str
    project_id: str
    statement: str
    source_evidence: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"  # unverified | verified | disputed
    confidence: float = 0.0
    approval_state: str = "AI_SUGGESTED"  # AI_SUGGESTED | APPROVED | REJECTED | REVIEW_REQUIRED
    created_at: str


class ClaimVerification(BaseModel):
    """Result of verifying a claim against the corpus."""

    claim_id: str
    # SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    relevant_papers: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    verified_at: str


class AuditRecord(BaseModel):
    """Lightweight audit trail entry for a research action."""

    record_id: str
    project_id: str
    # search | retrieval | paper_inclusion | paper_exclusion |
    # claim_verification | evidence_classification |
    # researcher_approval | manuscript_verification
    action: str
    entity_type: str  # paper | gap | hypothesis | claim | manuscript | evidence
    entity_id: str
    details: dict[str, object] = Field(default_factory=dict)
    approval_state: str = "AI_SUGGESTED"
    created_at: str


class GapSnapshot(BaseModel):
    """A point-in-time snapshot of a gap's validation state."""

    snapshot_id: str
    project_id: str
    gap_text: str
    verdict: str
    confidence: float
    n_papers: int
    max_similarity: float
    created_at: str


class GapEvolution(BaseModel):
    """The evolution history of a research gap across snapshots."""

    gap_text: str
    snapshots: list[GapSnapshot] = Field(default_factory=list)


class PaperChatMessage(BaseModel):
    """A response from paper-specific Q&A."""

    role: str  # user | assistant
    content: str
    paper_id: str | None = None
    page: int | None = None
    section: str | None = None
    citations: list[str] = Field(default_factory=list)


class ComparisonDimension(BaseModel):
    """One dimension of a multi-paper comparison."""

    dimension: str
    comparisons: list[PaperComparisonEntry] = Field(default_factory=list)


class PaperComparisonEntry(BaseModel):
    """A single paper's content for one comparison dimension."""

    paper_id: str
    content: str


class PaperComparison(BaseModel):
    """Structured comparison across multiple papers."""

    paper_ids: list[str]
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
