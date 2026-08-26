"""Tests for research memory schemas."""

from __future__ import annotations

import pytest

from lfx_insights.projects.schemas import (
    AuditRecord,
    ClaimRecord,
    ClaimVerification,
    EvidenceRecord,
    GapEvolution,
    GapSnapshot,
    PaperChatMessage,
    PaperComparison,
    PaperRecord,
    ProjectWorkspace,
    SessionRecord,
)

pytestmark = pytest.mark.unit


def test_project_workspace_roundtrip() -> None:
    ws = ProjectWorkspace(
        project_id="abc123",
        name="Test Project",
        topic="gene therapy",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    data = ws.model_dump()
    ws2 = ProjectWorkspace.model_validate(data)
    assert ws2.project_id == "abc123"
    assert ws2.status == "active"
    assert ws2.paper_ids == []


def test_session_record_defaults() -> None:
    s = SessionRecord(
        session_id="s1",
        project_id="p1",
        topic="test",
        created_at="2026-01-01T00:00:00Z",
    )
    assert s.stages_run == []
    assert s.n_papers == 0
    assert s.evidence_ids == []


def test_evidence_record_validation() -> None:
    e = EvidenceRecord(
        evidence_id="e1",
        claim_id="c1",
        document_id="d1",
        passage="supporting text",
        support_type="supporting",
        confidence=0.8,
    )
    assert e.confidence == 0.8
    assert e.support_type == "supporting"


def test_evidence_record_confidence_bounds() -> None:
    with pytest.raises(Exception):
        EvidenceRecord(
            evidence_id="e1",
            claim_id="c1",
            document_id="d1",
            passage="text",
            support_type="supporting",
            confidence=1.5,
        )


def test_claim_record_defaults() -> None:
    c = ClaimRecord(
        claim_id="c1",
        project_id="p1",
        statement="Gene X regulates pathway Y",
        created_at="2026-01-01T00:00:00Z",
    )
    assert c.verification_status == "unverified"
    assert c.approval_state == "AI_SUGGESTED"


def test_claim_verification_statuses() -> None:
    for status in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]:
        cv = ClaimVerification(
            claim_id="c1",
            status=status,
            confidence=0.7,
            verified_at="2026-01-01T00:00:00Z",
        )
        assert cv.status == status


def test_audit_record_defaults() -> None:
    a = AuditRecord(
        record_id="a1",
        project_id="p1",
        action="claim_verification",
        entity_type="claim",
        entity_id="c1",
        created_at="2026-01-01T00:00:00Z",
    )
    assert a.approval_state == "AI_SUGGESTED"
    assert a.details == {}


def test_gap_snapshot_roundtrip() -> None:
    gs = GapSnapshot(
        snapshot_id="gs1",
        project_id="p1",
        gap_text="No studies on X",
        verdict="Confirmed",
        confidence=0.8,
        n_papers=10,
        max_similarity=0.45,
        created_at="2026-01-01T00:00:00Z",
    )
    data = gs.model_dump()
    gs2 = GapSnapshot.model_validate(data)
    assert gs2.verdict == "Confirmed"


def test_paper_chat_message() -> None:
    msg = PaperChatMessage(
        role="assistant",
        content="The paper discusses gene therapy for cystic fibrosis.",
        paper_id="p1",
        page=5,
        section="Discussion",
    )
    assert msg.paper_id == "p1"
    assert msg.page == 5


def test_paper_comparison() -> None:
    from lfx_insights.projects.schemas import ComparisonDimension, PaperComparisonEntry

    comp = PaperComparison(
        paper_ids=["p1", "p2"],
        dimensions=[
            ComparisonDimension(
                dimension="methods",
                comparisons=[
                    PaperComparisonEntry(paper_id="p1", content="RNA-seq"),
                    PaperComparisonEntry(paper_id="p2", content="Western blot"),
                ],
            )
        ],
    )
    assert len(comp.dimensions) == 1
    assert comp.dimensions[0].dimension == "methods"
