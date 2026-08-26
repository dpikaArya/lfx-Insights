"""Tests for workspace, session memory, audit, and evidence ledger."""

from __future__ import annotations

import pytest

from lfx_insights.projects.audit import AuditTrail
from lfx_insights.projects.evidence_ledger import EvidenceLedger
from lfx_insights.projects.session_memory import SessionMemory
from lfx_insights.projects.workspace import WorkspaceManager

pytestmark = pytest.mark.unit


class TestWorkspaceManager:
    def test_create_and_get(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        ws = wm.create_project("Test", "gene therapy", "A test project")
        assert ws.name == "Test"
        assert ws.topic == "gene therapy"
        got = wm.get_project(ws.project_id)
        assert got is not None
        assert got.project_id == ws.project_id

    def test_list_projects(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        wm.create_project("P1", "topic1")
        wm.create_project("P2", "topic2")
        assert len(wm.list_projects()) == 2

    def test_update_project(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        ws = wm.create_project("Test", "topic")
        updated = wm.update_project(ws.project_id, name="Updated")
        assert updated is not None
        assert updated.name == "Updated"

    def test_delete_project(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        ws = wm.create_project("Test", "topic")
        assert wm.delete_project(ws.project_id) is True
        assert wm.get_project(ws.project_id) is None

    def test_add_paper(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        ws = wm.create_project("Test", "topic")
        wm.add_paper(ws.project_id, "paper1")
        wm.add_paper(ws.project_id, "paper1")  # duplicate
        got = wm.get_project(ws.project_id)
        assert got is not None
        assert got.paper_ids == ["paper1"]

    def test_get_nonexistent(self, tmp_path: object) -> None:
        wm = WorkspaceManager(tmp_path)
        assert wm.get_project("nonexistent") is None


class TestSessionMemory:
    def test_record_and_list(self, tmp_path: object) -> None:
        sm = SessionMemory(tmp_path)
        s = sm.record_session("p1", "gene therapy", stages_run=["themes", "novelty"])
        assert s.project_id == "p1"
        assert s.stages_run == ["themes", "novelty"]
        sessions = sm.list_sessions("p1")
        assert len(sessions) == 1

    def test_get_recent(self, tmp_path: object) -> None:
        sm = SessionMemory(tmp_path)
        for i in range(5):
            sm.record_session("p1", f"topic{i}")
        recent = sm.get_recent(3)
        assert len(recent) == 3

    def test_list_all(self, tmp_path: object) -> None:
        sm = SessionMemory(tmp_path)
        sm.record_session("p1", "t1")
        sm.record_session("p2", "t2")
        assert len(sm.list_sessions()) == 2


class TestAuditTrail:
    def test_record_and_list(self, tmp_path: object) -> None:
        at = AuditTrail(tmp_path)
        entry = at.record("p1", "claim_verification", "claim", "c1")
        assert entry.action == "claim_verification"
        assert entry.approval_state == "AI_SUGGESTED"
        records = at.list_for_project("p1")
        assert len(records) == 1

    def test_update_approval(self, tmp_path: object) -> None:
        at = AuditTrail(tmp_path)
        entry = at.record("p1", "researcher_approval", "claim", "c1")
        updated = at.update_approval(entry.record_id, "APPROVED")
        assert updated is not None
        assert updated.approval_state == "APPROVED"

    def test_list_for_entity(self, tmp_path: object) -> None:
        at = AuditTrail(tmp_path)
        at.record("p1", "search", "paper", "paper1")
        at.record("p1", "search", "paper", "paper2")
        at.record("p1", "claim_verification", "claim", "c1")
        papers = at.list_for_entity("paper", "paper1")
        assert len(papers) == 1


class TestEvidenceLedger:
    def test_add_and_get(self, tmp_path: object) -> None:
        el = EvidenceLedger(tmp_path)
        rec = el.add("c1", "d1", "supporting passage", support_type="supporting", confidence=0.9)
        assert rec.claim_id == "c1"
        got = el.get(rec.evidence_id)
        assert got is not None
        assert got.passage == "supporting passage"

    def test_list_for_claim(self, tmp_path: object) -> None:
        el = EvidenceLedger(tmp_path)
        el.add("c1", "d1", "passage1")
        el.add("c1", "d2", "passage2")
        el.add("c2", "d1", "passage3")
        assert len(el.list_for_claim("c1")) == 2

    def test_delete(self, tmp_path: object) -> None:
        el = EvidenceLedger(tmp_path)
        rec = el.add("c1", "d1", "passage")
        assert el.delete(rec.evidence_id) is True
        assert el.get(rec.evidence_id) is None
