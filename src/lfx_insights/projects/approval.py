"""Human approval — lightweight governance between self-evaluation and learning.

States: AI_SUGGESTED → APPROVED | REJECTED | REVIEW_REQUIRED

Low-risk changes can auto-promote. Medium and high-risk changes require
explicit researcher approval before promotion into adaptive config.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

VALID_STATES = ("AI_SUGGESTED", "APPROVED", "REJECTED", "REVIEW_REQUIRED")


class ApprovalRecord(BaseModel):
    """A single approval record for a learning signal or config change."""

    approval_id: str
    entity_type: str  # learning_signal | adaptive_config
    entity_id: str
    state: str = "AI_SUGGESTED"
    risk_level: str = "LOW_RISK"
    reason: str = ""
    reviewed_by: str | None = None
    created_at: str
    updated_at: str


class ApprovalManager:
    """Manages human approval for learning signals and adaptive config changes."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "approvals.json"

    def request_approval(
        self,
        entity_type: str,
        entity_id: str,
        risk_level: str = "LOW_RISK",
        reason: str = "",
    ) -> ApprovalRecord:
        """Create an approval request. Auto-approves LOW_RISK."""
        state = "APPROVED" if risk_level == "LOW_RISK" else "AI_SUGGESTED"
        record = ApprovalRecord(
            approval_id=uuid.uuid4().hex[:8],
            entity_type=entity_type,
            entity_id=entity_id,
            state=state,
            risk_level=risk_level,
            reason=reason,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        records = self._load()
        records.append(record)
        self._save(records)
        return record

    def approve(self, approval_id: str, reviewer: str = "researcher") -> ApprovalRecord | None:
        return self._update_state(approval_id, "APPROVED", reviewer)

    def reject(self, approval_id: str, reviewer: str = "researcher", reason: str = "") -> ApprovalRecord | None:
        records = self._load()
        for i, r in enumerate(records):
            if r.approval_id == approval_id:
                data = r.model_dump()
                data["state"] = "REJECTED"
                data["reviewed_by"] = reviewer
                data["updated_at"] = _now_iso()
                if reason:
                    data["reason"] = reason
                updated = ApprovalRecord.model_validate(data)
                records[i] = updated
                self._save(records)
                return updated
        return None

    def get_state(self, approval_id: str) -> str | None:
        for r in self._load():
            if r.approval_id == approval_id:
                return r.state
        return None

    def is_approved(self, approval_id: str) -> bool:
        return self.get_state(approval_id) == "APPROVED"

    def get_pending(self, risk_level: str | None = None) -> list[ApprovalRecord]:
        records = self._load()
        pending = [r for r in records if r.state == "AI_SUGGESTED"]
        if risk_level:
            pending = [r for r in pending if r.risk_level == risk_level]
        return pending

    def get_all(self) -> list[ApprovalRecord]:
        return self._load()

    def _update_state(self, approval_id: str, state: str, reviewer: str) -> ApprovalRecord | None:
        records = self._load()
        for i, r in enumerate(records):
            if r.approval_id == approval_id:
                data = r.model_dump()
                data["state"] = state
                data["reviewed_by"] = reviewer
                data["updated_at"] = _now_iso()
                updated = ApprovalRecord.model_validate(data)
                records[i] = updated
                self._save(records)
                return updated
        return None

    def _load(self) -> list[ApprovalRecord]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [ApprovalRecord.model_validate(r) for r in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    def _save(self, records: list[ApprovalRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
