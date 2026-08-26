"""Lightweight audit trail for research actions."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lfx_insights.projects.schemas import AuditRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_db(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _save_db(path: Path, db: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class AuditTrail:
    """Manages audit records for research actions."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "audit.json"

    def _load(self) -> list[AuditRecord]:
        return [AuditRecord.model_validate(r) for r in _load_db(self._path)]

    def _save(self, records: list[AuditRecord]) -> None:
        _save_db(self._path, [r.model_dump() for r in records])

    def record(
        self,
        project_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
        approval_state: str = "AI_SUGGESTED",
    ) -> AuditRecord:
        entry = AuditRecord(
            record_id=uuid.uuid4().hex[:12],
            project_id=project_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details or {},
            approval_state=approval_state,
            created_at=_now_iso(),
        )
        records = self._load()
        records.append(entry)
        self._save(records)
        return entry

    def list_for_project(self, project_id: str) -> list[AuditRecord]:
        return [r for r in self._load() if r.project_id == project_id]

    def list_for_entity(self, entity_type: str, entity_id: str) -> list[AuditRecord]:
        return [
            r for r in self._load()
            if r.entity_type == entity_type and r.entity_id == entity_id
        ]

    def update_approval(self, record_id: str, state: str) -> AuditRecord | None:
        records = self._load()
        for i, r in enumerate(records):
            if r.record_id == record_id:
                data = r.model_dump()
                data["approval_state"] = state
                updated = AuditRecord.model_validate(data)
                records[i] = updated
                self._save(records)
                return updated
        return None
