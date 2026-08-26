"""Compact session memory — stores research session records across runs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lfx_insights.projects.schemas import SessionRecord


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


class SessionMemory:
    """Manages session records in a JSON file."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "sessions.json"

    def _load(self) -> list[SessionRecord]:
        return [SessionRecord.model_validate(r) for r in _load_db(self._path)]

    def _save(self, sessions: list[SessionRecord]) -> None:
        _save_db(self._path, [s.model_dump() for s in sessions])

    def record_session(
        self,
        project_id: str,
        topic: str,
        stages_run: list[str] | None = None,
        kb_id: str | None = None,
        n_papers: int = 0,
        evidence_ids: list[str] | None = None,
        summary: dict[str, Any] | None = None,
        researcher_decision: str | None = None,
    ) -> SessionRecord:
        session = SessionRecord(
            session_id=uuid.uuid4().hex[:12],
            project_id=project_id,
            topic=topic,
            stages_run=stages_run or [],
            kb_id=kb_id,
            n_papers=n_papers,
            evidence_ids=evidence_ids or [],
            summary=summary or {},
            researcher_decision=researcher_decision,
            created_at=_now_iso(),
        )
        sessions = self._load()
        sessions.append(session)
        self._save(sessions)
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        for s in self._load():
            if s.session_id == session_id:
                return s
        return None

    def list_sessions(self, project_id: str | None = None) -> list[SessionRecord]:
        sessions = self._load()
        if project_id:
            return [s for s in sessions if s.project_id == project_id]
        return sessions

    def get_recent(self, n: int = 10) -> list[SessionRecord]:
        return self._load()[-n:]
