"""Persistent project workspace manager — CRUD for ProjectWorkspace."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lfx_insights.projects.schemas import ProjectWorkspace, SessionRecord


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


class WorkspaceManager:
    """Manages project workspaces stored as JSON."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "workspaces.json"

    def _load(self) -> list[ProjectWorkspace]:
        return [ProjectWorkspace.model_validate(r) for r in _load_db(self._path)]

    def _save(self, workspaces: list[ProjectWorkspace]) -> None:
        _save_db(self._path, [w.model_dump() for w in workspaces])

    def create_project(self, name: str, topic: str, description: str = "") -> ProjectWorkspace:
        now = _now_iso()
        ws = ProjectWorkspace(
            project_id=uuid.uuid4().hex[:12],
            name=name,
            topic=topic,
            description=description,
            created_at=now,
            updated_at=now,
        )
        workspaces = self._load()
        workspaces.append(ws)
        self._save(workspaces)
        return ws

    def get_project(self, project_id: str) -> ProjectWorkspace | None:
        for w in self._load():
            if w.project_id == project_id:
                return w
        return None

    def list_projects(self) -> list[ProjectWorkspace]:
        return self._load()

    def update_project(self, project_id: str, **fields: Any) -> ProjectWorkspace | None:
        workspaces = self._load()
        for i, w in enumerate(workspaces):
            if w.project_id == project_id:
                data = w.model_dump()
                data.update(fields)
                data["updated_at"] = _now_iso()
                updated = ProjectWorkspace.model_validate(data)
                workspaces[i] = updated
                self._save(workspaces)
                return updated
        return None

    def delete_project(self, project_id: str) -> bool:
        workspaces = self._load()
        before = len(workspaces)
        workspaces = [w for w in workspaces if w.project_id != project_id]
        if len(workspaces) < before:
            self._save(workspaces)
            return True
        return False

    def add_session(self, project_id: str, session: SessionRecord) -> None:
        ws = self.get_project(project_id)
        if ws and session.session_id not in ws.run_ids:
            ws.run_ids.append(session.session_id)
            self.update_project(project_id, run_ids=ws.run_ids)

    def add_paper(self, project_id: str, paper_id: str) -> None:
        ws = self.get_project(project_id)
        if ws and paper_id not in ws.paper_ids:
            ws.paper_ids.append(paper_id)
            self.update_project(project_id, paper_ids=ws.paper_ids)

    def add_gap(self, project_id: str, gap_id: str) -> None:
        ws = self.get_project(project_id)
        if ws and gap_id not in ws.gap_ids:
            ws.gap_ids.append(gap_id)
            self.update_project(project_id, gap_ids=ws.gap_ids)
