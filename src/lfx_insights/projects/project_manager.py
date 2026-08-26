"""Track research projects across runs in a JSON database."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def record_project(
    base_dir: str | Path, topic: str, summary: dict[str, Any], *, created: str | None = None
) -> Path:
    """Append a project record to ``<base_dir>/projects/project_database.json``.

    A pre-existing file with corrupt JSON is recovered to an empty database
    rather than crashing. The write is atomic: the new database is written to a
    temporary file in the same directory and then :func:`os.replace`-d into
    place, so a crash mid-write cannot leave a truncated database.
    """
    path = Path(base_dir) / "projects" / "project_database.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    db: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, list):
            db = loaded
    stages = summary.get("stages") or {}
    db.append(
        {
            "topic": topic,
            "kb_id": summary.get("kb_id"),
            "stages": list(stages.keys()) if isinstance(stages, dict) else [],
            "created": created,
        }
    )
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path
