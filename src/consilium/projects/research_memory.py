"""Persistent cross-session history of pipeline runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def record_run(
    base_dir: str | Path, topic: str, summary: dict[str, Any], *, created: str | None = None
) -> Path:
    """Append a run record to ``<base_dir>/research_history.json``.

    A pre-existing file with corrupt JSON is recovered to an empty history
    rather than crashing. The write is atomic: the new history is written to a
    temporary file in the same directory and then :func:`os.replace`-d into
    place, so a crash mid-write cannot leave a truncated history.
    """
    path = Path(base_dir) / "research_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, list):
            history = loaded
    history.append({"topic": topic, "summary": summary, "created": created})
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path
