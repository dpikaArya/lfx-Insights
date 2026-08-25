"""OutputStore: writes artifacts (md/json/text) under a run directory.

Tracks the names written during the current run so aggregation reads only this
run's artifacts (and never stale files left in a shared output directory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OutputStore:
    def __init__(self, base_dir: str | Path, run: str = "default") -> None:
        self.run_dir = Path(base_dir) / run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def _record(self, name: str) -> None:
        if name not in self.written:
            self.written.append(name)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text, encoding="utf-8")
        self._record(name)
        return p

    def write_markdown(self, name: str, text: str) -> Path:
        return self.write_text(name, text)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        self._record(name)
        return p
