"""Evidence ledger — compact evidence records linking passages to claims."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lfx_insights.projects.schemas import EvidenceRecord


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


class EvidenceLedger:
    """Manages evidence records in a JSON file."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "evidence_ledger.json"

    def _load(self) -> list[EvidenceRecord]:
        return [EvidenceRecord.model_validate(r) for r in _load_db(self._path)]

    def _save(self, records: list[EvidenceRecord]) -> None:
        _save_db(self._path, [r.model_dump() for r in records])

    def add(
        self,
        claim_id: str,
        document_id: str,
        passage: str,
        support_type: str = "supporting",
        confidence: float = 0.5,
        page: int | None = None,
        section: str | None = None,
    ) -> EvidenceRecord:
        record = EvidenceRecord(
            evidence_id=uuid.uuid4().hex[:12],
            claim_id=claim_id,
            document_id=document_id,
            page=page,
            section=section,
            passage=passage,
            support_type=support_type,
            confidence=confidence,
        )
        records = self._load()
        records.append(record)
        self._save(records)
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        for r in self._load():
            if r.evidence_id == evidence_id:
                return r
        return None

    def list_for_claim(self, claim_id: str) -> list[EvidenceRecord]:
        return [r for r in self._load() if r.claim_id == claim_id]

    def list_for_document(self, document_id: str) -> list[EvidenceRecord]:
        return [r for r in self._load() if r.document_id == document_id]

    def update(self, evidence_id: str, **fields: Any) -> EvidenceRecord | None:
        records = self._load()
        for i, r in enumerate(records):
            if r.evidence_id == evidence_id:
                data = r.model_dump()
                data.update(fields)
                updated = EvidenceRecord.model_validate(data)
                records[i] = updated
                self._save(records)
                return updated
        return None

    def delete(self, evidence_id: str) -> bool:
        records = self._load()
        before = len(records)
        records = [r for r in records if r.evidence_id != evidence_id]
        if len(records) < before:
            self._save(records)
            return True
        return False
