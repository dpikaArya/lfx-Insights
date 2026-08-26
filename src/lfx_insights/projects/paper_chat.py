"""Paper-specific Q&A chat — ask questions grounded in a single paper's full text.

Uses retrieval over paper chunks + LLM for grounded answers with page citations.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient


class PaperChatEntry(BaseModel):
    """A single Q&A exchange."""

    entry_id: str
    paper_id: str
    question: str
    answer: str
    page: int | None = None
    section: str | None = None
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str


class PaperChatSession:
    """Conversational Q&A over a specific paper's full text."""

    def __init__(self, paper_id: str, base_dir: str | Path) -> None:
        self.paper_id = paper_id
        self._path = Path(base_dir) / "projects" / f"chat_{paper_id}.json"
        self._history: list[PaperChatEntry] = []

    @property
    def history(self) -> list[PaperChatEntry]:
        if not self._history:
            self._history = self._load()
        return self._history

    def ask(
        self,
        question: str,
        chunks: list[str],
        llm: LLMClient,
    ) -> PaperChatEntry:
        """Ask a question about the paper, given pre-retrieved text chunks."""
        context = "\n\n---\n\n".join(chunks[:5]) if chunks else "No relevant content found."

        prompt = (
            f"You are answering a question about paper {self.paper_id}.\n"
            f"Use ONLY the following context from the paper:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely. If the context does not contain the answer, say so. "
            "Cite page numbers or sections when visible in the context."
        )

        response = llm.complete(prompt)
        answer_text = response if isinstance(response, str) else str(response)

        # Simple citation extraction from answer
        import re
        pages = re.findall(r"page\s+(\d+)", answer_text.lower())
        page_num = int(pages[0]) if pages else None

        entry = PaperChatEntry(
            entry_id=uuid.uuid4().hex[:8],
            paper_id=self.paper_id,
            question=question,
            answer=answer_text,
            page=page_num,
            citations=[c.strip() for c in chunks[:3]],
            confidence=min(len(chunks) / 3, 1.0) if chunks else 0.1,
            created_at=_now_iso(),
        )

        self._history.append(entry)
        self._save()
        return entry

    def get_history(self) -> list[PaperChatEntry]:
        return self.history

    def clear(self) -> None:
        self._history = []
        self._save()

    def _load(self) -> list[PaperChatEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [PaperChatEntry.model_validate(e) for e in data]
            return []
        except (json.JSONDecodeError, Exception):
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([e.model_dump() for e in self._history], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now(UTC).isoformat()
