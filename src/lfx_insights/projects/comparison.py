"""Multi-paper comparison — structured comparison across configurable dimensions.

Extracts and contrasts specific aspects (methodology, findings, limitations, etc.)
across a set of papers using LLM structured output.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus


DEFAULT_DIMENSIONS = [
    "research_question",
    "methodology",
    "sample_size",
    "key_findings",
    "limitations",
    "future_directions",
]


class ComparisonResult(BaseModel):
    """A completed multi-paper comparison."""

    comparison_id: str
    paper_ids: list[str]
    dimensions: list[str]
    entries: dict[str, dict[str, str]] = Field(default_factory=dict)
    synthesis: str = ""
    created_at: str


class ComparisonEngine:
    """Runs structured multi-paper comparisons."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "comparisons.json"

    def compare(
        self,
        paper_ids: list[str],
        corpus: Corpus,
        llm: LLMClient,
        dimensions: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare papers across specified dimensions."""
        dims = dimensions or DEFAULT_DIMENSIONS
        papers = [corpus.by_id(pid) for pid in paper_ids]
        papers = [p for p in papers if p is not None]

        if not papers:
            return ComparisonResult(
                comparison_id=uuid.uuid4().hex[:8],
                paper_ids=paper_ids,
                dimensions=dims,
                created_at=_now_iso(),
            )

        entries: dict[str, dict[str, str]] = {}
        for dim in dims:
            dim_results: dict[str, str] = {}
            for paper in papers:
                prompt = (
                    f"Paper: {paper.title}\n"
                    f"Abstract: {paper.abstract or 'N/A'}\n\n"
                    f"What is this paper's {dim}? Reply in 1-2 sentences."
                )
                try:
                    resp = llm.complete(prompt)
                    dim_results[paper.id] = resp if isinstance(resp, str) else str(resp)
                except Exception:
                    dim_results[paper.id] = "Unable to extract"
            entries[dim] = dim_results

        # Synthesis
        synthesis_prompt = self._build_synthesis_prompt(papers, dims, entries)
        try:
            synthesis_resp = llm.complete(synthesis_prompt)
            synthesis = synthesis_resp if isinstance(synthesis_resp, str) else str(synthesis_resp)
        except Exception:
            synthesis = ""

        result = ComparisonResult(
            comparison_id=uuid.uuid4().hex[:8],
            paper_ids=paper_ids,
            dimensions=dims,
            entries=entries,
            synthesis=synthesis,
            created_at=_now_iso(),
        )

        self._append(result)
        return result

    def get_all(self) -> list[ComparisonResult]:
        return self._load()

    def get_by_id(self, comparison_id: str) -> ComparisonResult | None:
        for c in self._load():
            if c.comparison_id == comparison_id:
                return c
        return None

    def _build_synthesis_prompt(
        self, papers: list[Any], dims: list[str], entries: dict[str, dict[str, str]]
    ) -> str:
        lines = ["Summarize the key similarities and differences across these papers:\n"]
        for dim in dims:
            lines.append(f"## {dim}")
            for paper in papers:
                val = entries.get(dim, {}).get(paper.id, "N/A")
                lines.append(f"- {paper.title}: {val}")
            lines.append("")
        lines.append("Provide a 3-5 sentence synthesis highlighting convergent and divergent findings.")
        return "\n".join(lines)

    def _load(self) -> list[ComparisonResult]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [ComparisonResult.model_validate(c) for c in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    def _append(self, result: ComparisonResult) -> None:
        results = self._load()
        results.append(result)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([r.model_dump() for r in results], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
