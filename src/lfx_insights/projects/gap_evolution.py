"""Gap evolution tracking — monitors how research gaps change over time.

Records snapshots of gap validation results and detects transitions
(e.g., newly addressed, newly identified, confidence changes).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from lfx_insights.projects.schemas import GapEvolution, GapSnapshot


class GapTransition(BaseModel):
    """A detected change in gap status between snapshots."""

    transition_id: str
    gap_text: str
    from_verdict: str | None = None
    to_verdict: str
    from_confidence: float | None = None
    to_confidence: float
    delta_confidence: float = 0.0
    description: str
    created_at: str


class GapEvolutionTracker:
    """Tracks and analyzes the evolution of research gaps across pipeline runs."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "gap_evolutions.json"

    def record_snapshot(
        self,
        project_id: str,
        gap_text: str,
        verdict: str,
        confidence: float,
        n_papers: int,
        max_similarity: float,
    ) -> GapSnapshot:
        """Record a new snapshot for a gap and detect any transitions."""
        snapshot = GapSnapshot(
            snapshot_id=uuid.uuid4().hex[:8],
            project_id=project_id,
            gap_text=gap_text,
            verdict=verdict,
            confidence=confidence,
            n_papers=n_papers,
            max_similarity=max_similarity,
            created_at=_now_iso(),
        )

        evolutions = self._load()
        evolution = self._find_or_create(evolutions, gap_text)
        transition = self._detect_transition(evolution, snapshot)
        evolution.snapshots.append(snapshot)

        self._save(evolutions)
        return snapshot

    def get_evolutions(self) -> list[GapEvolution]:
        return self._load()

    def get_evolution(self, gap_text: str) -> GapEvolution | None:
        for e in self._load():
            if e.gap_text == gap_text:
                return e
        return None

    def get_transitions(self, gap_text: str | None = None) -> list[GapTransition]:
        """Get all detected transitions, optionally filtered by gap."""
        transitions: list[GapTransition] = []
        for evolution in self._load():
            if gap_text and evolution.gap_text != gap_text:
                continue
            snapshots = evolution.snapshots
            for i in range(1, len(snapshots)):
                prev = snapshots[i - 1]
                curr = snapshots[i]
                if prev.verdict != curr.verdict or abs(prev.confidence - curr.confidence) > 0.1:
                    transitions.append(GapTransition(
                        transition_id=uuid.uuid4().hex[:8],
                        gap_text=evolution.gap_text,
                        from_verdict=prev.verdict,
                        to_verdict=curr.verdict,
                        from_confidence=prev.confidence,
                        to_confidence=curr.confidence,
                        delta_confidence=round(curr.confidence - prev.confidence, 3),
                        description=self._describe_transition(prev, curr),
                        created_at=curr.created_at,
                    ))
        return transitions

    def get_summary(self) -> dict[str, Any]:
        """Summary statistics across all tracked gaps."""
        evolutions = self._load()
        total_gaps = len(evolutions)
        total_snapshots = sum(len(e.snapshots) for e in evolutions)

        verdict_counts: dict[str, int] = {}
        for e in evolutions:
            if e.snapshots:
                latest = e.snapshots[-1]
                verdict_counts[latest.verdict] = verdict_counts.get(latest.verdict, 0) + 1

        return {
            "total_gaps": total_gaps,
            "total_snapshots": total_snapshots,
            "verdict_distribution": verdict_counts,
        }

    def _find_or_create(self, evolutions: list[GapEvolution], gap_text: str) -> GapEvolution:
        for e in evolutions:
            if e.gap_text == gap_text:
                return e
        new = GapEvolution(gap_text=gap_text, snapshots=[])
        evolutions.append(new)
        return new

    def _detect_transition(self, evolution: GapEvolution, snapshot: GapSnapshot) -> GapTransition | None:
        """Detect transition if this snapshot differs from the last one."""
        if not evolution.snapshots:
            return None
        prev = evolution.snapshots[-1]
        if prev.verdict == snapshot.verdict and abs(prev.confidence - snapshot.confidence) < 0.1:
            return None
        return GapTransition(
            transition_id=uuid.uuid4().hex[:8],
            gap_text=evolution.gap_text,
            from_verdict=prev.verdict,
            to_verdict=snapshot.verdict,
            from_confidence=prev.confidence,
            to_confidence=snapshot.confidence,
            delta_confidence=round(snapshot.confidence - prev.confidence, 3),
            description=self._describe_transition(prev, snapshot),
            created_at=snapshot.created_at,
        )

    def _describe_transition(self, prev: GapSnapshot, curr: GapSnapshot) -> str:
        if prev.verdict != curr.verdict:
            return f"Verdict changed: {prev.verdict} -> {curr.verdict}"
        delta = curr.confidence - prev.confidence
        if delta > 0:
            return f"Confidence increased by {delta:.2f}"
        return f"Confidence decreased by {abs(delta):.2f}"

    def _load(self) -> list[GapEvolution]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [GapEvolution.model_validate(e) for e in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    def _save(self, evolutions: list[GapEvolution]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([e.model_dump() for e in evolutions], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
