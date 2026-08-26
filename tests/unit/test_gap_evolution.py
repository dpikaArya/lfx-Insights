"""Tests for gap_evolution module."""

from __future__ import annotations

import pytest

from lfx_insights.projects.gap_evolution import GapEvolutionTracker, GapTransition
from lfx_insights.projects.schemas import GapEvolution, GapSnapshot

pytestmark = pytest.mark.unit


class TestGapEvolutionTracker:
    def test_record_snapshot(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        snapshot = tracker.record_snapshot(
            project_id="proj1",
            gap_text="No studies on gene Z in cancer",
            verdict="validated_gap",
            confidence=0.9,
            n_papers=0,
            max_similarity=0.3,
        )
        assert snapshot.snapshot_id
        assert snapshot.gap_text == "No studies on gene Z in cancer"

    def test_record_multiple_snapshots(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("proj1", "gap A", "validated_gap", 0.9, 0, 0.3)
        tracker.record_snapshot("proj1", "gap A", "addressed_gap", 0.7, 5, 0.8)
        evolutions = tracker.get_evolutions()
        assert len(evolutions) == 1
        assert len(evolutions[0].snapshots) == 2

    def test_get_evolution(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("proj1", "gap X", "validated_gap", 0.8, 0, 0.3)
        evo = tracker.get_evolution("gap X")
        assert evo is not None
        assert evo.gap_text == "gap X"

    def test_get_evolution_not_found(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        assert tracker.get_evolution("nonexistent") is None

    def test_detect_transition_verdict_change(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("proj1", "gap", "validated_gap", 0.9, 0, 0.3)
        tracker.record_snapshot("proj1", "gap", "addressed_gap", 0.7, 5, 0.8)
        transitions = tracker.get_transitions("gap")
        assert len(transitions) == 1
        assert transitions[0].from_verdict == "validated_gap"
        assert transitions[0].to_verdict == "addressed_gap"

    def test_detect_transition_confidence_change(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("proj1", "gap", "validated_gap", 0.5, 2, 0.5)
        tracker.record_snapshot("proj1", "gap", "validated_gap", 0.9, 2, 0.5)
        transitions = tracker.get_transitions("gap")
        assert len(transitions) == 1
        assert transitions[0].delta_confidence == pytest.approx(0.4, abs=0.01)

    def test_no_transition_when_stable(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("proj1", "gap", "validated_gap", 0.8, 2, 0.5)
        tracker.record_snapshot("proj1", "gap", "validated_gap", 0.8, 2, 0.5)
        transitions = tracker.get_transitions("gap")
        assert len(transitions) == 0

    def test_get_summary(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        tracker.record_snapshot("p", "g1", "validated_gap", 0.8, 0, 0.3)
        tracker.record_snapshot("p", "g2", "addressed_gap", 0.7, 5, 0.8)
        summary = tracker.get_summary()
        assert summary["total_gaps"] == 2
        assert summary["total_snapshots"] == 2
        assert "validated_gap" in summary["verdict_distribution"]

    def test_persistence(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker1 = GapEvolutionTracker(base)
        tracker1.record_snapshot("p", "gap", "validated_gap", 0.8, 0, 0.3)
        tracker2 = GapEvolutionTracker(base)
        assert len(tracker2.get_evolutions()) == 1

    def test_empty_transitions(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        tracker = GapEvolutionTracker(base)
        assert tracker.get_transitions() == []
