"""Tests for comparison module."""

from __future__ import annotations

import pytest

from lfx_insights.llm.client import MockLLM
from lfx_insights.models import Corpus, Paper
from lfx_insights.projects.comparison import ComparisonEngine, ComparisonResult

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="test",
        papers=[
            Paper(id="p1", title="Gene X study", abstract="We studied gene X."),
            Paper(id="p2", title="Gene Y study", abstract="We studied gene Y."),
        ],
    )


class TestComparisonEngine:
    def test_compare_basic(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine = ComparisonEngine(base)
        llm = MockLLM()
        corpus = _corpus()
        result = engine.compare(["p1", "p2"], corpus, llm, dimensions=["methodology"])
        assert result.comparison_id
        assert "methodology" in result.entries
        assert "p1" in result.entries["methodology"]
        assert "p2" in result.entries["methodology"]

    def test_compare_default_dimensions(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine = ComparisonEngine(base)
        llm = MockLLM()
        corpus = _corpus()
        result = engine.compare(["p1", "p2"], corpus, llm)
        assert len(result.dimensions) > 0
        assert result.synthesis  # MockLLM returns a string

    def test_compare_empty_corpus(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine = ComparisonEngine(base)
        llm = MockLLM()
        empty = Corpus(kb_id="empty", papers=[])
        result = engine.compare(["p1"], empty, llm)
        assert result.entries == {}

    def test_compare_persists(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine1 = ComparisonEngine(base)
        llm = MockLLM()
        corpus = _corpus()
        engine1.compare(["p1", "p2"], corpus, llm)

        engine2 = ComparisonEngine(base)
        all_comp = engine2.get_all()
        assert len(all_comp) == 1

    def test_get_by_id(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine = ComparisonEngine(base)
        llm = MockLLM()
        corpus = _corpus()
        result = engine.compare(["p1", "p2"], corpus, llm)
        found = engine.get_by_id(result.comparison_id)
        assert found is not None
        assert found.comparison_id == result.comparison_id

    def test_get_by_id_not_found(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        engine = ComparisonEngine(base)
        assert engine.get_by_id("nonexistent") is None

    def test_comparison_result_model(self) -> None:
        r = ComparisonResult(
            comparison_id="c1",
            paper_ids=["p1"],
            dimensions=["methodology"],
            created_at="2025-01-01T00:00:00Z",
        )
        assert r.entries == {}
        assert r.synthesis == ""
