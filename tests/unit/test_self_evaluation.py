"""Tests for self_evaluation module — evidence-grounded evaluation."""

from __future__ import annotations

import pytest

from lfx_insights.llm.client import MockLLM
from lfx_insights.models import Corpus, GeneratedSection, Hypothesis, Paper
from lfx_insights.projects.self_evaluation import (
    EvaluationResult,
    SelfEvaluator,
    _score_citation_correctness,
    _score_completeness,
    _score_evidence_support,
    _score_retrieval_quality,
    _weighted_mean,
)

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="test",
        papers=[
            Paper(id="p1", title="Gene X inhibits pathway Y", abstract="Gene X strongly inhibits pathway Y."),
            Paper(id="p2", title="Drug A treats disease B", abstract="Drug A showed therapeutic effects."),
        ],
    )


class TestDeterministicScorers:
    def test_evidence_support_supported(self) -> None:
        score = _score_evidence_support("SUPPORTED", 0.9, 3, 0)
        assert score > 0.7

    def test_evidence_support_contradicted(self) -> None:
        score = _score_evidence_support("CONTRADICTED", 0.8, 0, 3)
        assert score < 0.3

    def test_evidence_support_insufficient(self) -> None:
        score = _score_evidence_support("INSUFFICIENT_EVIDENCE", 0.3, 0, 0)
        assert 0.0 < score < 0.5

    def test_citation_correctness_all_verified(self) -> None:
        score = _score_citation_correctness(5, 5)
        assert score == 1.0

    def test_citation_correctness_none_verified(self) -> None:
        score = _score_citation_correctness(5, 0)
        assert score == 0.0

    def test_citation_correctness_no_citations(self) -> None:
        score = _score_citation_correctness(0, 0)
        assert score == 0.0

    def test_completeness_all_sections(self) -> None:
        score = _score_completeness(
            {"Introduction", "Methods", "Results", "Discussion"},
            {"Introduction", "Methods", "Results", "Discussion"},
            has_citations=True,
            has_evidence=True,
        )
        assert score >= 0.8

    def test_completeness_missing_sections(self) -> None:
        score = _score_completeness(
            {"Introduction"},
            {"Introduction", "Methods", "Results", "Discussion"},
            has_citations=False,
            has_evidence=False,
        )
        assert score < 0.5

    def test_retrieval_quality_good(self) -> None:
        score = _score_retrieval_quality(10, 8, 0.9)
        assert score > 0.7

    def test_retrieval_quality_poor(self) -> None:
        score = _score_retrieval_quality(10, 1, 0.2)
        assert score < 0.4

    def test_weighted_mean(self) -> None:
        assert _weighted_mean({"a": 0.8, "b": 0.6}) == pytest.approx(0.7, abs=0.01)

    def test_weighted_mean_empty(self) -> None:
        assert _weighted_mean({}) == 0.0


class TestEvaluationResult:
    def test_fields(self) -> None:
        r = EvaluationResult(
            evaluation_id="e1",
            task_id="t1",
            task_type="claim",
            overall_score=0.85,
            dimension_scores={"evidence_support": 0.9},
            identified_problems=[],
            confidence=0.8,
            recommended_action="approve",
            created_at="2025-01-01T00:00:00Z",
        )
        assert r.overall_score == 0.85
        assert r.recommended_action == "approve"


class TestSelfEvaluator:
    def test_evaluate_claim(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        llm = MockLLM()
        from lfx_insights.themes.discover import SimpleEmbedder
        embedder = SimpleEmbedder()

        result = evaluator.evaluate_claim(
            "Gene X inhibits pathway Y", "task_1", _corpus(), llm, embedder
        )
        assert isinstance(result, EvaluationResult)
        assert result.task_type == "claim_verification"
        assert result.overall_score > 0.0
        assert result.evaluation_id

    def test_evaluate_manuscript(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        llm = MockLLM()
        from lfx_insights.themes.discover import SimpleEmbedder
        embedder = SimpleEmbedder()

        sections = [
            GeneratedSection(name="Introduction", text="Gene X inhibits pathway Y.", citations=["[1]"]),
            GeneratedSection(name="Methods", text="We measured expression.", citations=["[2]"]),
            GeneratedSection(name="Results", text="Significant changes found.", citations=[]),
            GeneratedSection(name="Discussion", text="Our findings suggest.", citations=["[3]"]),
        ]
        result = evaluator.evaluate_manuscript(sections, "task_2", _corpus(), llm, embedder)
        assert result.task_type == "manuscript"
        assert "claim_support" in result.dimension_scores
        assert "completeness" in result.dimension_scores

    def test_evaluate_hypothesis(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        llm = MockLLM()
        from lfx_insights.themes.discover import SimpleEmbedder
        embedder = SimpleEmbedder()

        hypothesis = Hypothesis(
            subject="Gene X",
            qualifier="inhibits",
            object="pathway Y",
            statement="Gene X inhibits pathway Y",
            independent_var="Gene X expression",
            dependent_var="pathway Y activity",
            methodology="knockdown",
        )
        result = evaluator.evaluate_hypothesis(hypothesis, "task_3", _corpus(), llm, embedder)
        assert result.task_type == "hypothesis"
        assert "feasibility" in result.dimension_scores
        assert "evidence_support" in result.dimension_scores

    def test_persists_and_summarizes(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        llm = MockLLM()
        from lfx_insights.themes.discover import SimpleEmbedder
        embedder = SimpleEmbedder()

        evaluator.evaluate_claim("Gene X inhibits Y", "t1", _corpus(), llm, embedder)
        evaluator.evaluate_claim("Random claim", "t2", _corpus(), llm, embedder)

        summary = evaluator.get_summary()
        assert summary["total_evaluations"] == 2
        assert summary["avg_score"] > 0.0

    def test_summary_empty(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        summary = evaluator.get_summary()
        assert summary["total_evaluations"] == 0

    def test_evaluate_citations_all_exist(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        corpus = Corpus(
            kb_id="test",
            papers=[
                Paper(id="W1", title="Paper A", year=2020, authors=[], abstract="text A"),
                Paper(id="W2", title="Paper B", year=2019, authors=[], abstract="text B"),
            ],
        )
        sections = [
            GeneratedSection(name="intro", text="Text.", citations=["W1", "W2"]),
        ]
        result = evaluator.evaluate_citations(sections, "t1", corpus)
        assert result.task_type == "citation_evaluation"
        assert result.overall_score > 0.8
        assert result.dimension_scores["citation_accuracy"] == 1.0

    def test_evaluate_citations_missing_paper(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        corpus = Corpus(
            kb_id="test",
            papers=[Paper(id="W1", title="Paper A", year=2020, authors=[], abstract="text A")],
        )
        sections = [
            GeneratedSection(name="intro", text="Text.", citations=["W1", "W999"]),
        ]
        result = evaluator.evaluate_citations(sections, "t1", corpus)
        assert result.dimension_scores["citation_accuracy"] < 1.0
        assert len(result.citation_warnings) > 0

    def test_evaluate_citations_no_citations(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        evaluator = SelfEvaluator(base)
        corpus = _corpus()
        sections = [
            GeneratedSection(name="intro", text="Plain text.", citations=[]),
        ]
        result = evaluator.evaluate_citations(sections, "t1", corpus)
        assert result.dimension_scores["citation_coverage"] == 0.0
        assert any("No citations" in p for p in result.identified_problems)
