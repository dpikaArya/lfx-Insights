"""End-to-end ablation: run the eval runner over a small dataset with a real
TfidfBackend and a deterministic MockLLM, asserting the runner discriminates a
retrieving condition (tfidf) from a closed-book one (null) and emits honest Scores.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from lfx_insights.cli import main
from lfx_insights.config import load_settings
from lfx_insights.eval.answer import AnswerDraft, CitedMarker
from lfx_insights.eval.metrics.quality import QualityRubric
from lfx_insights.eval.models import EvalCase, RetrievedDoc
from lfx_insights.eval.report import render_markdown
from lfx_insights.eval.runner import run_ablation

pytestmark = pytest.mark.e2e

# A phrase present verbatim in every source, so the cited quote grounds regardless
# of which source TF-IDF ranks first.
_PHRASE = "grounds claims in retrieved sources"
_DOC_TEXT = (
    "Machine learning grounds claims in retrieved sources to reduce hallucination "
    "in scientific question answering."
)
_ANSWER = (
    "Machine learning grounds claims in retrieved sources to reduce hallucination "
    "in scientific question answering [0]."
)


def _responder(prompt: str, model: type[BaseModel]) -> BaseModel:
    """Cite [0] with a verbatim quote for answers; fixed rubric for quality."""
    if model is AnswerDraft:
        return AnswerDraft(text=_ANSWER, cited=[CitedMarker(marker=0, paper_id="", quote=_PHRASE)])
    if model is QualityRubric:
        return QualityRubric(organization=4, coverage=4, relevance=4)
    return model()  # pragma: no cover - defensive


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            id=f"c{i}",
            question="How does retrieval reduce hallucination in scientific QA?",
            reference_answer=_DOC_TEXT,
            ctxs=[
                RetrievedDoc(doc_id=f"c{i}-D0", title="Grounded generation", text=_DOC_TEXT),
                RetrievedDoc(
                    doc_id=f"c{i}-D1",
                    title="Retrieval methods",
                    text=f"Hybrid retrieval {_DOC_TEXT}",
                ),
            ],
            metrics=["citation", "quality"],
        )
        for i in range(2)
    ]


def test_ablation_discriminates_retrieval_and_is_honest() -> None:
    from lfx_insights.llm.client import MockLLM

    settings = load_settings(None)
    report = run_ablation(
        _cases(),
        conditions=["null", "tfidf"],
        llm=MockLLM(responder=_responder),
        settings=settings,
        dataset="test",
        judge="lexical",
    )

    by = {c.condition: c for c in report.conditions}
    assert by["null"].citation is not None and by["tfidf"].citation is not None
    # Closed-book cannot cite; tfidf retrieves the source and grounds the citation.
    assert by["null"].citation.value == 0.0
    assert by["tfidf"].citation.value > by["null"].citation.value

    # Honest Score: components, method and uncertainty are all present.
    cit = by["tfidf"].citation
    assert {c.name for c in cit.components} == {"citation_precision", "citation_recall"}
    assert cit.method and cit.uncertainty is not None and cit.interpretation

    # No perspicacite condition -> no lift; the judge caveat is always disclosed.
    assert report.lift == {}
    assert any("judge" in c.lower() for c in report.caveats)

    md = render_markdown(report)
    assert "ScholarQABench ablation" in md and "| null |" in md and "| tfidf |" in md


def test_eval_cli_offline_smoke() -> None:
    result = CliRunner().invoke(
        main,
        ["eval", "scholarqa", "--offline", "--conditions", "null,tfidf", "--judge", "lexical"],
    )
    assert result.exit_code == 0, result.output
    assert '"dataset": "bundled"' in result.output
    assert '"tfidf"' in result.output


def test_eval_cli_bad_condition_is_clean_error() -> None:
    # An invalid condition raises ValueError deep in the runner; the CLI must surface
    # it as a clean error, not an uncaught traceback.
    result = CliRunner().invoke(
        main, ["eval", "scholarqa", "--offline", "--conditions", "bogus", "--judge", "lexical"]
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "bogus" in result.output


def test_eval_cli_bad_judge_is_clean_error() -> None:
    result = CliRunner().invoke(
        main, ["eval", "scholarqa", "--offline", "--conditions", "tfidf", "--judge", "bogus"]
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "bogus" in result.output


def test_litsearch_retrieval_ablation() -> None:
    from lfx_insights.eval.dataset import load_dataset
    from lfx_insights.llm.client import MockLLM

    settings = load_settings(None)
    report = run_ablation(
        load_dataset("litsearch-bundled"),
        conditions=["null", "tfidf"],
        llm=MockLLM(),
        settings=settings,
        dataset="litsearch",
        metrics_override=["retrieval"],
    )
    by = {c.condition: c for c in report.conditions}
    assert by["null"].retrieval is not None and by["tfidf"].retrieval is not None
    # Closed-book retrieves nothing; TF-IDF over the pool finds the gold papers.
    assert by["null"].retrieval.value == 0.0
    assert by["tfidf"].retrieval.value > 0.9
    # Honest Score: exposes recall@k and nDCG@k components.
    assert {c.name for c in by["tfidf"].retrieval.components} == {"recall@k", "ndcg@k"}


def test_oracle_condition_grounds() -> None:
    from lfx_insights.llm.client import MockLLM

    settings = load_settings(None)
    report = run_ablation(
        _cases(),
        conditions=["null", "oracle"],
        llm=MockLLM(responder=_responder),
        settings=settings,
        dataset="test",
        judge="lexical",
    )
    by = {c.condition: c for c in report.conditions}
    # Oracle feeds the case's own gold ctxs -> the cited phrase grounds -> non-zero F1.
    assert by["oracle"].citation is not None and by["oracle"].citation.value > 0.0
    assert by["null"].citation is not None and by["null"].citation.value == 0.0


def test_ground_generation_self_verifies_citations() -> None:
    from lfx_insights.llm.client import MockLLM

    settings = load_settings(None)
    settings.eval.ground_generation = True
    report = run_ablation(
        _cases(),
        conditions=["tfidf"],
        llm=MockLLM(responder=_responder),
        settings=settings,
        dataset="test",
        judge="lexical",
    )
    [tfidf] = report.conditions
    # Every kept citation was entailment-checked at generation time, so the scored
    # precision is perfect (no kept-but-unsupported citation survives).
    assert tfidf.citation is not None and tfidf.citation.value > 0.0
    prec = next(c.value for c in tfidf.citation.components if c.name == "citation_precision")
    assert prec == 1.0
