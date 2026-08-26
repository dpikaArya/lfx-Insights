"""Ablation runner: drive each retrieval condition over a dataset and aggregate.

For every (condition, case) the runner retrieves a corpus, synthesises a grounded
answer, and applies the case's metrics. Per-condition aggregates are honest
:class:`~consilium.models.Score`s (components/weights/method/interpretation/
uncertainty) â€” never bare means â€” and the headline ``lift`` isolates PerspicacitÃ©'s
contribution (``perspicacite - tfidf`` on citation F1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx_insights.eval.answer import answer_question
from lfx_insights.eval.dataset import candidate_pool
from lfx_insights.eval.entailment import build_entailer
from lfx_insights.eval.metrics.citation import compute_citation_prf
from lfx_insights.eval.metrics.correctness import match_score, remove_citations, rouge_l
from lfx_insights.eval.metrics.quality import judge_quality
from lfx_insights.eval.metrics.retrieval import score_retrieval
from lfx_insights.eval.models import (
    AblationReport,
    CaseResult,
    CitationScore,
    ConditionReport,
    CorrectnessScore,
    QualityScore,
    RetrievalScore,
)
from lfx_insights.eval.retrieval import build_eval_backend
from lfx_insights.models import Provenance, Score, ScoreComponent
from lfx_insights.scoring.common import band, clamp01, make_score, sample_uncertainty

if TYPE_CHECKING:
    from lfx_insights.config import Settings
    from lfx_insights.eval.models import EvalCase
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus, Paper

# Disclosed deviations from the paper (so partial parity is never read as full parity).
_JUDGE_CAVEAT = (
    "Citation entailment judge is '{judge}', not the paper's AttrScore/TRUE-NLI; "
    "absolute citation numbers are not bit-comparable to ScholarQABench."
)
_PERSP_CAVEAT = (
    "PerspicacitÃ© retrieves over a different corpus than peS2o(45M); only the "
    "open-retrieval comparison is meaningful, not 'provided datastore' parity."
)
_QUALITY_CAVEAT = (
    "Answer-quality is an LLM-judge approximation of Prometheus (organization/"
    "coverage/relevance), not the 8x7B checkpoints."
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm(s: str) -> str:
    """Normalise an identifier/title for gold-vs-retrieved matching."""
    return " ".join(s.lower().split())


def _paper_keys(paper: Paper) -> set[str]:
    """The identifiers a retrieved paper could match gold on: id, DOI, normalised title."""
    keys = {_norm(paper.id)}
    if paper.doi:
        keys.add(_norm(paper.doi))
    if paper.title:
        keys.add(_norm(paper.title))
    return {k for k in keys if k}


def _citation_aggregate(scores: list[CitationScore]) -> Score | None:
    """Honest aggregate whose value is the mean per-case F1; components expose P/R."""
    if not scores:
        return None
    mp, mr, mf = (
        _mean([s.precision for s in scores]),
        _mean([s.recall for s in scores]),
        _mean([s.f1 for s in scores]),
    )
    return Score(
        value=clamp01(mf),
        components=[
            ScoreComponent(name="citation_precision", value=clamp01(mp), weight=0.5),
            ScoreComponent(name="citation_recall", value=clamp01(mr), weight=0.5),
        ],
        method="mean per-case citation F1 (AutoAIS); components are mean precision/recall",
        interpretation=band(mf),
        uncertainty=sample_uncertainty(len(scores)),
    )


def _correctness_aggregate(scores: list[CorrectnessScore]) -> Score | None:
    if not scores:
        return None
    by_metric: dict[str, list[float]] = {}
    for s in scores:
        by_metric.setdefault(s.metric, []).append(s.value)
    components = [
        ScoreComponent(name=m, value=clamp01(_mean(v)), weight=float(len(v)))
        for m, v in by_metric.items()
    ]
    overall = _mean([s.value for s in scores])
    return Score(
        value=clamp01(overall),
        components=components,
        method="mean per-case correctness (match / rouge_l)",
        interpretation=band(overall),
        uncertainty=sample_uncertainty(len(scores)),
    )


def _retrieval_aggregate(scores: list[RetrievalScore]) -> Score | None:
    if not scores:
        return None
    return make_score(
        [
            ScoreComponent(
                name="recall@k", value=clamp01(_mean([s.recall for s in scores])), weight=1.0
            ),
            ScoreComponent(
                name="ndcg@k", value=clamp01(_mean([s.ndcg for s in scores])), weight=1.0
            ),
        ],
        method=f"mean per-case retrieval (recall@k & nDCG@k, k={scores[0].k})",
        uncertainty=sample_uncertainty(len(scores)),
    )


def _quality_aggregate(scores: list[QualityScore]) -> Score | None:
    if not scores:
        return None
    components = [
        ScoreComponent(
            name="organization", value=_mean([s.organization for s in scores]), weight=1.0
        ),
        ScoreComponent(name="coverage", value=_mean([s.coverage for s in scores]), weight=1.0),
        ScoreComponent(name="relevance", value=_mean([s.relevance for s in scores]), weight=1.0),
    ]
    return make_score(
        components,
        method="mean per-case quality (Prometheus-approx, 3 aspects)",
        uncertainty=sample_uncertainty(len(scores)),
    )


def _score_case(
    case: EvalCase,
    condition: str,
    *,
    pool: list[Paper],
    llm: LLMClient,
    settings: Settings,
    metrics: list[str],
    entailer_judge: str,
) -> CaseResult:
    from lfx_insights.models import Paper

    case_docs = [
        Paper(id=d.doc_id, title=d.title, abstract=d.text, source="oracle") for d in case.ctxs
    ]
    backend = build_eval_backend(
        condition,
        pool=pool,
        case_docs=case_docs,
        url=settings.perspicacite.url,
        timeout=settings.perspicacite.timeout,
    )
    corpus = backend.build_or_select_kb(case.question, max_papers=settings.eval.retrieval_k)
    gen_entailer = (
        build_entailer(
            settings.eval.generation_judge, threshold=settings.eval.lexical_threshold, llm=llm
        )
        if settings.eval.ground_generation
        else None
    )
    answer = answer_question(
        case.question, corpus, llm, max_docs=settings.eval.retrieval_k, entailer=gen_entailer
    )

    citation: CitationScore | None = None
    if "citation" in metrics:
        entailer = build_entailer(
            entailer_judge,
            threshold=settings.eval.lexical_threshold,
            llm=llm if entailer_judge == "llm" else None,
        )
        citation = compute_citation_prf(answer, entailer)

    correctness: CorrectnessScore | None = None
    if "match" in metrics and case.gold_label:
        correctness = CorrectnessScore(
            metric="match", value=match_score(answer.text, case.gold_label)
        )
    elif "rouge" in metrics and case.reference_answer:
        correctness = CorrectnessScore(
            metric="rouge_l", value=rouge_l(remove_citations(answer.text), case.reference_answer)
        )

    quality: QualityScore | None = None
    if "quality" in metrics and case.reference_answer:
        quality = judge_quality(answer, case.reference_answer, llm)

    retrieval = _score_retrieval_case(case, corpus, metrics, settings)

    return CaseResult(
        case_id=case.id,
        condition=condition,
        answer=answer,
        n_retrieved=len(corpus),
        citation=citation,
        correctness=correctness,
        quality=quality,
        retrieval=retrieval,
    )


def _score_retrieval_case(
    case: EvalCase, corpus: Corpus, metrics: list[str], settings: Settings
) -> RetrievalScore | None:
    """Intrinsic retrieval score: did this condition fetch the case's gold papers."""
    if "retrieval" not in metrics or not case.gold_docs:
        return None
    gold = {_norm(g) for g in case.gold_docs}
    keys = [_paper_keys(p) for p in corpus.papers]  # rank order
    return score_retrieval(keys, gold, settings.eval.retrieval_k)


def run_ablation(
    cases: list[EvalCase],
    *,
    conditions: list[str],
    llm: LLMClient,
    settings: Settings,
    dataset: str = "dataset",
    metrics_override: list[str] | None = None,
    judge: str | None = None,
    max_cases: int = 0,
) -> AblationReport:
    """Run the retrieval ablation over ``cases`` for each condition.

    ``metrics_override`` forces the metric set for every case (still gated by data
    availability); otherwise each case's own ``metrics`` apply. ``judge`` selects the
    citation entailer (default ``settings.eval.judge``). ``max_cases`` (>0) caps the
    case count and is recorded as a caveat. Aggregates are honest Scores.
    """
    caveats: list[str] = []
    if max_cases > 0 and len(cases) > max_cases:
        caveats.append(f"Capped: evaluated the first {max_cases} of {len(cases)} cases.")
        cases = cases[:max_cases]

    entailer_judge = judge or settings.eval.judge
    caveats.append(_JUDGE_CAVEAT.format(judge=entailer_judge))
    pool = candidate_pool(cases)

    condition_reports: list[ConditionReport] = []
    all_results: list[CaseResult] = []
    for condition in conditions:
        results = [
            _score_case(
                case,
                condition,
                pool=pool,
                llm=llm,
                settings=settings,
                metrics=metrics_override or case.metrics,
                entailer_judge=entailer_judge,
            )
            for case in cases
        ]
        all_results.extend(results)
        cond_caveats: list[str] = []
        if condition == "perspicacite":
            cond_caveats.append(_PERSP_CAVEAT)
        if any(r.quality is not None for r in results):
            cond_caveats.append(_QUALITY_CAVEAT)
        condition_reports.append(
            ConditionReport(
                condition=condition,
                n_cases=len(results),
                citation=_citation_aggregate([r.citation for r in results if r.citation]),
                correctness=_correctness_aggregate(
                    [r.correctness for r in results if r.correctness]
                ),
                quality=_quality_aggregate([r.quality for r in results if r.quality]),
                retrieval=_retrieval_aggregate([r.retrieval for r in results if r.retrieval]),
                caveats=cond_caveats,
            )
        )

    lift = _compute_lift(condition_reports)
    return AblationReport(
        dataset=dataset,
        conditions=condition_reports,
        cases=all_results,
        lift=lift,
        caveats=caveats,
        provenance=Provenance(generated_by="lfx-insights.eval", model=settings.llm.model),
    )


def _compute_lift(reports: list[ConditionReport]) -> dict[str, float]:
    """Headline lift: perspicacite - tfidf on citation F1 and quality, when both ran."""
    by_cond = {r.condition: r for r in reports}
    persp, tfidf = by_cond.get("perspicacite"), by_cond.get("tfidf")
    lift: dict[str, float] = {}
    if persp and tfidf:
        if persp.citation and tfidf.citation:
            lift["citation_f1"] = round(persp.citation.value - tfidf.citation.value, 4)
        if persp.quality and tfidf.quality:
            lift["quality"] = round(persp.quality.value - tfidf.quality.value, 4)
    return lift
