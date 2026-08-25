"""In-process pipeline (DAG of stages): themes slice + scoring layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from consilium.aggregate import insight_counts, load_run
from consilium.context import RunContext
from consilium.errors import ConsiliumError
from consilium.generation.common import build_section_bundle
from consilium.generation.grant import draft_grant
from consilium.generation.hypotheses import generate_hypotheses
from consilium.generation.manuscript import draft_manuscript
from consilium.generation.questions import generate_questions
from consilium.generation.reviewer_sim import simulate_review
from consilium.lifescience.bioinformatics import detect_omics
from consilium.lifescience.datasets import discover_datasets
from consilium.lifescience.reproducibility import audit_reproducibility
from consilium.lifescience.study_design import recommend_designs
from consilium.projects.project_manager import record_project
from consilium.projects.research_memory import record_run
from consilium.reporting.aggregate_report import (
    render_brief,
    render_dashboard,
    render_explainability,
)
from consilium.reporting.generation_report import (
    render_hypotheses,
    render_questions,
    render_review,
    render_sections,
)
from consilium.reporting.insights_report import render_insights
from consilium.reporting.themes_report import render_themes_report
from consilium.scoring.evidence_strength import score_evidence_strength
from consilium.scoring.funding_alignment import align_funding
from consilium.scoring.gap_validation import validate_gaps
from consilium.scoring.meta_analysis_readiness import assess_meta_readiness
from consilium.scoring.novelty import score_novelty
from consilium.scoring.opportunity import rank_opportunities
from consilium.standards.asb_export import run_to_capsule
from consilium.standards.astra_export import insights_to_collection
from consilium.standards.indicium_export import claims_to_document, sources_to_indicium
from consilium.themes.discover import discover_themes
from consilium.themes.label import label_themes

if TYPE_CHECKING:
    from consilium.models import Insight, Theme

StageFn = Callable[[RunContext, str], dict[str, object]]


@dataclass
class Stage:
    name: str
    fn: StageFn


def _ensure_themes(ctx: RunContext) -> list[Theme]:
    assert ctx.corpus is not None, "corpus must be built before stages run"
    if not ctx.themes:
        themes = discover_themes(ctx.corpus.papers, ctx.embedder)
        ctx.themes = label_themes(themes, ctx.corpus, ctx.llm)
    return ctx.themes


def _persist(ctx: RunContext, name: str, title: str, insights: list[Insight]) -> dict[str, object]:
    ctx.store.write_markdown(f"{name}.md", render_insights(title, insights))
    ctx.store.write_json(f"{name}.astra.json", insights_to_collection(insights, title=title))
    return {"insights": len(insights)}


def run_themes_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    themes = _ensure_themes(ctx)
    ctx.store.write_markdown("themes.md", render_themes_report(themes, ctx.corpus))
    ctx.store.write_json("indicium_sources.json", sources_to_indicium(ctx.corpus))
    return {"themes": len(themes)}


def run_evidence_strength_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx,
        "evidence_strength",
        "Evidence Strength",
        score_evidence_strength(_ensure_themes(ctx), ctx.corpus),
    )


def run_novelty_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(ctx, "novelty", "Novelty", score_novelty(_ensure_themes(ctx), ctx.corpus))


def run_opportunity_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx,
        "opportunities",
        "Research Opportunities",
        rank_opportunities(_ensure_themes(ctx), ctx.corpus),
    )


def run_funding_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx, "funding", "Funding Alignment", align_funding(_ensure_themes(ctx), ctx.corpus)
    )


def run_meta_analysis_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx,
        "meta_analysis",
        "Meta-analysis Readiness",
        assess_meta_readiness(_ensure_themes(ctx), ctx.corpus),
    )


def run_gaps_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    if not ctx.gaps:
        ctx.log.info("gaps_skipped", reason="no gaps provided")
        return {"skipped": "no gaps provided"}
    return _persist(
        ctx,
        "gaps",
        "Research Gap Validation",
        validate_gaps(ctx.gaps, ctx.corpus, ctx.embedder),
    )


def run_hypotheses_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    hyps = generate_hypotheses(ctx.corpus, ctx.llm)
    ctx.store.write_markdown("hypotheses.md", render_hypotheses(hyps))
    ctx.store.write_json("hypotheses.indicium.json", claims_to_document(hyps, ctx.corpus))
    return {"hypotheses": len(hyps)}


def run_questions_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    questions = generate_questions(ctx.corpus, ctx.llm)
    ctx.store.write_markdown("questions.md", render_questions(questions))
    return {"questions": len(questions)}


def run_manuscript_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    ctx.manuscript = draft_manuscript(ctx.corpus, ctx.llm)
    ctx.store.write_markdown(
        "manuscript.md", render_sections("Manuscript Draft", ctx.manuscript, ctx.corpus)
    )
    ctx.store.write_json(
        "manuscript.sections.json",
        build_section_bundle("Manuscript Draft", ctx.manuscript, ctx.corpus).model_dump(),
    )
    return {"sections": len(ctx.manuscript)}


def run_grant_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    sections = draft_grant(ctx.corpus, ctx.llm)
    ctx.store.write_markdown("grant.md", render_sections("Grant Draft", sections, ctx.corpus))
    ctx.store.write_json(
        "grant.sections.json",
        build_section_bundle("Grant Draft", sections, ctx.corpus).model_dump(),
    )
    return {"sections": len(sections)}


def run_review_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    sections = ctx.manuscript or draft_manuscript(ctx.corpus, ctx.llm)
    comments = simulate_review(sections, ctx.corpus, ctx.llm)
    ctx.store.write_markdown("review.md", render_review(comments))
    return {"comments": len(comments)}


def run_study_design_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx, "study_design", "Study Design", recommend_designs(_ensure_themes(ctx), ctx.corpus)
    )


def run_bioinformatics_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(ctx, "bioinformatics", "Bioinformatics", detect_omics(ctx.corpus))


def _full_texts(ctx: RunContext) -> dict[str, str] | None:
    """Fetch full text per paper from the backend when ``pipeline.full_text`` is on.

    Returns ``None`` (use abstracts) when disabled, settings are absent, or nothing
    could be fetched. Backend failures are skipped per-paper, not fatal.
    """
    assert ctx.corpus is not None
    if ctx.settings is None or not ctx.settings.pipeline.full_text:
        return None
    texts: dict[str, str] = {}
    for paper in ctx.corpus.papers:
        try:
            content = ctx.backend.paper_content(paper.id)
        except ConsiliumError:
            continue
        if content:
            texts[paper.id] = content
    return texts or None


def run_reproducibility_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(
        ctx,
        "reproducibility",
        "Reproducibility Audit",
        audit_reproducibility(ctx.corpus, _full_texts(ctx)),
    )


def run_datasets_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    return _persist(ctx, "datasets", "Datasets", discover_datasets(ctx.corpus, _full_texts(ctx)))


def run_kb_snapshot_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    assert ctx.corpus is not None
    themes = _ensure_themes(ctx)
    snapshot = {
        "topic": topic,
        "kb_id": ctx.corpus.kb_id,
        "n_papers": len(ctx.corpus),
        "themes": [t.model_dump() for t in themes],
        "artifacts": load_run(ctx.store),
    }
    ctx.store.write_json("knowledge_base.json", snapshot)
    return {"papers": len(ctx.corpus), "themes": len(themes)}


def run_explainability_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    run = load_run(ctx.store)
    ctx.store.write_markdown("explainability.md", render_explainability(run))
    return {"traced_stages": len(run["astra"])}


def run_dashboard_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    run = load_run(ctx.store)
    ctx.store.write_markdown("dashboard.md", render_dashboard(run))
    return {"artifacts": len(run["markdown"])}


def run_brief_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    run = load_run(ctx.store)
    ctx.store.write_markdown("brief.md", render_brief(run))
    return {"ok": True}


def run_capsule_stage(ctx: RunContext, topic: str) -> dict[str, object]:
    run = load_run(ctx.store)
    model = ctx.settings.llm.model if ctx.settings is not None else None
    ctx.store.write_json("run.capsule.json", run_to_capsule(topic, run, model=model))
    return {"artifacts": len(run["markdown"]) + len(run["astra"]) + len(run["indicium"])}


def _stages_summary(ctx: RunContext, executed: list[str] | None) -> dict[str, object]:
    """Record-summary stages: ALL executed stages, not just ASTRA-producing ones.

    ``insight_counts`` only sees stages that emitted ``*.astra.json`` artifacts, so
    on its own it drops themes/generation/lifescience stages. We merge it over the
    full executed-stage list so project/memory records reflect everything that ran.
    """
    assert ctx.corpus is not None
    counts = insight_counts(load_run(ctx.store))
    stages: dict[str, object] = {name: counts.get(name, 0) for name in (executed or [])}
    stages.update(counts)
    return {"kb_id": ctx.corpus.kb_id, "stages": stages}


def run_project_stage(
    ctx: RunContext, topic: str, executed: list[str] | None = None
) -> dict[str, object]:
    path = record_project(ctx.store.run_dir.parent, topic, _stages_summary(ctx, executed))
    return {"database": str(path)}


def run_memory_stage(
    ctx: RunContext, topic: str, executed: list[str] | None = None
) -> dict[str, object]:
    path = record_run(ctx.store.run_dir.parent, topic, _stages_summary(ctx, executed))
    return {"history": str(path)}


PIPELINE: list[Stage] = [
    Stage("themes", run_themes_stage),
    Stage("evidence_strength", run_evidence_strength_stage),
    Stage("novelty", run_novelty_stage),
    Stage("opportunity", run_opportunity_stage),
    Stage("funding", run_funding_stage),
    Stage("meta_analysis", run_meta_analysis_stage),
    Stage("gaps", run_gaps_stage),
    Stage("hypotheses", run_hypotheses_stage),
    Stage("questions", run_questions_stage),
    Stage("manuscript", run_manuscript_stage),
    Stage("grant", run_grant_stage),
    Stage("review", run_review_stage),
    Stage("study_design", run_study_design_stage),
    Stage("bioinformatics", run_bioinformatics_stage),
    Stage("reproducibility", run_reproducibility_stage),
    Stage("datasets", run_datasets_stage),
    Stage("kb_snapshot", run_kb_snapshot_stage),
    Stage("explainability", run_explainability_stage),
    Stage("dashboard", run_dashboard_stage),
    Stage("brief", run_brief_stage),
    Stage("capsule", run_capsule_stage),
    Stage("project", run_project_stage),
    Stage("memory", run_memory_stage),
]

_BY_NAME = {s.name: s for s in PIPELINE}


def _resolve(stages: list[str] | None) -> list[Stage]:
    if not stages:
        return list(PIPELINE)
    unknown = [name for name in stages if name not in _BY_NAME]
    if unknown:
        known = ", ".join(_BY_NAME)
        raise ConsiliumError(
            f"unknown pipeline stage(s): {', '.join(unknown)}. known stages: {known}"
        )
    return [_BY_NAME[name] for name in stages]


def run(
    topic: str,
    ctx: RunContext,
    stages: list[str] | None = None,
    max_papers: int = 30,
) -> dict[str, object]:
    """Build the corpus from Perspicacité, then run the selected stages."""
    ctx.corpus = ctx.backend.build_or_select_kb(topic, max_papers=max_papers)
    ctx.log.info("corpus_built", kb_id=ctx.corpus.kb_id, papers=len(ctx.corpus))
    summary: dict[str, object] = {"topic": topic, "kb_id": ctx.corpus.kb_id, "stages": {}}
    resolved = _resolve(stages)
    executed: list[str] = [s.name for s in resolved]
    for stage in resolved:
        ctx.log.info("stage_start", stage=stage.name)
        if stage.name == "project":
            result = run_project_stage(ctx, topic, executed)
        elif stage.name == "memory":
            result = run_memory_stage(ctx, topic, executed)
        else:
            result = stage.fn(ctx, topic)
        assert isinstance(summary["stages"], dict)
        summary["stages"][stage.name] = result
    return summary
