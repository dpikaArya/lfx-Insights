"""lfx Insights CLI (click)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from lfx_insights import __version__
from lfx_insights.config import load_settings
from lfx_insights.context import build_context
from lfx_insights.errors import ConsiliumError
from lfx_insights.io.store import OutputStore
from lfx_insights.lifescience.protocols import AVAILABLE_PROTOCOLS, generate_protocol
from lfx_insights.lifescience.statistics import recommend_sample_size
from lfx_insights.llm.client import LLMClient, MockLLM
from lfx_insights.models import SectionBundle
from lfx_insights.pipeline import PIPELINE
from lfx_insights.pipeline import run as run_pipeline
from lfx_insights.reporting.docx_export import write_docx
from lfx_insights.reporting.lifescience_report import render_protocol, render_stat


@click.group()
def main() -> None:
    """Consilium â€” research strategy & authoring copilot (layered on PerspicacitÃ©)."""


@main.command()
def version() -> None:
    """Print the lfx Insights version."""
    click.echo(__version__)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def themes(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Discover and label research themes for a topic."""
    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline, output_dir=output_dir)
    try:
        summary = run_pipeline(topic, ctx, stages=["themes"])
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, indent=2))


def _run_stage(
    stage: str,
    topic: str,
    offline: bool,
    config_path: str | None,
    output_dir: str | None,
    gaps: tuple[str, ...] = (),
) -> None:
    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline, output_dir=output_dir)
    if gaps:
        ctx.gaps = list(gaps)
    try:
        summary = run_pipeline(topic, ctx, stages=[stage])
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def novelty(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Score theme novelty / emergence."""
    _run_stage("novelty", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def opportunities(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Rank research opportunities."""
    _run_stage("opportunity", topic, offline, config_path, output_dir)


@main.command(name="evidence-strength")
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def evidence_strength(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Score evidence strength per theme."""
    _run_stage("evidence_strength", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def funding(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Align themes with funding priority areas."""
    _run_stage("funding", topic, offline, config_path, output_dir)


@main.command(name="meta-analysis")
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def meta_analysis(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Assess meta-analysis readiness per theme."""
    _run_stage("meta_analysis", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option(
    "--gap", "gaps", multiple=True, required=True, help="A claimed research gap (repeatable)."
)
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def gaps(
    topic: str,
    gaps: tuple[str, ...],
    offline: bool,
    config_path: str | None,
    output_dir: str | None,
) -> None:
    """Validate claimed research gaps against the corpus."""
    _run_stage("gaps", topic, offline, config_path, output_dir, gaps=gaps)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def hypotheses(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Generate grounded, testable hypotheses (exported as indicium draft Claims)."""
    _run_stage("hypotheses", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def questions(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Generate ranked research questions."""
    _run_stage("questions", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def manuscript(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Draft manuscript sections grounded in the corpus."""
    _run_stage("manuscript", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def grant(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Draft grant proposal sections."""
    _run_stage("grant", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def review(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Simulate peer review of a drafted manuscript."""
    _run_stage("review", topic, offline, config_path, output_dir)


@main.command(name="study-design")
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def study_design(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Recommend study designs per theme (OBI-aligned)."""
    _run_stage("study_design", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def bioinformatics(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Detect omics data types and map to repositories (EDAM-aligned)."""
    _run_stage("bioinformatics", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def reproducibility(
    topic: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Audit corpus papers for reproducibility (weighted, 6 dimensions)."""
    _run_stage("reproducibility", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def datasets(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Discover datasets referenced in the corpus (by accession)."""
    _run_stage("datasets", topic, offline, config_path, output_dir)


@main.command()
@click.option(
    "--design",
    required=True,
    type=click.Choice(
        ["two_sample_t", "paired_t", "one_way_anova", "two_proportions", "correlation"]
    ),
    help="Statistical design.",
)
@click.option("--effect-size", "effect_size", required=True, type=float, help="Effect size.")
@click.option("--alpha", default=0.05, type=float, help="Significance level.")
@click.option("--power", default=0.80, type=float, help="Target power.")
@click.option("--groups", default=2, type=int, help="Number of groups (ANOVA).")
@click.option("--output-dir", default=None, help="Output directory.")
def stats(
    design: str,
    effect_size: float,
    alpha: float,
    power: float,
    groups: int,
    output_dir: str | None,
) -> None:
    """Recommend a sample size / power analysis (scipy/statsmodels-backed)."""
    try:
        rec = recommend_sample_size(design, effect_size, alpha=alpha, power=power, groups=groups)
    except (ValueError, ConsiliumError) as exc:
        raise click.ClickException(str(exc)) from exc
    OutputStore(output_dir or "outputs").write_markdown("statistics.md", render_stat(rec))
    click.echo(rec.model_dump_json(indent=2))


@main.command()
@click.option(
    "--kind",
    required=True,
    type=click.Choice(AVAILABLE_PROTOCOLS),
    help="Protocol template.",
)
@click.option("--output-dir", default=None, help="Output directory.")
def protocol(kind: str, output_dir: str | None) -> None:
    """Generate a lab/bioinformatics protocol checklist."""
    try:
        proto = generate_protocol(kind)
    except (ValueError, ConsiliumError) as exc:
        raise click.ClickException(str(exc)) from exc
    OutputStore(output_dir or "outputs").write_markdown(
        f"protocol_{kind}.md", render_protocol(proto)
    )
    click.echo(proto.model_dump_json(indent=2))


@main.command(name="export-docx")
@click.option(
    "--run", "run_dir", default="outputs/default", help="Run directory holding the artifact."
)
@click.option(
    "--artifact",
    type=click.Choice(["manuscript", "grant"]),
    default="manuscript",
    help="Which section artifact to export.",
)
@click.option("--output-dir", default=None, help="Where to write the .docx (default: the run dir).")
def export_docx(run_dir: str, artifact: str, output_dir: str | None) -> None:
    """Render a grounded section artifact (manuscript/grant) to an APA .docx."""
    src = Path(run_dir) / f"{artifact}.sections.json"
    if not src.exists():
        raise click.ClickException(f"Artifact not found: {src} (run the '{artifact}' stage first)")
    bundle = SectionBundle.model_validate_json(src.read_text(encoding="utf-8"))
    out_dir = Path(output_dir) if output_dir else Path(run_dir)
    try:
        path = write_docx(bundle, out_dir / f"{artifact}.docx")
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(path))


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def dashboard(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Aggregate a run's artifacts into a single-page dashboard."""
    _run_stage("dashboard", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def brief(topic: str, offline: bool, config_path: str | None, output_dir: str | None) -> None:
    """Generate a unified research brief from a run's artifacts."""
    _run_stage("brief", topic, offline, config_path, output_dir)


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--paper-ids", "paper_ids_str", required=True, help="Comma-separated paper IDs.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def compare(
    topic: str, paper_ids_str: str, offline: bool, config_path: str | None, output_dir: str | None
) -> None:
    """Compare multiple papers across standard dimensions."""
    from lfx_insights.projects.comparison import ComparisonEngine

    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline, output_dir=output_dir)
    paper_ids = [p.strip() for p in paper_ids_str.split(",") if p.strip()]
    try:
        ctx.corpus = ctx.backend.build_or_select_kb(topic, max_papers=30)
        engine = ComparisonEngine(ctx.store.run_dir)
        result = engine.compare(paper_ids, ctx.corpus, ctx.llm)
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.model_dump(), indent=2))


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--paper-id", "paper_id", required=True, help="Paper ID to chat about.")
@click.option("--question", required=True, help="Question about the paper.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
def chat(
    topic: str,
    paper_id: str,
    question: str,
    offline: bool,
    config_path: str | None,
) -> None:
    """Ask a question about a specific paper."""
    from lfx_insights.projects.paper_chat import PaperChatSession

    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline)
    try:
        ctx.corpus = ctx.backend.build_or_select_kb(topic, max_papers=30)
        paper = ctx.corpus.by_id(paper_id)
        chunks = [paper.text()] if paper else []
        session = PaperChatSession(paper_id, ctx.store.run_dir)
        entry = session.ask(question, chunks=chunks, llm=ctx.llm)
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(entry.model_dump(), indent=2))


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--claim", required=True, help="Scientific claim to evaluate.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
def evaluate(
    topic: str, claim: str, offline: bool, config_path: str | None
) -> None:
    """Evaluate a claim against the corpus (evidence-grounded)."""
    from lfx_insights.projects.self_evaluation import SelfEvaluator

    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline)
    try:
        ctx.corpus = ctx.backend.build_or_select_kb(topic, max_papers=30)
        evaluator = SelfEvaluator(ctx.store.run_dir)
        result = evaluator.evaluate_claim(
            claim, topic, ctx.corpus, ctx.llm, ctx.embedder
        )
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.model_dump(), indent=2))


@main.command(name="validate-citations")
@click.option(
    "--run", "run_dir", default="outputs/default", help="Run directory holding the artifact."
)
@click.option(
    "--artifact",
    type=click.Choice(["manuscript", "grant"]),
    default="manuscript",
    help="Which section artifact to validate.",
)
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
def validate_citations(
    run_dir: str, artifact: str, topic: str, offline: bool, config_path: str | None
) -> None:
    """Validate citations in a generated artifact against the corpus."""
    from lfx_insights.generation.common import (
        build_cited_reference_list,
        build_evidence_chain,
        format_reference_list,
        validate_manuscript_citations,
    )

    src = Path(run_dir) / f"{artifact}.sections.json"
    if not src.exists():
        raise click.ClickException(f"Artifact not found: {src} (run the '{artifact}' stage first)")
    bundle = SectionBundle.model_validate_json(src.read_text(encoding="utf-8"))

    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline)
    try:
        ctx.corpus = ctx.backend.build_or_select_kb(topic, max_papers=30)
        validation = validate_manuscript_citations(bundle.sections, ctx.corpus)
        evidence_chain = build_evidence_chain(bundle.sections, ctx.corpus)
        ref_list = build_cited_reference_list(bundle.sections, ctx.corpus)
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc

    result = {
        "validation": validation,
        "reference_list": format_reference_list(ref_list),
        "evidence_chain": evidence_chain,
    }
    click.echo(json.dumps(result, indent=2))


@main.command()
@click.option("--topic", required=True, help="Research topic / query.")
@click.option("--quick", is_flag=True, help="Run the quick stage set.")
@click.option("--life-science", "life_science", is_flag=True, help="Run the life-science set.")
@click.option("--only", default=None, help="Comma-separated stages to run.")
@click.option("--skip", default=None, help="Comma-separated stages to skip.")
@click.option("--until", default=None, help="Run up to (and including) this stage.")
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def run(
    topic: str,
    quick: bool,
    life_science: bool,
    only: str | None,
    skip: str | None,
    until: str | None,
    offline: bool,
    config_path: str | None,
    output_dir: str | None,
) -> None:
    """Run the lfx Insights pipeline."""
    settings = load_settings(config_path)
    ctx = build_context(settings, offline=offline, output_dir=output_dir)

    stages: list[str] | None = None
    if only:
        stages = [s.strip() for s in only.split(",") if s.strip()]
    elif quick:
        stages = settings.pipeline.quick
    elif life_science:
        stages = settings.pipeline.life_science
    else:
        # No stage-set flag: default to the full pipeline so --skip/--until apply.
        stages = [s.name for s in PIPELINE]

    if stages and skip:
        skip_set = {s.strip() for s in skip.split(",")}
        stages = [s for s in stages if s not in skip_set]
    if stages and until and until in stages:
        stages = stages[: stages.index(until) + 1]

    try:
        summary = run_pipeline(topic, ctx, stages=stages)
    except ConsiliumError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option(
    "--transport", type=click.Choice(["stdio", "http"]), default="stdio", help="MCP transport."
)
@click.option("--host", default="127.0.0.1", help="Host for the http transport.")
@click.option("--port", default=8100, type=int, help="Port for the http transport.")
@click.option("--offline", is_flag=True, help="Serve with in-memory fakes (demo/testing).")
def serve(transport: str, host: str, port: int, offline: bool) -> None:
    """Run lfx Insights as an MCP server (requires the `mcp` extra)."""
    from lfx_insights.mcp import build_server

    try:
        server = build_server(offline=offline)
    except ImportError as exc:  # fastmcp not installed
        raise click.ClickException(
            "MCP support needs the 'mcp' extra. Install with: uv sync --extra mcp"
        ) from exc
    if transport == "http":
        server.run(transport="http", host=host, port=port)
    else:
        server.run()


@main.group()
def eval() -> None:
    """Evaluation harness (ScholarQABench-style retrieval ablation)."""


@eval.command(name="scholarqa")
@click.option(
    "--dataset",
    default="bundled",
    help="'bundled'/'expertqa-bundled'/'litsearch-bundled', or a path to a .jsonl/.json "
    "dataset (ScholarQABench / ExpertQA / LitSearch shape, auto-detected).",
)
@click.option(
    "--conditions",
    default=None,
    help="Comma-separated retrieval conditions (null,tfidf,oracle,perspicacite). Default: config.",
)
@click.option(
    "--metrics",
    default=None,
    help="Comma-separated metric override (citation,match,rouge,quality,retrieval). "
    "Default: per-case.",
)
@click.option("--judge", default=None, help="Citation entailer: lexical|grounding|llm.")
@click.option("--max-cases", "max_cases", default=None, type=int, help="Cap cases (0 = all).")
@click.option(
    "--ground-generation",
    "ground_generation",
    is_flag=True,
    help="Self-ground citations at generation time (drop misattributed + add uncited-supported).",
)
@click.option(
    "--generation-judge",
    "generation_judge",
    default=None,
    help="Entailer for --ground-generation: lexical|grounding|llm. Default: config (lexical).",
)
@click.option("--offline", is_flag=True, help="Use in-memory fakes (no network/LLM).")
@click.option("--config", "config_path", default=None, help="Path to config.yml.")
@click.option("--output-dir", default=None, help="Output directory.")
def eval_scholarqa(
    dataset: str,
    conditions: str | None,
    metrics: str | None,
    judge: str | None,
    max_cases: int | None,
    ground_generation: bool,
    generation_judge: str | None,
    offline: bool,
    config_path: str | None,
    output_dir: str | None,
) -> None:
    """Run the PerspicacitÃ©â†’Consilium retrieval ablation on a ScholarQABench dataset."""
    from lfx_insights.eval.dataset import load_dataset
    from lfx_insights.eval.report import render_markdown
    from lfx_insights.eval.runner import run_ablation

    settings = load_settings(config_path)
    if ground_generation:
        settings.eval.ground_generation = True
    if generation_judge:
        settings.eval.generation_judge = generation_judge
    conds = conditions.split(",") if conditions else list(settings.eval.conditions)
    if offline and "perspicacite" in conds:
        conds = [c for c in conds if c != "perspicacite"]
        click.echo("note: dropped 'perspicacite' condition (--offline).", err=True)
    metric_override = metrics.split(",") if metrics else None
    llm: LLMClient
    if offline:
        llm = MockLLM()
    else:
        from lfx_insights.llm.client import build_client, validate_ollama

        validate_ollama(settings)
        llm = build_client(settings)

    try:
        cases = load_dataset(dataset)
        report = run_ablation(
            cases,
            conditions=conds,
            llm=llm,
            settings=settings,
            dataset=dataset,
            metrics_override=metric_override,
            judge=judge,
            max_cases=settings.eval.max_cases if max_cases is None else max_cases,
        )
    except (ConsiliumError, ValueError) as exc:
        # ValueError covers an invalid --conditions/--judge (build_eval_backend /
        # build_entailer): surface it as a clean CLI error, not a traceback.
        raise click.ClickException(str(exc)) from exc

    store = OutputStore(output_dir or settings.output_dir, run="eval")
    store.write_json("eval_results.json", report.model_dump())
    store.write_markdown("eval_report.md", render_markdown(report))
    summary = {
        "dataset": report.dataset,
        "conditions": {
            c.condition: {
                "n_cases": c.n_cases,
                "citation_f1": None if c.citation is None else round(c.citation.value, 4),
                "quality": None if c.quality is None else round(c.quality.value, 4),
                "retrieval": None if c.retrieval is None else round(c.retrieval.value, 4),
            }
            for c in report.conditions
        },
        "lift": report.lift,
    }
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
