"""FastMCP server exposing Consilium's research-copilot capabilities as tools.

Each tool builds a corpus via the configured Perspicacité backend (or in-memory
fakes when ``offline=True``) and returns structured, JSON-serializable results.
Tools that need an LLM (theme labels, generation) use the configured model; in
offline mode they use the deterministic mock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from consilium.config import load_settings
from consilium.context import build_context
from consilium.generation.grant import draft_grant
from consilium.generation.hypotheses import generate_hypotheses
from consilium.generation.manuscript import draft_manuscript
from consilium.generation.questions import generate_questions
from consilium.generation.reviewer_sim import simulate_review
from consilium.lifescience.bioinformatics import detect_omics
from consilium.lifescience.datasets import discover_datasets
from consilium.lifescience.protocols import AVAILABLE_PROTOCOLS, generate_protocol
from consilium.lifescience.reproducibility import audit_reproducibility
from consilium.lifescience.statistics import recommend_sample_size
from consilium.lifescience.study_design import recommend_designs
from consilium.scoring.evidence_strength import score_evidence_strength
from consilium.scoring.funding_alignment import align_funding
from consilium.scoring.gap_validation import validate_gaps as run_gap_validation
from consilium.scoring.meta_analysis_readiness import assess_meta_readiness
from consilium.scoring.novelty import score_novelty
from consilium.scoring.opportunity import rank_opportunities
from consilium.themes.discover import discover_themes
from consilium.themes.label import label_themes

if TYPE_CHECKING:
    from consilium.context import RunContext
    from consilium.models import Corpus, Theme

_DEFAULT_MAX_PAPERS = 30


def build_server(*, offline: bool = False) -> Any:
    """Build the Consilium FastMCP server. ``offline`` uses in-memory fakes."""
    from fastmcp import FastMCP

    mcp: Any = FastMCP("consilium")

    def _ctx(topic: str) -> RunContext:
        return build_context(load_settings(None), offline=offline, topic=topic)

    def _corpus(ctx: RunContext, topic: str, max_papers: int) -> Corpus:
        return ctx.backend.build_or_select_kb(topic, max_papers=max_papers)

    def _themes(ctx: RunContext, corpus: Corpus) -> list[Theme]:
        return label_themes(discover_themes(corpus.papers, ctx.embedder), corpus, ctx.llm)

    @mcp.tool()
    def themes(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Discover and label research themes in the literature on a topic."""
        ctx = _ctx(topic)
        return [t.model_dump() for t in _themes(ctx, _corpus(ctx, topic, max_papers))]

    @mcp.tool()
    def validate_gaps(
        topic: str, gaps: list[str], max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Validate claimed research gaps against the corpus (Confirmed/Uncertain/Not Supported)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in run_gap_validation(gaps, corpus, ctx.embedder)]

    @mcp.tool()
    def novelty(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Score each theme's novelty/emergence (recency, growth, scarcity)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in score_novelty(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def opportunities(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Rank research opportunities per theme (emergence, scarcity, momentum)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in rank_opportunities(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def evidence_strength(
        topic: str, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Score the evidence strength of each theme (study count, recency, source diversity)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in score_evidence_strength(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def funding_alignment(
        topic: str, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Align each theme with funding priority areas."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in align_funding(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def meta_analysis_readiness(
        topic: str, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Assess meta-analysis readiness per theme (heuristic)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in assess_meta_readiness(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def hypotheses(
        topic: str, n: int = 5, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Generate grounded, testable hypotheses (citations quote-verified against the corpus)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [h.model_dump() for h in generate_hypotheses(corpus, ctx.llm, n=n)]

    @mcp.tool()
    def questions(
        topic: str, n: int = 10, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Generate ranked research questions for a topic."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [q.model_dump() for q in generate_questions(corpus, ctx.llm, n=n)]

    @mcp.tool()
    def manuscript(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Draft manuscript sections grounded in the corpus."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [s.model_dump() for s in draft_manuscript(corpus, ctx.llm)]

    @mcp.tool()
    def grant(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Draft grant proposal sections grounded in the corpus."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [s.model_dump() for s in draft_grant(corpus, ctx.llm)]

    @mcp.tool()
    def reviewer_simulation(
        topic: str, max_papers: int = _DEFAULT_MAX_PAPERS
    ) -> list[dict[str, Any]]:
        """Simulate peer review of a drafted manuscript for the topic."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        sections = draft_manuscript(corpus, ctx.llm)
        return [c.model_dump() for c in simulate_review(sections, corpus, ctx.llm)]

    @mcp.tool()
    def study_design(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Recommend a study design per theme (OBI-aligned, maturity-gated)."""
        ctx = _ctx(topic)
        corpus = _corpus(ctx, topic, max_papers)
        return [i.model_dump() for i in recommend_designs(_themes(ctx, corpus), corpus)]

    @mcp.tool()
    def bioinformatics(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Detect omics data types and map them to repositories (EDAM-aligned, assay-aware)."""
        ctx = _ctx(topic)
        return [i.model_dump() for i in detect_omics(_corpus(ctx, topic, max_papers))]

    @mcp.tool()
    def reproducibility(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Audit corpus papers for reproducibility (weighted 6-dimension score)."""
        ctx = _ctx(topic)
        return [i.model_dump() for i in audit_reproducibility(_corpus(ctx, topic, max_papers))]

    @mcp.tool()
    def datasets(topic: str, max_papers: int = _DEFAULT_MAX_PAPERS) -> list[dict[str, Any]]:
        """Discover datasets referenced in the corpus (by accession)."""
        ctx = _ctx(topic)
        return [i.model_dump() for i in discover_datasets(_corpus(ctx, topic, max_papers))]

    @mcp.tool()
    def sample_size(
        design: str,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        groups: int = 2,
    ) -> dict[str, Any]:
        """Recommend a sample size / power analysis (scipy/statsmodels-backed).

        design: two_sample_t | paired_t | one_way_anova | two_proportions | correlation.
        """
        rec = recommend_sample_size(design, effect_size, alpha=alpha, power=power, groups=groups)
        return rec.model_dump()

    @mcp.tool()
    def protocol(kind: str) -> dict[str, Any]:
        """Generate a lab/bioinformatics protocol checklist.

        kind: one of rna_seq | variant_calling | pcr | western_blot.
        """
        return generate_protocol(kind).model_dump()

    @mcp.tool()
    def list_protocols() -> list[str]:
        """List the available protocol templates."""
        return list(AVAILABLE_PROTOCOLS)

    return mcp
