"""Gap validation: test whether a proposed research gap is covered by a corpus.

For each candidate gap, we embed the gap text alongside every paper in the corpus
and measure how well the corpus already covers it (max cosine similarity). The
verdict is scoped to THIS corpus â€” it is not a claim about the whole literature:

- *Not Supported* â€” already covered (high similarity).
- *Uncertain* â€” partially covered.
- *Confirmed* â€” on-topic but no close prior work in this corpus (a candidate gap).
- *Out of Scope* â€” similarity to the entire corpus is so low the gap is likely
  off-topic/unrelated rather than open; flagged with high uncertainty so a
  nonsense or off-domain "gap" is not rubber-stamped as confirmed.

Every verdict ships as an honest :class:`~consilium.models.Insight` whose
:class:`~consilium.models.Score` exposes its components and band.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Protocol

from lfx_insights.scoring.common import clamp01, cosine, make_score, sample_uncertainty

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from lfx_insights.models import Corpus, Insight

# Absolute backstop "related" floor, used only when the corpus has ~no internal
# coherence (e.g. mutually orthogonal vectors). Normally the EFFECTIVE related
# threshold is RELATIVE to the corpus's own coherence (see _RELATED_RATIO), like
# the off-topic floor â€” both adapt across embedders (TF-IDF vs sentence-transformers
# vs OpenAI), whose cosine magnitudes differ a lot.
_RELATED_FLOOR = 0.30
# A paper counts as "related" to the gap if its similarity is at or above this
# fraction of the corpus's internal coherence (median per-paper max-sim-to-others).
_RELATED_RATIO = 0.6
# Absolute backstop off-topic floor, used only when the corpus has ~no internal
# coherence (e.g. mutually orthogonal vectors). Normally the EFFECTIVE floor is
# RELATIVE to the corpus's own similarity scale (see _OFFTOPIC_RATIO), so the gate
# adapts across embedders (TF-IDF vs sentence-transformers vs OpenAI), whose cosine
# magnitudes differ a lot.
_OFFTOPIC_FLOOR = 0.12
# A gap is off-topic if its max similarity is below this fraction of the corpus's
# internal coherence (median per-paper max-similarity-to-other-papers).
_OFFTOPIC_RATIO = 0.5
# How many supporting papers to attach as evidence.
_TOP_EVIDENCE = 3


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float64]: ...


def _verdict(
    coverage: float, covered_threshold: float, offtopic_floor: float, related_threshold: float
) -> str:
    if coverage >= covered_threshold:
        return "Not Supported"
    if coverage < offtopic_floor:
        return "Out of Scope"
    if coverage <= related_threshold:
        return "Confirmed"
    return "Uncertain"


def _corpus_coherence(paper_rows: list[list[float]]) -> float:
    """Median of each paper's max cosine similarity to the OTHER papers.

    A proxy for how similar related work sits in THIS embedding space, so the
    off-topic gate can scale with the embedder rather than an absolute cutoff.
    Returns 0.0 for < 2 papers (no internal scale -> rely on the absolute floor).
    """
    n = len(paper_rows)
    if n < 2:
        return 0.0
    best = [max(cosine(paper_rows[i], paper_rows[j]) for j in range(n) if j != i) for i in range(n)]
    return float(statistics.median(best))


def validate_gaps(
    gaps: list[str],
    corpus: Corpus,
    embedder: Embedder,
    *,
    covered_threshold: float = 0.6,
    offtopic_floor: float = _OFFTOPIC_FLOOR,
    offtopic_ratio: float = _OFFTOPIC_RATIO,
    related_floor: float = _RELATED_FLOOR,
    related_ratio: float = _RELATED_RATIO,
) -> list[Insight]:
    """Validate each candidate ``gap`` against ``corpus`` coverage.

    Deterministic given a deterministic ``embedder``. Both thresholds are RELATIVE to
    the corpus's own coherence so the gates adapt across embedders whose cosine scales
    differ (TF-IDF vs sentence-transformers vs OpenAI): the off-topic floor is
    ``max(offtopic_floor, offtopic_ratio * coherence)`` and the "related" threshold
    (the Confirmed/Uncertain boundary, and what counts a paper as related) is
    ``max(related_floor, related_ratio * coherence)``. Returns one
    :class:`~consilium.models.Insight` per gap, in input order; ``[]`` for empty gaps
    or empty corpus.
    """
    from lfx_insights.models import EvidenceRef, Insight, ScoreComponent

    papers = corpus.papers
    if not gaps or not papers:
        return []

    n_papers = len(papers)
    # Embed gaps and papers together so they share one space (and one TF-IDF vocabulary).
    vectors = embedder.encode([*gaps, *(p.text() for p in papers)])
    gap_rows = [vectors[i].tolist() for i in range(len(gaps))]
    paper_rows = [vectors[len(gaps) + j].tolist() for j in range(n_papers)]

    coherence = _corpus_coherence(paper_rows)
    effective_floor = max(offtopic_floor, offtopic_ratio * coherence)
    related_threshold = max(related_floor, related_ratio * coherence)

    insights: list[Insight] = []
    for gi, gap in enumerate(gaps):
        sims = [cosine(gap_rows[gi], paper_rows[j]) for j in range(n_papers)]

        coverage = max(sims)
        n_related = sum(1 for s in sims if s > related_threshold)
        verdict = _verdict(coverage, covered_threshold, effective_floor, related_threshold)

        under_coverage = clamp01(1.0 - coverage)
        sparsity = clamp01(1.0 - n_related / n_papers)
        components = [
            ScoreComponent(name="under_coverage", value=under_coverage, weight=0.7),
            ScoreComponent(name="sparsity", value=sparsity, weight=0.3),
        ]
        # An off-topic gap is something we are LESS sure about, not more: bump uncertainty.
        uncertainty = sample_uncertainty(n_papers)
        if verdict == "Out of Scope":
            uncertainty = max(uncertainty, 0.8)
        score = make_score(components, uncertainty=uncertainty)

        ranked = sorted(range(n_papers), key=sims.__getitem__, reverse=True)
        evidence = [
            EvidenceRef(paper_id=papers[i].id, location="abstract") for i in ranked[:_TOP_EVIDENCE]
        ]

        reasoning = (
            f"Max similarity to this corpus is {coverage:.2f} (off-topic floor "
            f"{effective_floor:.2f}, corpus coherence {coherence:.2f}) with {n_related}/{n_papers} "
            f"papers above the related threshold {related_threshold:.2f}. Scoped to THIS corpus "
            "only. "
        )
        if verdict == "Not Supported":
            reasoning += (
                f"At or above the covered threshold ({covered_threshold:.2f}): the corpus "
                "already addresses this gap."
            )
        elif verdict == "Confirmed":
            reasoning += (
                "On-topic but no close prior work here â€” a candidate open gap; confirm "
                "against the broader literature."
            )
        elif verdict == "Out of Scope":
            reasoning += (
                f"Below the off-topic floor ({effective_floor:.2f}): the gap shares far less "
                "with this corpus than its papers share with each other, so it is more likely "
                "OFF-TOPIC than open â€” verify it is on-topic before treating it as a gap "
                "(high uncertainty)."
            )
        else:
            reasoning += "Partial coverage â€” related work exists but does not fully address it."

        insights.append(
            Insight(
                statement=(f"Gap '{gap}': {verdict} (max corpus similarity {coverage:.2f})."),
                evidence=evidence,
                is_synthesized=True,
                reasoning=reasoning,
                tags=["gap", verdict.lower().replace(" ", "_")],
                score=score,
            )
        )

    return insights
