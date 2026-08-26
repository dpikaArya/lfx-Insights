"""LLM-backed, corpus-grounded hypothesis generation.

Proposes falsifiable hypotheses in the Bucur SuperPattern (subject-qualifier-object)
form, grounded in a PerspicacitÃ© corpus. The direction/qualifier is supplied by the
LLM (never RNG). For every cited paper the LLM must emit a short verbatim quote drawn
from that paper's abstract; the (paper_id, quote) pairs are then passed through the
quote-grounding gate :func:`grounded_evidence` so any reference whose quote is not
verbatim present in the cited paper's text is dropped (anti-hallucination). The
surviving quotes are carried onto each :class:`EvidenceRef` so the indicium Evidence
later exposes a real ``textual_quotation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lfx_insights.generation.common import grounded_evidence, has_output_leak
from lfx_insights.models import EvidenceRef, Hypothesis, Provenance

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus


class CitedRef(BaseModel):
    """A single citation: a corpus paper id plus a verbatim supporting quote."""

    paper_id: str
    quote: str = ""


class HypothesisDraft(BaseModel):
    """One LLM-proposed hypothesis, before grounding/citation verification."""

    subject: str
    qualifier: str
    object: str
    statement: str
    rationale: str = ""
    independent_var: str | None = None
    dependent_var: str | None = None
    methodology: str | None = None
    evidence: list[CitedRef] = Field(default_factory=list)


class HypothesisBatch(BaseModel):
    """Batch wrapper so N hypotheses come back in a single structured call."""

    items: list[HypothesisDraft] = Field(default_factory=list)


def _build_prompt(corpus: Corpus, n: int) -> str:
    lines = [f"{p.id}: {p.title}" for p in corpus.papers]
    catalog = "\n".join(lines) if lines else "(no papers)"
    return (
        f"You are a research strategist. Using ONLY the papers below, propose up to "
        f"{n} falsifiable, testable hypotheses that advance this literature.\n\n"
        "Papers (cite ONLY these ids):\n"
        f"{catalog}\n\n"
        "For each hypothesis provide:\n"
        "- subject and object: the two entities related by the claim\n"
        "- qualifier: the directional relation, drawn from the Bucur SuperPattern "
        "vocabulary (e.g. causes, prevents, inhibits, activates, increases, decreases, "
        "correlates_with, is_associated_with, predicts, interacts_with, produces, "
        "requires, enables, treats, enhances, reduces). Choose the qualifier that "
        "reflects the actual claimed direction; do not guess randomly.\n"
        "- statement: a one-sentence falsifiable hypothesis (no template placeholders)\n"
        "- rationale: why the corpus motivates it\n"
        "- independent_var / dependent_var / methodology: how it could be tested\n"
        "- evidence: for each supporting paper, an item with 'paper_id' (taken ONLY "
        "from the list above) AND 'quote', a SHORT verbatim quote copied EXACTLY "
        "(word for word) from that paper's abstract that supports the hypothesis. "
        "Do not paraphrase the quote and do not cite a paper you cannot quote.\n"
    )


def generate_hypotheses(corpus: Corpus, llm: LLMClient, *, n: int = 5) -> list[Hypothesis]:
    """Generate up to ``n`` corpus-grounded, falsifiable hypotheses.

    The qualifier/direction comes from the LLM (never RNG). Drafts whose statement
    leaks an unrendered placeholder or ``NaN`` are rejected. Each citation must carry
    a verbatim supporting quote; the (paper_id, quote) pairs pass through
    :func:`grounded_evidence`, so a citation with no quote, an unresolvable paper, or
    a quote not verbatim-present in the cited paper is dropped (anti-hallucination).
    The surviving quotes are attached to each :class:`EvidenceRef`. A hypothesis with
    no surviving grounded evidence is still allowed (``evidence=[]``).
    """
    prompt = _build_prompt(corpus, n)
    batch = llm.complete_structured(prompt, HypothesisBatch)

    out: list[Hypothesis] = []
    for draft in batch.items:
        if len(out) >= n:
            break
        if has_output_leak(draft.statement):
            continue
        grounded = grounded_evidence([(c.paper_id, c.quote) for c in draft.evidence], corpus)
        out.append(
            Hypothesis(
                subject=draft.subject,
                qualifier=draft.qualifier,
                object=draft.object,
                statement=draft.statement,
                rationale=draft.rationale,
                independent_var=draft.independent_var,
                dependent_var=draft.dependent_var,
                methodology=draft.methodology,
                evidence=[EvidenceRef(paper_id=pid, quote=quote) for pid, quote in grounded],
                status="draft",
                provenance=Provenance(model="(see settings)"),
            )
        )
    return out
