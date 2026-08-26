"""LLM-backed, corpus-grounded research-question generation.

Generates open research questions from a PerspicacitÃ©-built corpus. The LLM is
prompted with the corpus papers listed as ``<id>: <title>`` and asked to propose
questions with novelty/feasibility/impact components. Every question score is an
honest :class:`~lfx_insights.models.Score` built with ``make_score`` over the
LLM-provided components â€” never an RNG or hash. Output with formatting leaks
(``{placeholder}`` / ``NaN``) is rejected.

Citations are QUOTE-GROUNDED: each citation carries both a corpus paper id and a
short verbatim quote, and survives only if that quote is verbatim-present in the
cited paper's text (``ground_cited``). No-quote, unresolvable, or ungrounded
citations are dropped (anti-hallucination), not merely id-verified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lfx_insights.generation.common import ground_cited, has_output_leak
from lfx_insights.models import EvidenceRef, ResearchQuestion, ScoreComponent
from lfx_insights.scoring.common import clamp01, make_score

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus

_SYSTEM = (
    "You are a research strategist. Given a corpus of papers, propose forward-looking, "
    "answerable research questions that build on (but are not answered by) this corpus. "
    "For each question provide a short rationale and three 0..1 scores: novelty "
    "(how unexplored), feasibility (how tractable to study), and impact (how consequential). "
    "Base the direction and scores on the corpus; do not invent papers."
)


class CitedRef(BaseModel):
    """A citation that carries BOTH a corpus paper id and a short verbatim quote
    (copied exactly from that paper's abstract) supporting the citation.

    The quote is the anti-hallucination anchor: a citation only survives if its
    quote is verbatim-present in the cited paper's text (see :func:`ground_cited`).
    """

    paper_id: str
    quote: str = ""


class QuestionDraft(BaseModel):
    """One LLM-proposed research question with its scoring components."""

    question: str
    rationale: str = ""
    novelty: float = 0.5
    feasibility: float = 0.5
    impact: float = 0.5
    citations: list[CitedRef] = Field(default_factory=list)


class QuestionBatch(BaseModel):
    """Batch wrapper so N questions are produced in a single structured call."""

    items: list[QuestionDraft] = Field(default_factory=list)


def _build_prompt(corpus: Corpus, n: int) -> str:
    lines = [f"{p.id}: {p.title}" for p in corpus.papers]
    catalogue = "\n".join(lines) if lines else "(empty corpus)"
    return (
        f"{_SYSTEM}\n\n"
        f"Corpus papers (cite only these ids):\n{catalogue}\n\n"
        f"Propose up to {n} research questions as a JSON object with an 'items' list. "
        "For each item include a 'citations' list of objects that motivate the "
        "question. Each citation object has a 'paper_id' (cite ONLY ids from the list "
        "above) and a 'quote' that is a SHORT VERBATIM span copied EXACTLY from that "
        "paper's abstract supporting the citation. Do not paraphrase the quote and do "
        "not invent text: a citation whose quote is not found verbatim in the cited "
        "paper will be dropped."
    )


def generate_questions(corpus: Corpus, llm: LLMClient, *, n: int = 10) -> list[ResearchQuestion]:
    """Generate up to ``n`` scored research questions grounded in ``corpus``.

    The direction and the novelty/feasibility/impact components come from the LLM;
    each score is built with :func:`make_score`. Questions whose text triggers
    :func:`has_output_leak` are dropped. Results are sorted by score value
    (descending) and capped at ``n``.
    """
    prompt = _build_prompt(corpus, n)
    batch = llm.complete_structured(prompt, QuestionBatch)

    questions: list[ResearchQuestion] = []
    for draft in batch.items:
        if has_output_leak(draft.question):
            continue
        components = [
            ScoreComponent(name="novelty", value=clamp01(draft.novelty), weight=0.4),
            ScoreComponent(name="feasibility", value=clamp01(draft.feasibility), weight=0.3),
            ScoreComponent(name="impact", value=clamp01(draft.impact), weight=0.3),
        ]
        # Keep only citations whose verbatim quote grounds in the cited paper's
        # text; drop no-quote, unresolvable, or ungrounded cites (anti-hallucination).
        grounded = ground_cited([(c.paper_id, c.quote) for c in draft.citations], corpus)
        evidence = [EvidenceRef(paper_id=pid) for pid in grounded]
        questions.append(
            ResearchQuestion(
                question=draft.question,
                rationale=draft.rationale,
                score=make_score(components),
                evidence=evidence,
            )
        )

    questions.sort(key=lambda q: q.score.value if q.score is not None else 0.0, reverse=True)
    return questions[:n]
