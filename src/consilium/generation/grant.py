"""LLM-backed, corpus-grounded grant drafting.

Drafts grant sections (specific aims, significance, approach, project summary)
strictly from the supplied corpus. Each citation the LLM emits must carry a short
verbatim quote copied from the cited paper's abstract; citations are passed
through :func:`ground_cited`, which keeps only those whose quote is verifiably
present in the cited paper's text (hallucinated or paraphrased refs are dropped).
Any section whose body trips :func:`has_output_leak` is rejected — these are
exactly the failures of the old tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from consilium.generation.common import (
    format_apa,
    ground_cited,
    has_output_leak,
)
from consilium.models import GeneratedSection, Provenance

if TYPE_CHECKING:
    from consilium.llm.client import LLMClient
    from consilium.models import Corpus

DEFAULT_SECTIONS: tuple[str, ...] = (
    "specific_aims",
    "significance",
    "approach",
    "project_summary",
)


class CitedRef(BaseModel):
    """One citation: a corpus paper id plus a verbatim supporting quote.

    The quote must be copied exactly from the cited paper's abstract; it is the
    evidence that grounds the citation (see :func:`ground_cited`).
    """

    paper_id: str = Field(description="A corpus paper id, taken from the listed ids.")
    quote: str = Field(
        default="",
        description="A short verbatim span copied exactly from that paper's abstract.",
    )


class SectionDraft(BaseModel):
    """One LLM-drafted grant section with the corpus refs it cites."""

    text: str = Field(description="The drafted prose for this section.")
    cited: list[CitedRef] = Field(
        default_factory=list,
        description="Citations, each a paper id plus a verbatim supporting quote.",
    )


def _corpus_listing(corpus: Corpus) -> str:
    """Render the corpus as '<id>: <title>' lines for the prompt."""
    return "\n".join(f"{p.id}: {p.title}" for p in corpus.papers)


def _references_block(corpus: Corpus) -> str:
    """Full APA references for the prompt, so the model has bibliographic context."""
    return "\n".join(format_apa(p) for p in corpus.papers)


def _build_prompt(section: str, corpus: Corpus) -> str:
    listing = _corpus_listing(corpus)
    refs = _references_block(corpus)
    return (
        f"You are drafting the '{section}' section of a research grant proposal, "
        "grounded ONLY in the corpus below.\n\n"
        "Corpus papers (cite ONLY these ids; do not invent or cite any other ids):\n"
        f"{listing}\n\n"
        "References (APA):\n"
        f"{refs}\n\n"
        f"Write the '{section}' section as clear, specific prose. Do NOT leave any "
        "template placeholders (e.g. '{example}') and do NOT emit 'NaN'.\n"
        "For EACH citation, return an entry in 'cited' with the exact corpus paper "
        "id AND a short verbatim 'quote' copied EXACTLY (word for word) from that "
        "paper's abstract to support the citation. Cite ONLY the listed paper ids; "
        "do not paraphrase the quote — copy it verbatim or it will be dropped."
    )


def draft_grant(
    corpus: Corpus,
    llm: LLMClient,
    *,
    sections: list[str] | None = None,
) -> list[GeneratedSection]:
    """Draft grant sections grounded in ``corpus``.

    For each requested section the LLM is prompted with the corpus listing and
    instructed to cite only in-corpus ids, each paired with a verbatim supporting
    quote. The returned text is rejected if it contains output leaks, and
    citations are passed through :func:`ground_cited` so only those whose quote is
    verifiably present in the cited paper survive (anti-hallucination).
    """
    wanted = list(sections) if sections is not None else list(DEFAULT_SECTIONS)
    out: list[GeneratedSection] = []
    for section in wanted:
        prompt = _build_prompt(section, corpus)
        draft = llm.complete_structured(prompt, SectionDraft)
        if has_output_leak(draft.text):
            continue
        citations = ground_cited([(c.paper_id, c.quote) for c in draft.cited], corpus)
        out.append(
            GeneratedSection(
                name=section,
                text=draft.text,
                citations=citations,
                provenance=Provenance(model="(see settings)"),
            )
        )
    return out
