"""LLM-backed, corpus-grounded manuscript section drafting.

Drafts manuscript sections (introduction, related work, methods, discussion) from
a Perspicacité-derived corpus. For every citation the LLM must emit a short verbatim
quote drawn from the cited paper's abstract; citations are then passed through the
quote-grounding gate :func:`ground_cited` so any reference whose quote is not verbatim
present in the cited paper's text is dropped (anti-hallucination). Any section whose
text leaks an unformatted placeholder or ``NaN`` is skipped (the exact hygiene
failures of the old tool).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from consilium.generation.common import (
    disambiguation_suffixes,
    grounded_evidence,
    has_output_leak,
    mark_intext_citations,
    render_intext_citations,
)
from consilium.models import GeneratedSection, Provenance

if TYPE_CHECKING:
    from consilium.llm.client import LLMClient
    from consilium.models import Corpus

DEFAULT_SECTIONS: tuple[str, ...] = (
    "introduction",
    "related_work",
    "methods",
    "discussion",
)


class CitedRef(BaseModel):
    """A single citation: a corpus paper id plus a verbatim supporting quote."""

    paper_id: str
    quote: str = ""


class SectionDraft(BaseModel):
    """Structured LLM output for a single drafted section."""

    text: str
    cited: list[CitedRef] = Field(default_factory=list)


def _corpus_listing(corpus: Corpus) -> str:
    """Render the corpus as ``<id>: <title>`` lines for the prompt."""
    return "\n".join(f"{p.id}: {p.title}" for p in corpus.papers)


def _build_prompt(section: str, corpus: Corpus) -> str:
    """Build a corpus-grounded prompt for one section, constraining citations."""
    listing = _corpus_listing(corpus)
    return (
        f"You are drafting the '{section}' section of a scientific manuscript, "
        "grounded ONLY in the corpus below.\n\n"
        "Corpus papers (cite ONLY these ids, never invent references):\n"
        f"{listing}\n\n"
        f"Write the '{section}' section as flowing prose. Do not leave any "
        "unfilled placeholders. Place an inline citation marker in square "
        "brackets at the exact point you rely on a source, using the paper id, "
        "e.g. '... improves accuracy [W1]' or '... as shown previously [W1; W3]'. "
        "Cite a paper in-text ONLY with these ids; do not write author names or "
        "years yourself (they are formatted for you). In 'cited', return one entry "
        "per distinct paper you cited in-text. Each entry MUST use an exact paper "
        "id from the list above ('paper_id') AND a short 'quote' copied VERBATIM "
        "(word for word) from that paper's abstract that supports the citation. Do "
        "not paraphrase the quote and do not cite a paper for which you cannot "
        "supply a verbatim supporting quote."
    )


def draft_manuscript(
    corpus: Corpus,
    llm: LLMClient,
    *,
    sections: list[str] | None = None,
) -> list[GeneratedSection]:
    """Draft manuscript sections, grounded in ``corpus`` with verified citations.

    For each requested section the LLM produces a :class:`SectionDraft` containing
    prose with inline ``[paper_id]`` citation markers plus a ``cited`` list giving
    each citation a verbatim supporting quote. Sections whose text triggers
    :func:`has_output_leak` are skipped. Citations pass through the quote-grounding
    gate (:func:`grounded_evidence`); ungrounded markers are stripped from the prose
    and ``section.citations`` is reconciled to exactly the grounded paper ids that
    appear in-text, so the reference list maps 1:1 to the in-text citations (no
    orphan references, no fabricated citations).

    Citations are rendered in two passes so author-year collisions can be
    disambiguated at the document level: pass 1 grounds each section and normalizes
    its markers to sentinels; once every section is processed, a single APA
    disambiguation map (``a``/``b``/...) is computed across all cited papers and
    pass 2 renders the in-text citations with it. The same map drives the reference
    list (see :func:`consilium.reporting.docx_export.render_docx`), so two
    ``Smith et al., 2020`` papers read as ``2020a``/``2020b`` in both places.
    """
    names = sections if sections is not None else list(DEFAULT_SECTIONS)

    # Pass 1: ground + normalize markers to sentinels; collect the global cited set.
    marked: list[tuple[str, str]] = []  # (section_name, sentinel_text)
    placed_per_section: list[list[str]] = []
    global_placed: list[str] = []
    for section in names:
        draft = llm.complete_structured(_build_prompt(section, corpus), SectionDraft)
        if has_output_leak(draft.text):
            # Reject leaked output rather than emitting an unhygienic section.
            continue
        grounded_ids = [
            pid
            for pid, _ in grounded_evidence([(c.paper_id, c.quote) for c in draft.cited], corpus)
        ]
        text, placed = mark_intext_citations(draft.text, grounded_ids, corpus)
        marked.append((section, text))
        placed_per_section.append(placed)
        for pid in placed:
            if pid not in global_placed:
                global_placed.append(pid)

    # One disambiguation map over every cited paper, shared by in-text + references.
    cited_papers = [p for p in (corpus.by_id(pid) for pid in global_placed) if p is not None]
    suffixes = disambiguation_suffixes(cited_papers)

    # Pass 2: render the sentinels with the document-level suffix map.
    out: list[GeneratedSection] = []
    for (section, text), placed in zip(marked, placed_per_section, strict=True):
        out.append(
            GeneratedSection(
                name=section,
                text=render_intext_citations(text, corpus, suffixes),
                citations=placed,
                provenance=Provenance(model="(see settings)"),
            )
        )
    return out
