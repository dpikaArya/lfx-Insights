"""Corpus-grounded long-form answering with inline ``[n]`` citations.

This is the lfx Insights *synthesis* half of the eval pipeline: given a question and a
retrieved corpus, draft a flowing prose answer whose claims are grounded ONLY in the
supplied sources and carry inline numbered citations ``[n]``.

It reuses the manuscript grounding pattern â€” a numbered corpus listing fed to a
structured LLM call, followed by a verbatim-quote grounding gate. Every inline
citation the model emits must come with a quote copied word-for-word from the source
it points to; citations whose quote does not ground (or whose marker is out of range)
are dropped, and their ``[n]`` markers are stripped from the prose. This is
lfx Insights anti-hallucination discipline applied to the eval answerer: an answer
never keeps a citation it cannot quote, and never shows a marker it did not keep.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lfx_insights.eval.models import Citation, GeneratedAnswer, RetrievedDoc
from lfx_insights.models import Provenance
from lfx_insights.standards.grounding import verify_quote_in

if TYPE_CHECKING:
    from lfx_insights.eval.entailment import Entailer
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus

_MARKER = re.compile(r"\[(\d+)\]")
_DOUBLE_SPACE = re.compile(r" {2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Cap per-source text in the prompt so it stays bounded on full-text corpora while
# still giving the model the passage to quote from (titles alone are unquotable).
_SOURCE_CHARS = 1500


class CitedMarker(BaseModel):
    """One inline citation the model emitted: which marker, which source, why."""

    marker: int  # the n the model used in [n], 0-indexed into the source listing
    paper_id: str
    quote: str = ""  # verbatim span from that source supporting the citation


class AnswerDraft(BaseModel):
    """Structured LLM output for a grounded answer with inline ``[n]`` citations."""

    text: str
    cited: list[CitedMarker] = Field(default_factory=list)


def _source_listing(docs: list[RetrievedDoc]) -> str:
    """Render the docs as 0-indexed ``[i] <title>`` blocks followed by their text.

    The source text (truncated to ``_SOURCE_CHARS``) is included so the model can
    supply a verbatim supporting quote â€” a title-only listing would force every
    citation to be dropped by the grounding gate on a real LLM run.
    """
    blocks = []
    for i, d in enumerate(docs):
        body = d.text[:_SOURCE_CHARS].strip()
        blocks.append(f"[{i}] {d.title}\n{body}" if body else f"[{i}] {d.title}")
    return "\n\n".join(blocks)


def _build_prompt(question: str, docs: list[RetrievedDoc]) -> str:
    """Build a grounded answering prompt that constrains inline citations to sources."""
    listing = _source_listing(docs)
    return (
        "Answer the question below in flowing prose, grounded ONLY in the numbered "
        "sources provided. Do not use outside knowledge and do not invent sources.\n\n"
        f"Question:\n{question}\n\n"
        "Sources (cite ONLY these numbers):\n"
        f"{listing}\n\n"
        "Cite inline as [i], using the source number i shown above. In 'cited', "
        "return one entry per inline citation you used. Each entry MUST give the "
        "'marker' (the integer i you used inline), the 'paper_id' of that source, and "
        "a 'quote' copied VERBATIM (word for word) from that source supporting the "
        "citation. Do not paraphrase the quote, and do not cite a source for which you "
        "cannot supply a verbatim supporting quote."
    )


def _strip_ungrounded_markers(text: str, kept: set[int]) -> str:
    """Remove every ``[k]`` whose ``k`` is not kept, then collapse double spaces."""

    def _drop(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in kept else ""

    stripped = _MARKER.sub(_drop, text)
    return _DOUBLE_SPACE.sub(" ", stripped).strip()


# Sentences shorter than this (stripped) are skipped by the citation scorer, so there
# is no value in attributing a citation to them.
_MIN_SCORABLE_CHARS = 50
_TERMINAL = re.compile(r"[.!?]+\s*$")


def _selfground(text: str, docs: list[RetrievedDoc], entailer: Entailer) -> tuple[str, list[int]]:
    """Self-ground every sentence's citations against the sources (two directions).

    1. **Gate** (precision): drop a ``[k]`` whose ``docs[k]`` does not entail its sentence â€”
       no misattributed citation survives (CRAG/Self-RAG-style).
    2. **Attribute** (recall, "cite-as-you-write"): a scorable sentence left with no
       citation is given ``[d]`` for the first source that entails it â€” so supported claims
       are not left uncited (ALCE-style fine-grained attribution).

    Returns the revised text and the sorted kept markers.
    """
    kept: set[int] = set()
    out_parts: list[str] = []
    for sentence in _SENT_SPLIT.split(text):
        target = _MARKER.sub("", sentence).strip()

        def _keep(match: re.Match[str], target: str = target) -> str:
            k = int(match.group(1))
            if 0 <= k < len(docs) and entailer.entails(docs[k].text, target):
                kept.add(k)
                return match.group(0)
            return ""

        gated = _MARKER.sub(_keep, sentence)
        if not _MARKER.search(gated) and len(target) >= _MIN_SCORABLE_CHARS:
            for d in range(len(docs)):
                if entailer.entails(docs[d].text, target):
                    gated = _insert_marker(gated, d)
                    kept.add(d)
                    break
        out_parts.append(gated)
    revised = _DOUBLE_SPACE.sub(" ", " ".join(p.strip() for p in out_parts)).strip()
    return revised, sorted(kept)


def _insert_marker(sentence: str, marker: int) -> str:
    """Insert ``[marker]`` before the sentence's terminal punctuation (or at the end)."""
    m = _TERMINAL.search(sentence)
    if m:
        return f"{sentence[: m.start()].rstrip()} [{marker}]{sentence[m.start() :]}"
    return f"{sentence.rstrip()} [{marker}]"


def answer_question(
    question: str,
    corpus: Corpus,
    llm: LLMClient,
    *,
    max_docs: int = 10,
    entailer: Entailer | None = None,
) -> GeneratedAnswer:
    """Answer ``question`` grounded in ``corpus``, with verified inline ``[n]`` citations.

    The first ``max_docs`` corpus papers become 0-indexed sources. The LLM drafts a
    prose answer with inline ``[i]`` markers and, for each, a verbatim supporting quote.

    Grounding (one of two): by default a citation is kept iff its marker is in range and
    its quote is verbatim-present in the source (anti-fabrication). When an ``entailer`` is
    supplied, the answer is **self-grounded** instead (:func:`_selfground`): every ``[k]``
    is kept only if ``docs[k]`` entails its sentence (precision), and any scorable sentence
    left uncited is given a citation to the first source that entails it (recall,
    "cite-as-you-write"). Either way, an empty corpus yields a closed-book answer with no
    docs and all markers stripped.
    """
    docs = [
        RetrievedDoc(doc_id=f"D{i}", title=p.title, text=p.text(), paper_id=p.id)
        for i, p in enumerate(corpus.papers[:max_docs])
    ]
    draft = llm.complete_structured(_build_prompt(question, docs), AnswerDraft)

    if entailer is not None:
        stripped_text, kept_markers = _selfground(draft.text, docs, entailer)
    else:
        kept: set[int] = set()
        for cm in draft.cited:
            if 0 <= cm.marker < len(docs) and verify_quote_in(cm.quote, [docs[cm.marker].text]):
                kept.add(cm.marker)
        kept_markers = sorted(kept)
        stripped_text = _strip_ungrounded_markers(draft.text, kept)

    citations = [Citation(marker=k, doc_id=docs[k].doc_id) for k in kept_markers]

    return GeneratedAnswer(
        text=stripped_text,
        docs=docs,
        citations=citations,
        provenance=Provenance(generated_by="lfx-insights", model="(eval)"),
    )
