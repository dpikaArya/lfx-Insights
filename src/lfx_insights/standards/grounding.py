"""Grounding gate (anti-hallucination).

Prefers indicium's ``verify_quote`` kernel; falls back to a normalized substring
match when indicium is not installed. Also builds W3C Web Annotation
``TextQuoteSelector`` anchors used by the indicium/ASTRA exporters.
"""

from __future__ import annotations

import re

from lfx_insights.errors import GroundingError

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    # Case-SENSITIVE on purpose: mirror indicium.verify_quote's containment so
    # grounding strictness does not silently depend on whether indicium is
    # installed (the fallback path must match the kernel).
    return _WS.sub(" ", text).strip()


def text_quote_selector(quote: str, source_text: str, pad: int = 32) -> dict[str, str]:
    """Build an OA TextQuoteSelector (exact/prefix/suffix) for ``quote``."""
    idx = source_text.find(quote)
    if idx < 0:
        return {"exact": quote, "prefix": "", "suffix": ""}
    start = max(0, idx - pad)
    end = min(len(source_text), idx + len(quote) + pad)
    return {
        "exact": quote,
        "prefix": source_text[start:idx],
        "suffix": source_text[idx + len(quote) : end],
    }


def verify_quote_in(quote: str, candidates: list[str], near_threshold: float = 0.9) -> bool:
    """True if ``quote`` is grounded in any candidate passage."""
    if not quote or not candidates:
        return False
    try:
        from indicium.verify import verify_quote

        result = verify_quote(quote, candidates, near_threshold=near_threshold)
        return result.status in {"verified", "repaired"}
    except ImportError:
        nq = _normalize(quote)
        return any(nq and nq in _normalize(c) for c in candidates)


def require_grounded(quote: str, candidates: list[str], near_threshold: float = 0.9) -> None:
    """Raise :class:`GroundingError` if ``quote`` is not grounded."""
    if not verify_quote_in(quote, candidates, near_threshold):
        raise GroundingError(
            f"Quote not grounded in corpus (likely paraphrase/hallucination): {quote!r}"
        )


# ---------------------------------------------------------------------------
# Citation verification (extends the grounding gate)
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\(([^)]*\d{4}[^)]*)\)|\[(\d+(?:\s*[-,]\s*\d+)*)\]")


def verify_citation_in_text(citation_text: str, corpus_papers: list[dict[str, str]]) -> bool:
    """Verify a citation string resolves to a real paper in the corpus.

    ``corpus_papers`` is a list of dicts with at least "title" and optionally "doi".
    Matches by title substring or DOI containment.
    """
    if not citation_text or not corpus_papers:
        return False
    lower = citation_text.lower()
    for paper in corpus_papers:
        title = (paper.get("title") or "").lower()
        doi = (paper.get("doi") or "").lower()
        if title and title in lower:
            return True
        if doi and doi in lower:
            return True
    return False


def verify_paragraph_grounding(
    paragraph: str, candidates: list[str]
) -> list[dict[str, object]]:
    """Extract and verify all citations in a paragraph.

    Returns a list of {citation, verified} dicts.
    """
    results: list[dict[str, object]] = []
    matches = _CITE_RE.findall(paragraph)

    for groups in matches:
        cite_str = groups[0] or groups[1]
        if not cite_str:
            continue
        verified = verify_quote_in(cite_str, candidates) if candidates else False
        results.append({"citation": cite_str.strip(), "verified": verified})

    return results


def verify_manuscript_grounding(
    sections: list[dict[str, str]], candidates: list[str]
) -> dict[str, object]:
    """Verify all citations in a manuscript draft.

    ``sections`` is a list of dicts with "name" and "text" keys.
    Returns {total, verified, unverified, citations}.
    """
    all_citations: list[dict[str, object]] = []
    for section in sections:
        text = section.get("text", "")
        section_cites = verify_paragraph_grounding(text, candidates)
        for cite in section_cites:
            cite["section"] = section.get("name", "unknown")
        all_citations.extend(section_cites)

    verified = sum(1 for c in all_citations if c.get("verified"))
    return {
        "total": len(all_citations),
        "verified": verified,
        "unverified": len(all_citations) - verified,
        "citations": all_citations,
    }
