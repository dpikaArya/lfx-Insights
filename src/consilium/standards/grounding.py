"""Grounding gate (anti-hallucination).

Prefers indicium's ``verify_quote`` kernel; falls back to a normalized substring
match when indicium is not installed. Also builds W3C Web Annotation
``TextQuoteSelector`` anchors used by the indicium/ASTRA exporters.
"""

from __future__ import annotations

import re

from consilium.errors import GroundingError

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
