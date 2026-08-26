"""Export lfx Insights corpus/hypotheses to indicium-conformant dicts.

Shapes match the real indicium LinkML model (verified by round-trip instantiation
in tests): ``Source`` (FaBiO Expression), ``Claim`` (Bucur SuperPattern; requires
id/context/subject/relation/object, status via ``claim_status``), and ``Evidence``
(separate top-level records linked by ``for_claim``/``of_source``; ``eco_code`` is
an ECO *label* enum, not a CURIE). A hypothesis becomes a draft Claim plus one
Evidence line per grounding reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, Hypothesis

_DEFAULT_CONTEXT = "lfx-insights"


def indicium_available() -> bool:
    try:
        import indicium  # noqa: F401

        return True
    except ImportError:
        return False


def sources_to_indicium(corpus: Corpus) -> list[dict[str, Any]]:
    """Map each paper to an indicium ``Source`` dict (slot: identifier required)."""
    sources: list[dict[str, Any]] = []
    for paper in corpus.papers:
        sources.append(
            {
                "identifier": paper.doi or paper.id,
                "doi": paper.doi,
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
            }
        )
    return sources


def hypothesis_to_claim(
    h: Hypothesis, claim_id: str = "claim-0", context: str = _DEFAULT_CONTEXT
) -> dict[str, Any]:
    """Map a Hypothesis to an indicium ``Claim`` dict (Bucur 5-slot, draft).

    Evidence is NOT embedded in a Claim in indicium; see :func:`hypothesis_evidence`.
    ``relation`` is the required free-text predicate; ``qualifier`` is the controlled
    Bucur SuperPattern term.
    """
    return {
        "id": claim_id,
        "context": context,
        "subject": h.subject,
        "relation": h.qualifier,
        "qualifier": h.qualifier,
        "object": h.object,
        "claim_type": "explicit",
        "claim_status": h.status if h.status in {"draft", "verified", "published"} else "draft",
    }


def hypothesis_evidence(
    h: Hypothesis,
    claim_id: str = "claim-0",
    source_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map a Hypothesis's grounding refs to indicium ``Evidence`` records.

    ``eco_code`` uses the ECO-label enum: a quoted ref -> ``textual_quotation``
    (evidence_type ``citation``); an unquoted ref -> ``inference_from_background_knowledge``
    (evidence_type ``inference``). ``of_source`` is an INLINED Source object (the
    real indicium model embeds the Source, not a string id).
    """
    lookup = source_lookup or {}
    out: list[dict[str, Any]] = []
    for ref in h.evidence:
        quoted = bool(ref.quote)
        of_source = lookup.get(ref.paper_id) or {"identifier": ref.paper_id}
        out.append(
            {
                "evidence_type": "citation" if quoted else "inference",
                "eco_code": "textual_quotation"
                if quoted
                else "inference_from_background_knowledge",
                "of_source": of_source,
                "for_claim": claim_id,
                "quote": ref.quote,
                "quote_location": ref.location,
            }
        )
    return out


def claims_to_document(
    hypotheses: list[Hypothesis], corpus: Corpus, context: str = _DEFAULT_CONTEXT
) -> dict[str, Any]:
    """Bundle draft Claims + Evidence + Sources into an IndiciumDocument dict.

    Conforms to the real indicium model: ``claims`` is a dict keyed by claim id;
    ``evidences``/``sources`` are lists; each Evidence inlines its ``of_source``.
    """
    sources = sources_to_indicium(corpus)
    lookup: dict[str, dict[str, Any]] = {}
    for src, paper in zip(sources, corpus.papers, strict=True):
        for key in (paper.id, paper.doi, src["identifier"]):
            if key:
                lookup[key] = src
    claims: dict[str, dict[str, Any]] = {}
    evidences: list[dict[str, Any]] = []
    for i, h in enumerate(hypotheses):
        claim_id = f"claim-{i}"
        claims[claim_id] = hypothesis_to_claim(h, claim_id=claim_id, context=context)
        evidences.extend(hypothesis_evidence(h, claim_id=claim_id, source_lookup=lookup))
    return {"claims": claims, "evidences": evidences, "sources": sources}
