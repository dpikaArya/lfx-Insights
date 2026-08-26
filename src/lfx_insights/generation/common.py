"""Shared generation helpers: APA formatting, citation verification, grounding,
and output-hygiene checks (the guardrails that the old tool failed).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lfx_insights.standards.grounding import verify_quote_in

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, GeneratedSection, Paper, SectionBundle

# Curated single-word f-string placeholder names that templates substitute into
# output and that have been seen to leak verbatim. A bare {word} that is NOT in
# this set and has no underscore is treated as legitimate prose (e.g. set
# notation like "{x}"), not a template leak. Snake_case names ({top_theme}) are
# always flagged regardless of this set.
_CURATED_PLACEHOLDERS = frozenset(
    {
        "accuracy",
        "count",
        "example",
        "metric",
        "name",
        "result",
        "score",
        "theme",
        "title",
        "value",
        "year",
    }
)
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z][a-z0-9]*)\}")
_NAN = re.compile(r"\bnan\b", re.IGNORECASE)


def _doi_norm(doi: str | None) -> str:
    """Normalize a DOI for comparison: lowercase, strip any doi.org URL prefix."""
    return (
        (doi or "").lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    )


def _strip_doi_url(doi: str) -> str:
    """Remove a leading http(s)://doi.org/ prefix so it is not double-applied."""
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def _format_author(name: str) -> str:
    """Format one author as 'Surname, I. I.' (APA), robust to 'First Last',
    'Last, First', and initial-bearing inputs. Keeps surname particles.
    """
    name = name.strip()
    if not name:
        return ""
    if "," in name:
        surname, _, given = name.partition(",")
        surname = surname.strip()
        given_tokens = given.split()
    else:
        tokens = name.split()
        surname = tokens[-1] if tokens else name
        given_tokens = tokens[:-1]
        # Pull leading lowercase particles (van, de, der) back onto the surname.
        while given_tokens and given_tokens[-1].islower():
            surname = f"{given_tokens.pop()} {surname}"
    initials = " ".join(f"{t[0].upper()}." for t in given_tokens if t)
    return f"{surname}, {initials}".strip().rstrip(",") if initials else surname


def format_authors(names: list[str]) -> str:
    formatted = [_format_author(n) for n in names if n.strip()]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + ", & " + formatted[-1]


def _surname(name: str) -> str:
    """Extract the surname from an author name (mirrors :func:`_format_author`),
    keeping lowercase particles (van, de, der) attached to the surname."""
    name = name.strip()
    if not name:
        return ""
    if "," in name:
        return name.partition(",")[0].strip()
    tokens = name.split()
    if not tokens:
        return name
    surname = tokens[-1]
    given = tokens[:-1]
    while given and given[-1].islower():
        surname = f"{given.pop()} {surname}"
    return surname


def _short_title(title: str, words: int = 3) -> str:
    """First few words of a title, for an author-less APA in-text citation."""
    toks = title.split()
    short = " ".join(toks[:words])
    return f'"{short}"' if short else "Untitled"


def _year_token(year: int | None, suffix: str = "") -> str:
    """APA year token with an optional disambiguation suffix: ``2020`` -> ``2020a``;
    a missing/zero year -> ``n.d.`` (or ``n.d.-a`` when disambiguated)."""
    if year:
        return f"{year}{suffix}"
    return f"n.d.-{suffix}" if suffix else "n.d."


def format_apa_intext(paper: Paper, suffix: str = "") -> str:
    """Render an APA in-text citation body, e.g. ``Smith, 2020`` /
    ``Smith & Jones, 2020`` / ``Smith et al., 2020`` / ``"Deep learning for", 2020``.

    ``suffix`` is the year-disambiguation letter (``a``/``b``/...) applied when
    several cited papers share the same author+year (see
    :func:`disambiguation_suffixes`). Falls back to a short title when the paper
    has no authors, and ``n.d.`` when it has no year. The leading ``(`` and
    trailing ``)`` are NOT included, so callers can group several papers in one
    parenthetical (``(A, 2020; B, 2019)``).
    """
    year_token = _year_token(paper.year, suffix)
    surnames = [s for s in (_surname(a.name) for a in paper.authors) if s]
    if not surnames:
        return f"{_short_title(paper.title)}, {year_token}"
    if len(surnames) == 1:
        author = surnames[0]
    elif len(surnames) == 2:
        author = f"{surnames[0]} & {surnames[1]}"
    else:
        author = f"{surnames[0]} et al."
    return f"{author}, {year_token}"


def render_intext_group(papers: list[Paper], suffixes: dict[str, str] | None = None) -> str:
    """Render one parenthetical citing one or more papers: ``(A, 2020; B, 2019)``,
    applying each paper's disambiguation ``suffixes`` (id -> letter) when given."""
    sx = suffixes or {}
    inner = "; ".join(format_apa_intext(p, sx.get(p.id, "")) for p in papers)
    return f"({inner})"


def _letter(index: int) -> str:
    """0->'a', 1->'b', ..., 25->'z', 26->'aa' (spreadsheet-style), for APA year
    disambiguation of more than 26 same-author/year references."""
    out = ""
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def disambiguation_suffixes(papers: list[Paper]) -> dict[str, str]:
    """Assign APA year-disambiguation letters across a set of cited papers.

    Papers whose in-text citation would otherwise be identical (same rendered
    author + year, e.g. two ``Smith et al., 2020`` references) are disambiguated:
    they are ordered by title (APA 7th, Â§9.47) and assigned ``a``, ``b``, ``c`` ...
    A paper with a unique author+year gets an empty suffix. Returns ``{paper_id:
    suffix}`` for every (deduped) paper, so the SAME map can drive both the in-text
    citations and the reference list â€” guaranteeing they always agree.
    """
    unique: list[Paper] = []
    seen: set[str] = set()
    for p in papers:
        if p.id not in seen:
            seen.add(p.id)
            unique.append(p)
    groups: dict[str, list[Paper]] = {}
    for p in unique:
        groups.setdefault(format_apa_intext(p), []).append(p)
    suffixes: dict[str, str] = {}
    for group in groups.values():
        if len(group) == 1:
            suffixes[group[0].id] = ""
            continue
        for i, p in enumerate(sorted(group, key=lambda q: ((q.title or "").lower(), q.id))):
            suffixes[p.id] = _letter(i)
    return suffixes


# An inline citation marker the LLM emits in prose, e.g. ``[W1]`` or ``[W1; W3]``,
# and the private sentinel a grounded marker is normalized to before the global
# disambiguation pass renders it (NUL never occurs in model text).
_INTEXT_MARKER = re.compile(r"(\s*)\[([^\[\]]+)\]")
_CITE_SENTINEL = re.compile("\x00CITE:([^\x00]*)\x00")


def mark_intext_citations(
    text: str, grounded_ids: list[str], corpus: Corpus
) -> tuple[str, list[str]]:
    """First pass: normalize grounded ``[paper_id]`` markers to private sentinels.

    For each ``[...]`` marker: a bracket whose tokens do not resolve to any corpus
    paper is left untouched (prose, e.g. ``[sic]``); a citation marker is reduced
    to its grounded, resolved ids and replaced by a ``\\x00CITE:id;id\\x00``
    sentinel, while a marker with nothing grounded is removed with its leading
    space. Returns ``(text, placed_ids)`` with placed ids in first-appearance
    order (deduped). Rendering to APA happens later in
    :func:`render_intext_citations`, once cross-section disambiguation is known.
    """
    grounded = set(grounded_ids)
    placed: list[str] = []
    placed_seen: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        lead, inner = match.group(1), match.group(2)
        tokens = [t.strip() for t in re.split(r"[;,]", inner) if t.strip()]
        resolved = [_resolve_paper(t, corpus) for t in tokens]
        if not any(p is not None for p in resolved):
            return match.group(0)  # not a citation -> leave prose untouched
        kept: list[str] = []
        for p in resolved:
            if p is not None and p.id in grounded and p.id not in kept:
                kept.append(p.id)
        if not kept:
            return ""  # citation marker but nothing grounded -> drop it (and its space)
        for pid in kept:
            if pid not in placed_seen:
                placed_seen.add(pid)
                placed.append(pid)
        return f"{lead}\x00CITE:{';'.join(kept)}\x00"

    return _INTEXT_MARKER.sub(_replace, text), placed


def render_intext_citations(
    text: str, corpus: Corpus, suffixes: dict[str, str] | None = None
) -> str:
    """Second pass: render the ``\\x00CITE:...\\x00`` sentinels left by
    :func:`mark_intext_citations` into APA in-text parentheticals, applying the
    cross-document disambiguation ``suffixes`` (id -> letter)."""

    def _render(match: re.Match[str]) -> str:
        ids = [i for i in match.group(1).split(";") if i]
        papers = [p for p in (corpus.by_id(i) for i in ids) if p is not None]
        return render_intext_group(papers, suffixes) if papers else ""

    return _CITE_SENTINEL.sub(_render, text)


def apply_intext_citations(
    text: str,
    grounded_ids: list[str],
    corpus: Corpus,
    suffixes: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Resolve inline ``[paper_id]`` markers to APA in-text citations in one call.

    Equivalent to :func:`mark_intext_citations` followed by
    :func:`render_intext_citations`; used where no cross-section disambiguation is
    needed. :func:`draft_manuscript` instead calls the two passes directly so it
    can compute one disambiguation map across all sections. Returns
    ``(rendered_text, placed_ids)``; see :func:`mark_intext_citations` for the
    marker/prose/grounding rules.
    """
    marked, placed = mark_intext_citations(text, grounded_ids, corpus)
    return render_intext_citations(marked, corpus, suffixes), placed


def format_apa(paper: Paper, suffix: str = "") -> str:
    """Render a paper as an APA reference string, with an optional year-
    disambiguation suffix (``a``/``b``/...) matching its in-text citation."""
    authors = format_authors([a.name for a in paper.authors])
    year_token = _year_token(paper.year, suffix)
    parts = []
    if authors:
        parts.append(f"{authors} ({year_token}).")
    else:
        parts.append(f"({year_token}).")
    parts.append(f"{paper.title.rstrip('.')}.")
    if paper.source:
        parts.append(f"{paper.source}.")
    if paper.doi:
        parts.append(f"https://doi.org/{_strip_doi_url(paper.doi)}")
    return " ".join(parts).strip()


def verify_citation(ref: str, corpus: Corpus) -> bool:
    """True if ``ref`` (a paper id or DOI) resolves to a real paper in the corpus."""
    if not ref:
        return False
    if corpus.by_id(ref) is not None:
        return True
    ref_norm = _doi_norm(ref)
    if not ref_norm:
        return False
    return any(_doi_norm(p.doi) == ref_norm for p in corpus.papers)


def filter_verified_citations(refs: list[str], corpus: Corpus) -> list[str]:
    """Drop any citation that does not resolve to a corpus paper (anti-hallucination)."""
    seen: set[str] = set()
    out: list[str] = []
    for r in refs:
        if r not in seen and verify_citation(r, corpus):
            seen.add(r)
            out.append(r)
    return out


def ground_quote(quote: str, passages: list[str]) -> bool:
    """True if a quoted span is grounded in the supplied passages."""
    return verify_quote_in(quote, passages)


def _resolve_paper(ref: str, corpus: Corpus) -> Paper | None:
    """Resolve a citation ref (paper id or DOI, with or without doi.org prefix)."""
    paper = corpus.by_id(ref)
    if paper is not None:
        return paper
    ref_norm = _doi_norm(ref)
    if not ref_norm:
        return None
    return next((p for p in corpus.papers if _doi_norm(p.doi) == ref_norm), None)


def ground_cited(cited: list[tuple[str, str]], corpus: Corpus) -> list[str]:
    """Keep only citations whose supporting quote is verifiably present in the
    cited paper's text. A citation with no quote, an unresolvable paper, or a quote
    that does not ground (likely paraphrase/hallucination) is DROPPED.

    Stronger than :func:`filter_verified_citations` (id-membership only): this is the
    anti-hallucination grounding gate for generated prose.
    """
    out: list[str] = []
    seen: set[str] = set()
    for ref, quote in cited:
        if ref in seen:
            continue
        paper = _resolve_paper(ref, corpus)
        if paper is not None and quote and verify_quote_in(quote, [paper.text()]):
            seen.add(ref)
            out.append(paper.id)
    return out


def grounded_evidence(cited: list[tuple[str, str]], corpus: Corpus) -> list[tuple[str, str]]:
    """Like :func:`ground_cited` but returns ``(paper_id, grounded_quote)`` pairs,
    for building Evidence records that carry the verified quote.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ref, quote in cited:
        if ref in seen:
            continue
        paper = _resolve_paper(ref, corpus)
        if paper is not None and quote and verify_quote_in(quote, [paper.text()]):
            seen.add(ref)
            out.append((paper.id, quote))
    return out


def has_output_leak(text: str) -> bool:
    """Detect unformatted f-string placeholders ('{top_theme}') or 'NaN' leaks â€”
    the exact hygiene failures seen in the old tool's generated output.

    A placeholder is flagged only when it looks like a template name: a
    snake_case identifier (``{top_theme}``) or one of a small curated set of
    known leaked names (``{example}``). A bare ``{word}`` with no underscore is
    treated as legitimate prose (e.g. set-builder notation) and not flagged.
    """
    for match in _PLACEHOLDER.finditer(text):
        name = match.group(1)
        if "_" in name or name in _CURATED_PLACEHOLDERS:
            return True
    return bool(_NAN.search(text))


def build_section_bundle(
    title: str,
    sections: list[GeneratedSection],
    corpus: Corpus,
) -> SectionBundle:
    """Collect the papers cited across ``sections`` (deduped, first-citation order)
    into a serializable :class:`SectionBundle`. Uncited corpus papers are excluded;
    citation ids that do not resolve in ``corpus`` are skipped defensively."""
    from lfx_insights.models import SectionBundle

    seen: set[str] = set()
    references: list[Paper] = []
    for section in sections:
        for paper_id in section.citations:
            if paper_id in seen:
                continue
            seen.add(paper_id)
            paper = corpus.by_id(paper_id)
            if paper is not None:
                references.append(paper)
    return SectionBundle(title=title, sections=list(sections), references=references)


# ---------------------------------------------------------------------------
# Citation validation helpers
# ---------------------------------------------------------------------------

# Patterns for APA in-text citations that may appear in rendered text.
_APA_INTEXT = re.compile(r"\(([A-Z][^)]*\d{4}[a-z]?(?:;\s*[A-Z][^)]*\d{4}[a-z]?)*)\)")
_AUTHOR_YEAR = re.compile(r"([A-Z][a-z\u00C0-\u024F]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z\u00C0-\u024F]+))?)\s*,\s*(\d{4}[a-z]?)")


def extract_cited_paper_ids(text: str, corpus: Corpus) -> list[str]:
    """Extract paper IDs cited in rendered text by resolving author-year pairs.

    Scans the text for APA in-text citation patterns and resolves each
    author+year pair to a corpus paper. Returns resolved paper IDs in
    first-appearance order (deduped). Unresolvable pairs are silently
    skipped (they may be author names in prose, not citations).
    """
    seen: set[str] = set()
    out: list[str] = []
    for match in _APA_INTEXT.finditer(text):
        inner = match.group(1)
        for part in inner.split(";"):
            part = part.strip()
            ay = _AUTHOR_YEAR.match(part)
            if ay is None:
                continue
            author_part, year_str = ay.group(1), ay.group(2)
            year = int(year_str.rstrip("abcd"))
            suffix = year_str[len(str(year)):]
            surname = _surname(author_part.split("&")[0].split("et")[0].strip())
            for paper in corpus.papers:
                if paper.year != year:
                    continue
                paper_surnames = [_surname(a.name) for a in paper.authors]
                if not paper_surnames:
                    continue
                if len(author_part.split("&")) > 1 or "et al." in author_part:
                    if surname in paper_surnames and paper.id not in seen:
                        seen.add(paper.id)
                        out.append(paper.id)
                        break
                else:
                    if paper_surnames[0] == surname and paper.id not in seen:
                        seen.add(paper.id)
                        out.append(paper.id)
                        break
    return out


def validate_citations_in_text(text: str, corpus: Corpus) -> dict[str, object]:
    """Validate citations found in a block of rendered text.

    Returns a dict with:
      - ``cited_ids``: paper IDs resolved from in-text citations
      - ``unresolved``: author-year pairs that could not be resolved
      - ``all_exist``: True if every resolved citation exists in the corpus
      - ``metadata_matches``: list of (paper_id, field, expected, actual) mismatches
    """
    cited_ids = extract_cited_paper_ids(text, corpus)
    metadata_issues: list[tuple[str, str, str, str]] = []
    for pid in cited_ids:
        paper = corpus.by_id(pid)
        if paper is None:
            metadata_issues.append((pid, "existence", "in_corpus", "missing"))
    return {
        "cited_ids": cited_ids,
        "all_exist": all(corpus.by_id(pid) is not None for pid in cited_ids),
        "metadata_issues": metadata_issues,
    }


def build_cited_reference_list(
    sections: list[GeneratedSection], corpus: Corpus
) -> list[Paper]:
    """Build a deduped, citation-order reference list from rendered sections.

    Papers are deduplicated by DOI (normalized) when available, otherwise by
    paper ID. Only papers that exist in the corpus are included.
    """
    seen_ids: set[str] = set()
    seen_dois: set[str] = set()
    out: list[Paper] = []
    for section in sections:
        for pid in section.citations:
            paper = corpus.by_id(pid)
            if paper is None:
                continue
            if pid in seen_ids:
                continue
            doi_key = _doi_norm(paper.doi)
            if doi_key and doi_key in seen_dois:
                continue
            seen_ids.add(pid)
            if doi_key:
                seen_dois.add(doi_key)
            out.append(paper)
    return out


def format_reference_list(papers: list[Paper], corpus: Corpus | None = None) -> str:
    """Format a list of papers as an APA reference list (sorted, deduped).

    Applies disambiguation suffixes when the same author+year appears multiple
    times. Returns a single string with one reference per line.
    """
    suffixes = disambiguation_suffixes(papers)
    entries = sorted(
        format_apa(p, suffixes.get(p.id, "")) for p in papers
    )
    return "\n".join(entries)


def validate_manuscript_citations(
    sections: list[GeneratedSection], corpus: Corpus
) -> dict[str, object]:
    """Validate all citations across manuscript sections.

    Returns a dict with:
      - ``total_cited``: unique paper IDs cited across all sections
      - ``all_exist``: True if every cited paper exists in the corpus
      - ``reference_count``: number of papers in the deduped reference list
      - ``issues``: list of validation issues found
    """
    all_ids: set[str] = set()
    issues: list[str] = []
    for section in sections:
        for pid in section.citations:
            if pid in all_ids:
                continue
            all_ids.add(pid)
            paper = corpus.by_id(pid)
            if paper is None:
                issues.append(f"citation '{pid}' in section '{section.name}' not found in corpus")
    ref_list = build_cited_reference_list(sections, corpus)
    return {
        "total_cited": len(all_ids),
        "all_exist": all(corpus.by_id(pid) is not None for pid in all_ids),
        "reference_count": len(ref_list),
        "issues": issues,
    }


def build_evidence_chain(
    sections: list[GeneratedSection], corpus: Corpus
) -> list[dict[str, object]]:
    """Build a Claim→Evidence→Reference chain from rendered sections.

    For each section, extracts the citations and maps them to their
    corresponding corpus papers. Returns a list of chains, one per section,
    each containing the section name, text, and a list of evidence entries
    with paper_id, title, doi, and year.
    """
    chains: list[dict[str, object]] = []
    for section in sections:
        entries: list[dict[str, object]] = []
        for pid in section.citations:
            paper = corpus.by_id(pid)
            if paper is None:
                continue
            entries.append({
                "paper_id": paper.id,
                "title": paper.title,
                "doi": paper.doi,
                "year": paper.year,
            })
        chains.append({
            "section": section.name,
            "text_preview": section.text[:200] + ("..." if len(section.text) > 200 else ""),
            "citations": entries,
        })
    return chains
