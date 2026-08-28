"""Knowledge-base access for the Word Office Add-in.

Loads the existing lfx Insights knowledge base (``knowledge_base.json``),
exposes a lightweight offline retrieval over it, and reuses the project's own
APA citation logic (``lfx_insights.generation.common``) so references are
always derived from *verified* records — never invented by the model.

This module deliberately avoids any network/LLM calls: it is pure local data
so the add-in's evidence and citation features work even when Ollama is busy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lfx_insights.generation.common import (
    _surname,
    apply_intext_citations,
    format_reference_list,
)
from lfx_insights.models import Author, Corpus, Paper

# A parenthetical that contains a 4-digit year, e.g. "(Smith, 2021)".
_APPARENT_PAREN = re.compile(r"\(([^()]*\d{4}[a-z]?)\)")
# A single "Surname, 2021" or "First Last, 2021" citation token.
_AUTH_YEAR = re.compile(
    r"^([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s*,\s*(\d{4}[a-z]?)$"
)

# Repository root (src/lfx_insights/api.py -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "with", "on", "at",
    "by", "from", "as", "is", "are", "be", "this", "that", "these", "those",
    "it", "its", "we", "our", "their", "they", "which", "who", "how", "what",
    "why", "using", "used", "into", "via", "between", "among", "can", "may",
    "should", "could", "would", "not", "no", "yes", "but", "if", "than", "then",
    "such", "more", "most", "less", "few", "several", "many", "much", "also",
    "however", "therefore", "thus", "because", "due", "while", "although",
}


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


def _norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    return doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def _clean_str(value: Any) -> str | None:
    """Coerce a KB field to a clean string, mapping pandas ``NaN``/empty to None."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in ("nan", "none", "null", ""):
        return None
    return text


def _clean_int(value: Any) -> int | None:
    """Coerce a KB year field to an int, mapping ``NaN``/junk to None."""
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_knowledge_base_path() -> Path | None:
    """Locate the knowledge base JSON, preferring the repo root then outputs."""
    candidates = [
        _REPO_ROOT / "knowledge_base.json",
        _REPO_ROOT / "outputs" / "knowledge_base" / "knowledge_base.json",
        _REPO_ROOT / "outputs" / "knowledge_base" / "living_knowledge_base.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_authors(raw: Any) -> list[Author]:
    if not raw:
        return []
    names = [str(a) for a in raw] if isinstance(raw, list) else re.split(r"[;]", str(raw))
    authors: list[Author] = []
    for name in names:
        name = name.strip()
        if name:
            authors.append(Author(name=name))
    return authors


def _paper_id(paper: dict[str, Any], index: int) -> str:
    doi = _norm_doi(_clean_str(paper.get("doi")))
    if doi:
        return doi
    return f"kb-{index}"


def load_corpus(path: Path | None = None) -> Corpus:
    """Build a :class:`~lfx_insights.models.Corpus` from the knowledge base.

    The top-level ``papers`` array is the canonical record set. Papers without a
    DOI get a stable synthetic id (``kb-<index>``) so citations can still resolve.
    """
    kb_path = path or find_knowledge_base_path()
    if kb_path is None or not kb_path.exists():
        return Corpus(kb_id="lfx-insights-kb", papers=[])

    data = json.loads(kb_path.read_text(encoding="utf-8"))
    raw_papers: list[dict[str, Any]] = data.get("papers") or []

    papers: list[Paper] = []
    for i, raw in enumerate(raw_papers):
        if not isinstance(raw, dict):
            continue
        title = _clean_str(raw.get("title"))
        if not title:
            continue
        doi = _norm_doi(_clean_str(raw.get("doi")))
        papers.append(
            Paper(
                id=_paper_id(raw, i),
                title=title,
                doi=doi or None,
                authors=_parse_authors(raw.get("authors")),
                year=_clean_int(raw.get("year")),
                abstract=_clean_str(raw.get("abstract")),
                source=_clean_str(raw.get("venue")) or _clean_str(raw.get("source")),
                url=_clean_str(raw.get("url")),
            )
        )
    return Corpus(kb_id="lfx-insights-kb", papers=papers)


def retrieve(
    corpus: Corpus, query: str, k: int = 5
) -> list[tuple[Paper, float]]:
    """Offline lexical retrieval: score papers by token overlap with ``query``.

    Title tokens count triple, abstract/authors/source tokens count once. Returns
    the top ``k`` papers with a non-zero score, highest first.
    """
    if not corpus.papers or not query.strip():
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[Paper, float]] = []
    for paper in corpus.papers:
        title = _tokenize(paper.title)
        abstract = _tokenize(paper.abstract or "")
        authors = _tokenize(" ".join(a.name for a in paper.authors))
        src = _tokenize(paper.source or "")
        qset = set(q_tokens)
        score = (
            len(qset & set(title)) * 3.0
            + len(qset & set(abstract))
            + len(qset & set(authors))
            + len(qset & set(src)) * 0.5
        )
        if score > 0:
            scored.append((paper, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def snippet(paper: Paper, length: int = 280) -> str:
    """Return a short, clean excerpt of a paper's abstract for display."""
    text = (paper.abstract or paper.title or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "…"


def evidence_payload(papers: list[tuple[Paper, float]]) -> list[dict[str, Any]]:
    """Shape retrieved papers for the API / task pane evidence area."""
    out: list[dict[str, Any]] = []
    for paper, score in papers:
        out.append({
            "paper_id": paper.id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "year": paper.year,
            "doi": paper.doi,
            "url": paper.url,
            "source": paper.source,
            "relevance": round(score, 3),
            "snippet": snippet(paper),
        })
    return out


def ground_numeric_markers(text: str, papers: list[Paper]) -> str:
    """Replace ``[1]``/``[1;3]`` numeric labels with the matching paper DOI/id.

    The LLM is prompted to cite retrieved sources by their 1-based list index.
    This maps those indices back to verified paper identifiers so the shared
    citation renderer can resolve them; any token that is not a provided index
    is left untouched.
    """
    ids = {i + 1: p.id for i, p in enumerate(papers)}
    if not ids:
        return text

    def _replace(match: re.Match[str]) -> str:
        lead, inner = match.group(1), match.group(2)
        toks = [t.strip() for t in re.split(r"[;,]", inner) if t.strip()]
        out: list[str] = []
        for t in toks:
            if t.isdigit() and int(t) in ids:
                out.append(ids[int(t)])
            else:
                out.append(t)
        if not out:
            return ""
        return f"{lead}[{';'.join(out)}]"

    return re.sub(r"(\s*)\[([^\[\]]+)\]", _replace, text)


def render_citations(
    text: str, papers: list[Paper], corpus: Corpus
) -> tuple[str, str, list[str]]:
    """Render a draft (with ``[n]`` or ``[doi]`` markers) into APA form.

    Returns ``(intext_text, reference_list, cited_ids)``. Only citations that
    resolve to a provided/verified paper are kept — hallucinated references are
    dropped by the shared citation logic.
    """
    grounded_ids = [p.id for p in papers]
    grounded_text = ground_numeric_markers(text, papers)
    rendered, placed_ids = apply_intext_citations(
        grounded_text, grounded_ids, corpus
    )
    cited_papers = [p for p in papers if p.id in placed_ids]
    if not cited_papers:
        cited_papers = [p for p in papers if p.id in grounded_ids]
    references = format_reference_list(cited_papers, corpus) if cited_papers else ""
    return rendered, references, placed_ids


def find_apparent_citations(
    text: str, corpus: Corpus
) -> list[dict[str, Any]]:
    """Find every APA author-year citation in ``text`` and resolve it.

    Returns one dict per apparent citation with ``author``, ``year`` and
    ``paper_id`` (``None`` when it does not match any verified record). This is
    the anti-hallucination core of the 'Verify citations' action: citations the
    model (or a pasted reference) invents are surfaced as unresolved.
    """
    out: list[dict[str, Any]] = []
    for match in _APPARENT_PAREN.finditer(text):
        for part in match.group(1).split(";"):
            part = part.strip()
            ay = _AUTH_YEAR.match(part)
            if ay is None:
                continue
            author_part, year_str = ay.group(1), ay.group(2)
            year = int(year_str.rstrip("abcd"))
            # For "et al." / "&" forms use the first author's surname; otherwise
            # the last token (handles "First Last" and "Last" equally).
            low = author_part.lower()
            if "et al" in low or "&" in author_part:
                surname = author_part.split()[0]
            else:
                surname = author_part.split()[-1]
            paper_id = None
            for paper in corpus.papers:
                if paper.year != year:
                    continue
                surnames = [_surname(a.name) for a in paper.authors]
                if not surnames:
                    continue
                if ("&" in author_part or "et al." in author_part):
                    if surname in surnames:
                        paper_id = paper.id
                        break
                elif surnames[0] == surname:
                    paper_id = paper.id
                    break
            out.append({"author": surname, "year": year, "paper_id": paper_id})
    return out


def verify_text_citations(text: str, corpus: Corpus) -> dict[str, Any]:
    """Validate the citations present in ``text`` against the corpus.

    Resolves both APA author-year citations (via :func:`find_apparent_citations`)
    and ``[doi]`` markers. Returns a structured verification result used by the
    add-in's 'Verify citations' action. Invented references are reported as
    unresolved so they can never be silently accepted.
    """
    apparent = find_apparent_citations(text, corpus)
    cited_ids = [a["paper_id"] for a in apparent if a["paper_id"]]
    unresolved = [
        f"{a['author']} ({a['year']})" for a in apparent if not a["paper_id"]
    ]
    all_exist = not unresolved
    verified = bool(apparent) and all_exist
    evidence = [
        {
            "paper_id": p.id,
            "title": p.title,
            "authors": [a.name for a in p.authors],
            "year": p.year,
            "doi": p.doi,
            "url": p.url,
            "source": p.source,
            "relevance": 1.0,
            "snippet": snippet(p),
        }
        for pid in cited_ids
        if (p := corpus.by_id(pid)) is not None
    ]
    issues: list[str] = []
    if not apparent:
        issues.append("No recognizable citations found in the text.")
    for label in unresolved:
        issues.append(f"Citation '{label}' does not match any verified record.")
    return {
        "verified": verified,
        "cited_ids": cited_ids,
        "unresolved": unresolved,
        "all_exist": all_exist,
        "issues": issues,
        "evidence": evidence,
    }
