"""Full text PDF intelligence — extraction, section detection, chunking, references."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx_insights.models import Corpus
    from lfx_insights.projects.schemas import (
        ExtractedFigure,
        ExtractedTable,
        PaperReference,
        PaperSection,
        TextChunk,
    )

_HEADING_RE = re.compile(
    r"^(?:abstract|introduction|background|methods?|materials?\s+and\s+methods?|"
    r"results?|discussion|conclusion|acknowledgment|references|bibliography|"
    r"supplementary)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_REF_RE = re.compile(
    r"^\[?\d+\]?[\.\s]+(.+)$", re.MULTILINE
)


def extract_pages(pdf_path: str | Path) -> list[str] | None:
    """Extract per-page text from a PDF. Tries pypdf -> pdfplumber -> pymupdf.

    Returns a list of page texts, or None if every engine failed / the file is missing.
    Pages are used for provenance (page numbers) and for figure caption parsing without a
    vision model.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None

    # Try pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        if any(pages):
            return pages
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        if any(pages):
            return pages
    except Exception:
        pass

    # Try pymupdf
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        if any(pages):
            return pages
    except Exception:
        pass

    return None


def extract_full_text(pdf_path: str | Path) -> str | None:
    """Extract text from a PDF as a single string. Returns None on failure."""
    pages = extract_pages(pdf_path)
    if not pages:
        return None
    text = "\n\n".join(pages).strip()
    return text or None


def detect_sections(text: str) -> list[PaperSection]:
    """Detect standard paper sections using heading patterns."""
    from lfx_insights.projects.schemas import PaperSection

    sections: list[PaperSection] = []
    lines = text.split("\n")

    current_name = "Body"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _HEADING_RE.match(stripped):
            if current_lines:
                sections.append(PaperSection(
                    section_name=current_name,
                    paragraphs=["\n".join(current_lines)],
                ))
            current_name = stripped.title()
            current_lines = []
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        sections.append(PaperSection(
            section_name=current_name,
            paragraphs=["\n".join(current_lines)],
        ))

    return sections


def detect_sections_from_pages(pages: list[str]) -> list[PaperSection]:
    """Detect sections across a list of page texts, populating page_start/page_end."""
    from lfx_insights.projects.schemas import PaperSection

    sections: list[PaperSection] = []
    current_name = "Body"
    current_lines: list[str] = []
    current_start_page: int | None = None

    for page_idx, page_text in enumerate(pages, start=1):
        for line in page_text.split("\n"):
            stripped = line.strip()
            if _HEADING_RE.match(stripped):
                if current_lines:
                    sections.append(PaperSection(
                        section_name=current_name,
                        paragraphs=["\n".join(current_lines)],
                        page_start=current_start_page,
                        page_end=page_idx,
                    ))
                current_name = stripped.title()
                current_lines = []
                current_start_page = page_idx
            else:
                if stripped:
                    current_lines.append(stripped)

    if current_lines:
        sections.append(PaperSection(
            section_name=current_name,
            paragraphs=["\n".join(current_lines)],
            page_start=current_start_page,
            page_end=len(pages),
        ))

    return sections


def chunk_text(
    text: str,
    section_name: str = "",
    max_tokens: int = 512,
    pages: list[str] | None = None,
) -> list[TextChunk]:
    """Split text into overlapping chunks for retrieval. Approximates tokens as words.

    If ``pages`` is supplied (the source page texts), each chunk is annotated with the
    approximate page it starts on for provenance.
    """
    from lfx_insights.projects.schemas import TextChunk

    words = text.split()
    if not words:
        return []

    page_for_word: list[int] | None = None
    if pages:
        page_for_word = []
        for _page_idx, _page_text in enumerate(pages, start=1):
            page_for_word.extend([_page_idx] * len(_page_text.split()))

    chunks: list[TextChunk] = []
    overlap = max_tokens // 4
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_tokens]
        page: int | None = None
        if page_for_word:
            page = page_for_word[min(i, len(page_for_word) - 1)]
        chunks.append(TextChunk(
            chunk_id=uuid.uuid4().hex[:8],
            text=" ".join(chunk_words),
            paragraph_index=i // max_tokens,
            section_name=section_name or None,
            page=page,
        ))
        i += max_tokens - overlap

    return chunks


def extract_references(text: str) -> list[PaperReference]:
    """Extract and parse references from the References section."""
    from lfx_insights.projects.schemas import PaperReference

    refs: list[PaperReference] = []
    # Find the References section
    ref_start = -1
    for marker in ["References", "Bibliography", "LITERATURE CITED"]:
        idx = text.find(marker)
        if idx >= 0:
            ref_start = idx + len(marker)
            break

    if ref_start < 0:
        return refs

    ref_text = text[ref_start:]
    matches = _REF_RE.findall(ref_text)

    for i, match in enumerate(matches[:100]):  # cap at 100 references
        raw = match.strip()
        # Try to extract DOI
        doi = None
        doi_match = re.search(r"10\.\d{4,}/[^\s]+", raw)
        if doi_match:
            doi = doi_match.group(0).rstrip(".")

        refs.append(PaperReference(
            ref_id=f"ref_{i}",
            text=raw,
            doi=doi,
        ))

    return refs


_TABLE_CAPTION_RE = re.compile(
    r"Table\s+(\d+[A-Za-z]?)\.?\s*(.+?)(?=\n\s*(?:Figure|Table)\s+\d|$)",
    re.IGNORECASE | re.DOTALL,
)
_FIGURE_CAPTION_RE = re.compile(
    r"Figure\s+(\d+[A-Za-z]?)\.?\s*(.+?)(?=\n\s*(?:Figure|Table)\s+\d|$)",
    re.IGNORECASE | re.DOTALL,
)
_AXIS_RE = re.compile(
    r"(?:x-axis|y-axis|vertical axis|horizontal axis|abscissa|ordinate)"
    r"\s*[:\-]?\s*([^.;]+)",
    re.IGNORECASE,
)


def _structure_cache_path(pdf_path: Path) -> Path:
    cache_dir = pdf_path.parent / ".lfx_struct_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = pdf_path.stat()
    key = f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{pdf_path.stem}_{digest}.json"


def _parse_figure_captions(pages: list[str]) -> list[ExtractedFigure]:
    """Parse figure captions + axis labels from page text. No vision model."""
    from lfx_insights.projects.schemas import ExtractedFigure

    figures: list[ExtractedFigure] = []
    for page_idx, page_text in enumerate(pages, start=1):
        for para in re.split(r"\n\s*\n", page_text):
            m = _FIGURE_CAPTION_RE.search(para)
            if not m:
                continue
            figure_number = int(re.sub(r"[^0-9].*", "", m.group(1)) or len(figures) + 1)
            caption = m.group(2).strip().replace("\n", " ")
            axis_labels = "; ".join(_AXIS_RE.findall(caption)) or None
            figures.append(ExtractedFigure(
                figure_number=figure_number,
                caption=caption[:500] or None,
                page=page_idx,
                axis_labels=axis_labels,
                nearby_results=para.strip()[:500] or None,
            ))
    return figures


def _find_nearby_caption(pages: list[str], page_idx: int, prefix: str) -> str | None:
    if 1 <= page_idx <= len(pages):
        m = re.search(
            rf"{prefix}\s+\d+[A-Za-z]?\.\s*(.+?)(?=\n\s*(?:Figure|Table)\s+\d|$)",
            pages[page_idx - 1],
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip().replace("\n", " ")[:500]
    return None


def extract_structure(
    pdf_path: str | Path,
    force: bool = False,
) -> tuple[list[ExtractedTable], list[ExtractedFigure]]:
    """Extract tables and figures deterministically, with a content-hash cache.

    Tables use pdfplumber when available; figure captions / axis labels are parsed from
    page text (no vision model). Both degrade gracefully to empty lists if libraries or
    content are unavailable.
    """
    from lfx_insights.projects.schemas import ExtractedFigure, ExtractedTable

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return [], []

    tables: list[ExtractedTable] = []
    figures: list[ExtractedFigure] = []

    cache_path = _structure_cache_path(pdf_path)
    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            tables = [ExtractedTable(**t) for t in data.get("tables", [])]
            figures = [ExtractedFigure(**f) for f in data.get("figures", [])]
            return tables, figures
        except Exception:
            pass

    pages = extract_pages(pdf_path)
    if pages is not None:
        figures = _parse_figure_captions(pages)
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                table_number = 0
                for page_idx, page in enumerate(pdf.pages, start=1):
                    for tbl in page.extract_tables():
                        if not tbl:
                            continue
                        table_number += 1
                        headers = [str(c).strip() if c is not None else "" for c in tbl[0]]
                        rows = [
                            [str(c).strip() if c is not None else "" for c in r]
                            for r in tbl[1:]
                            if any(c not in (None, "") for c in r)
                        ]
                        if not rows:
                            continue
                        caption = _find_nearby_caption(pages, page_idx, "Table")
                        tables.append(ExtractedTable(
                            table_number=table_number,
                            caption=caption,
                            page=page_idx,
                            headers=headers,
                            rows=rows,
                        ))
        except Exception:
            pass

    with contextlib.suppress(Exception):
        cache_path.write_text(json.dumps({
            "tables": [t.model_dump() for t in tables],
            "figures": [f.model_dump() for f in figures],
        }, default=str), encoding="utf-8")

    return tables, figures


def extract_tables(pdf_path: str | Path, force: bool = False) -> list[ExtractedTable]:
    """Extract tables from a PDF (cached). Returns [] if unavailable."""
    tables, _ = extract_structure(pdf_path, force=force)
    return tables


def extract_figures(pdf_path: str | Path, force: bool = False) -> list[ExtractedFigure]:
    """Extract figures (caption / axis labels) from a PDF (cached). Returns [] if unavailable."""
    _, figures = extract_structure(pdf_path, force=force)
    return figures


def find_paper_pdf(
    pdf_dir: str | Path,
    paper_id: str,
    doi: str | None = None,
    title: str | None = None,
) -> Path | None:
    """Locate a paper's PDF in ``pdf_dir`` by id / doi / truncated title.

    Used by the evidence / claim-graph scripts so full-text extraction is opt-in and only
    runs when a matching file is present (lazy, capability-detected, no extra backend calls).
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return None
    candidates = [str(paper_id)]
    if doi:
        candidates.append(str(doi))
    if title:
        candidates.append(str(title)[:40])
    for cand in candidates:
        safe = re.sub(r"[^\w\-\.]+", "_", cand).strip("_")
        for ext in (".pdf", ".PDF"):
            p = pdf_dir / f"{safe}{ext}"
            if p.exists():
                return p
    return None


def extract_full_text_for_corpus(
    corpus: Corpus,
    base_dir: str | Path,
    force: bool = False,
) -> dict[str, str]:
    """Extract full text for all papers, caching results. Returns {paper_id: full_text}."""
    cache_dir = Path(base_dir) / "fulltext_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "extraction_cache.json"

    cache: dict[str, str] = {}
    if not force and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    results: dict[str, str] = {}
    for paper in corpus.papers:
        if paper.id in cache:
            results[paper.id] = cache[paper.id]
            continue

        # Try to find a cached full text file
        paper_file = cache_dir / f"{paper.id}.txt"
        if paper_file.exists() and not force:
            text = paper_file.read_text(encoding="utf-8")
            results[paper.id] = text
            cache[paper.id] = text
            continue

        # No extraction possible from abstracts alone; the backend must provide full text
        # This function is a placeholder for when PDF paths are available
        pass

    if cache:
        cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return results
