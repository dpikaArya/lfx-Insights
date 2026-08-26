"""Full text PDF intelligence — extraction, section detection, chunking, references."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx_insights.models import Corpus
    from lfx_insights.projects.schemas import PaperReference, PaperSection, TextChunk

_HEADING_RE = re.compile(
    r"^(?:abstract|introduction|background|methods?|materials?\s+and\s+methods?|"
    r"results?|discussion|conclusion|acknowledgment|references|bibliography|"
    r"supplementary)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_REF_RE = re.compile(
    r"^\[?\d+\]?[\.\s]+(.+)$", re.MULTILINE
)


def extract_full_text(pdf_path: str | Path) -> str | None:
    """Extract text from a PDF. Tries pypdf -> pdfplumber -> pymupdf. Returns None on failure."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None

    # Try pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # Try pymupdf
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    return None


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


def chunk_text(text: str, section_name: str = "", max_tokens: int = 512) -> list[TextChunk]:
    """Split text into overlapping chunks for retrieval. Approximates tokens as words."""
    from lfx_insights.projects.schemas import TextChunk

    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    overlap = max_tokens // 4
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_tokens]
        chunks.append(TextChunk(
            chunk_id=uuid.uuid4().hex[:8],
            text=" ".join(chunk_words),
            paragraph_index=i // max_tokens,
            section_name=section_name or None,
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
