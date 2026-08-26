"""Tests for full text intelligence."""

from __future__ import annotations

import pytest

from lfx_insights.lifescience.full_text import (
    chunk_text,
    detect_sections,
    extract_references,
)

pytestmark = pytest.mark.unit


def test_detect_sections_basic() -> None:
    text = """Abstract
This is the abstract.

Introduction
Background information here.

Methods
We performed RNA-seq analysis.

Results
We found significant changes.

Discussion
Our findings suggest."""

    sections = detect_sections(text)
    names = [s.section_name for s in sections]
    assert "Abstract" in names
    assert "Introduction" in names
    assert "Methods" in names
    assert "Results" in names
    assert "Discussion" in names


def test_detect_sections_no_headings() -> None:
    text = "This is just a plain paragraph with no headings."
    sections = detect_sections(text)
    assert len(sections) >= 1


def test_chunk_text_basic() -> None:
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    assert all(c.chunk_id for c in chunks)
    assert all(c.text for c in chunks)


def test_chunk_text_empty() -> None:
    chunks = chunk_text("", max_tokens=50)
    assert chunks == []


def test_chunk_text_short() -> None:
    text = "Short text"
    chunks = chunk_text(text, max_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text"


def test_extract_references_basic() -> None:
    text = """Some text here.

References
[1] Smith J. Gene therapy advances. Nature. 2024;100:1-10.
[2] Jones K. CRISPR applications. Science. 2023;50:20-30."""

    refs = extract_references(text)
    assert len(refs) == 2
    assert refs[0].ref_id == "ref_0"
    assert "Smith" in refs[0].text


def test_extract_references_none() -> None:
    text = "No references section here."
    refs = extract_references(text)
    assert refs == []


def test_extract_full_text_missing_file(tmp_path: object) -> None:
    from lfx_insights.lifescience.full_text import extract_full_text

    result = extract_full_text("/nonexistent/file.pdf")
    assert result is None
