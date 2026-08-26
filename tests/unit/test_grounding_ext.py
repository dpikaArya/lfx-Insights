"""Tests for extended grounding / citation verification."""

from __future__ import annotations

import pytest

from lfx_insights.standards.grounding import (
    verify_citation_in_text,
    verify_manuscript_grounding,
    verify_paragraph_grounding,
)

pytestmark = pytest.mark.unit


def test_verify_citation_in_text_found() -> None:
    papers = [
        {"title": "Gene therapy advances", "doi": "10.1234/example"},
        {"title": "CRISPR applications", "doi": None},
    ]
    assert verify_citation_in_text("Gene therapy advances (2024)", papers) is True


def test_verify_citation_in_text_not_found() -> None:
    papers = [{"title": "Gene therapy advances", "doi": None}]
    assert verify_citation_in_text("Random unrelated text", papers) is False


def test_verify_citation_in_text_empty() -> None:
    assert verify_citation_in_text("", [{"title": "x"}]) is False
    assert verify_citation_in_text("cite", []) is False


def test_verify_paragraph_grounding() -> None:
    paragraph = "This was shown by Smith et al. (2024) and confirmed by Jones (2023)."
    candidates = ["This was shown by Smith et al. (2024)", "Other text"]
    results = verify_paragraph_grounding(paragraph, candidates)
    assert len(results) == 2
    assert any(r["verified"] for r in results)


def test_verify_paragraph_grounding_no_citations() -> None:
    paragraph = "No citations here."
    results = verify_paragraph_grounding(paragraph, [])
    assert results == []


def test_verify_manuscript_grounding() -> None:
    sections = [
        {"name": "Introduction", "text": "Previous work (Smith 2024) showed X."},
        {"name": "Discussion", "text": "Our results align with (Jones 2023)."},
    ]
    candidates = ["Previous work (Smith 2024) showed X."]
    result = verify_manuscript_grounding(sections, candidates)
    assert result["total"] >= 1
    assert "verified" in result
    assert "unverified" in result
    assert "citations" in result
