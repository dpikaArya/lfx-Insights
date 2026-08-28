"""Focused tests for full-text / table / figure evidence support.

Covers the enhanced ``lfx_insights.lifescience.full_text`` extraction (provenance +
caching, lazy/graceful when PDF libs are absent) and the opt-in full-text wiring in the
``full_text_evidence_extraction`` / ``scientific_claim_graph`` scripts (abstract-only by
default, full text only when a matching PDF is present).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo-root standalone scripts importable under pytest's src-only pythonpath.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import full_text_evidence_extraction as fte
import scientific_claim_graph as scg
from lfx_insights.lifescience import full_text as ft
from lfx_insights.projects.schemas import (
    EvidenceRecord,
    ExtractedFigure,
    ExtractedTable,
    PaperSection,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Library: lfx_insights.lifescience.full_text
# --------------------------------------------------------------------------- #
def test_extract_pages_missing_file() -> None:
    assert ft.extract_pages("/nonexistent/file.pdf") is None


def test_detect_sections_from_pages_sets_pages() -> None:
    pages = [
        "Abstract\nAbstract text here.",
        "Methods\nWe did X.\nResults\nWe found Y.",
        "Discussion\nWe suggest Z.",
    ]
    sections = ft.detect_sections_from_pages(pages)
    names = {s.section_name for s in sections}
    assert {"Abstract", "Methods", "Results", "Discussion"} <= names
    results = next(s for s in sections if s.section_name == "Results")
    # Results begins on page 2 and spans until the Discussion heading on page 3.
    assert results.page_start == 2 and results.page_end == 3


def test_chunk_text_with_pages_sets_page() -> None:
    pages = ["word " * 10, "word " * 10]
    chunks = ft.chunk_text("word " * 30, max_tokens=15, pages=pages)
    assert chunks
    assert all(c.page is not None for c in chunks)


def test_extract_figures_missing_file() -> None:
    assert ft.extract_figures("/nonexistent/file.pdf") == []


def test_extract_figures_parses_captions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"not a real pdf")
    pages = [
        "Results\nFigure 1. Expression rises with dose (x-axis: dose; y-axis: response).",
        "Figure 2. Survival differs between groups.",
    ]
    monkeypatch.setattr(ft, "extract_pages", lambda p, **k: pages)
    figs = ft.extract_figures(pdf)
    assert {f.figure_number for f in figs} == {1, 2}
    fig1 = next(f for f in figs if f.figure_number == 1)
    assert fig1.axis_labels is not None
    assert "dose" in fig1.axis_labels.lower()
    assert fig1.page == 1


def test_extract_structure_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    calls = {"n": 0}
    pages = ["Figure 3. A caption about something."]

    def fake_extract_pages(p, **k):
        calls["n"] += 1
        return pages

    monkeypatch.setattr(ft, "extract_pages", fake_extract_pages)
    figs1 = ft.extract_figures(pdf)
    figs2 = ft.extract_figures(pdf)
    # Second call must be served from cache, not re-extracting pages.
    assert calls["n"] == 1
    assert [f.figure_number for f in figs1] == [3]
    assert [f.figure_number for f in figs2] == [3]


def test_extracted_models() -> None:
    t = ExtractedTable(table_number=1, headers=["A", "B"], rows=[["1", "2"]])
    assert t.source_type == "table"
    f = ExtractedFigure(figure_number=2, caption="cap")
    assert f.source_type == "figure"


def test_evidence_record_provenance() -> None:
    rec = EvidenceRecord(
        evidence_id="e1",
        claim_id="c1",
        document_id="d1",
        passage="p",
        support_type="supporting",
        confidence=0.9,
        source_type="table",
        table_number=2,
    )
    assert rec.source_type == "table"
    assert rec.table_number == 2


# --------------------------------------------------------------------------- #
# Scripts: opt-in full-text, abstract-only by default
# --------------------------------------------------------------------------- #
def _make_papers_csv(tmp_path: Path, title: str = "Paper A", doi: str = "10.1/abc") -> Path:
    df = pd.DataFrame([{
        "title": title,
        "doi": doi,
        "abstract": "We found a significant effect (p<0.01).",
    }])
    p = tmp_path / "papers.csv"
    df.to_csv(p, index=False)
    return p


def test_extraction_abstract_only_no_fulltext_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(fte, "extract_pages", lambda *a, **k: calls.append("pages") or None)
    monkeypatch.setattr(fte, "extract_tables", lambda *a, **k: calls.append("tables") or [])
    monkeypatch.setattr(fte, "extract_figures", lambda *a, **k: calls.append("figures") or [])
    monkeypatch.setattr(fte, "find_paper_pdf", lambda *a, **k: calls.append("find") or None)
    p = _make_papers_csv(tmp_path)
    out = tmp_path / "out"
    df = fte.build_evidence_matrix(str(p), out)
    # Abstract-only must not discover, download, parse, or cache any PDF.
    assert "pages" not in calls and "tables" not in calls and "figures" not in calls
    assert "find" not in calls
    assert df.iloc[0]["Source_Type"] == "abstract"
    assert not (tmp_path / ".lfx_struct_cache").exists()


def test_extraction_full_text_activates_extractions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "10.1_abc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls: list[str] = []
    monkeypatch.setattr(fte, "find_paper_pdf", lambda *a, **k: calls.append("find") or pdf)
    monkeypatch.setattr(fte, "extract_pages", lambda p, **k: calls.append("pages") or ["Abstract\nX.\nResults\nWe found Y."])
    monkeypatch.setattr(fte, "detect_sections_from_pages", lambda pages: [
        PaperSection(section_name="Results", paragraphs=["We found Y."])
    ])
    monkeypatch.setattr(fte, "extract_tables", lambda p, **k: calls.append("tables") or [ExtractedTable(table_number=1, caption="Tbl")])
    monkeypatch.setattr(fte, "extract_figures", lambda p, **k: calls.append("figures") or [ExtractedFigure(figure_number=1, caption="Fig")])
    p = _make_papers_csv(tmp_path)
    out = tmp_path / "out"
    df = fte.build_evidence_matrix(str(p), out, pdf_dir=str(tmp_path), full_text=True)
    # Full text must activate page + table + figure extraction exactly when requested.
    assert {"find", "pages", "tables", "figures"} <= set(calls)
    assert df.iloc[0]["Source_Type"] == "full_text"


def test_extraction_full_text_wires_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "10.1_abc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(fte, "find_paper_pdf", lambda *a, **k: pdf)
    monkeypatch.setattr(fte, "extract_pages", lambda p, **k: ["Abstract\nX.\nResults\nWe found Y."])
    monkeypatch.setattr(fte, "detect_sections_from_pages", lambda pages: [
        PaperSection(section_name="Results", paragraphs=["We found Y."])
    ])
    monkeypatch.setattr(fte, "extract_tables", lambda p, **k: [ExtractedTable(table_number=1, caption="Tbl")])
    monkeypatch.setattr(fte, "extract_figures", lambda p, **k: [ExtractedFigure(figure_number=1, caption="Fig")])
    p = _make_papers_csv(tmp_path)
    out = tmp_path / "out"
    df = fte.build_evidence_matrix(str(p), out, pdf_dir=str(tmp_path), full_text=True)
    row = df.iloc[0]
    assert row["Source_Type"] == "full_text"
    assert "Tbl" in row["Tables"]
    assert "Fig" in row["Figures"]


class _FakeST:
    def __init__(self, *a, **k) -> None:
        pass

    def encode(self, texts, show_progress_bar: bool = False):
        return np.array([[1.0, 0.0] for _ in texts])


def test_claim_graph_full_text_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scg, "SentenceTransformer", _FakeST)
    pdf = tmp_path / "10.1_abc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(scg, "find_paper_pdf", lambda *a, **k: pdf)
    monkeypatch.setattr(scg, "extract_pages", lambda p, **k: [
        "Results\nWe found that X improves Y (Figure 1)."
    ])
    monkeypatch.setattr(scg, "detect_sections_from_pages", lambda pages: [
        PaperSection(section_name="Results", paragraphs=["We found that X improves Y (Figure 1)."], page_start=1)
    ])
    p = _make_papers_csv(tmp_path)
    claims = scg.extract_claims_from_papers(str(p), pdf_dir=str(tmp_path), full_text=True)
    assert claims
    assert claims[0]["source_type"] == "full_text"
    assert claims[0]["section"] == "Results"
    assert claims[0]["figure_number"] == 1
    graph = scg.build_claim_graph(claims, str(p))
    node = graph["claims"][0]
    assert node["source_type"] == "full_text"
    assert node["figure_number"] == 1


def test_claim_graph_abstract_only_no_fulltext_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(scg, "extract_pages", lambda *a, **k: calls.append("pages") or None)
    monkeypatch.setattr(scg, "find_paper_pdf", lambda *a, **k: calls.append("find") or None)
    monkeypatch.setattr(scg, "SentenceTransformer", _FakeST)
    p = _make_papers_csv(tmp_path)
    claims = scg.extract_claims_from_papers(str(p))  # no --full-text, no --pdf-dir
    assert "pages" not in calls and "find" not in calls
    assert claims
    assert all(c["source_type"] == "abstract" for c in claims)
