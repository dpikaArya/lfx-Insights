from __future__ import annotations

from pathlib import Path

import pytest

from lfx_insights.models import GeneratedSection, Paper, SectionBundle

pytestmark = pytest.mark.unit

# Skip the whole module if the optional [docx] extra is not installed.
pytest.importorskip("docx")

from lfx_insights.reporting.docx_export import render_docx, write_docx


def _bundle() -> SectionBundle:
    return SectionBundle(
        title="Manuscript Draft",
        sections=[
            GeneratedSection(name="introduction", text="First para.\n\nSecond para.", citations=["W1"]),
        ],
        references=[
            Paper(id="W1", title="Zeta study", year=2020, source="Nature", doi="10.1/z"),
            Paper(id="W2", title="Alpha study", year=2019, source="Cell"),
        ],
    )


def test_render_docx_has_sections_and_apa_references() -> None:
    doc = render_docx(_bundle())
    text = "\n".join(p.text for p in doc.paragraphs)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert "Manuscript Draft" in headings
    assert any("Introduction" in h for h in headings)
    assert "First para." in text
    assert "Second para." in text
    assert "References (APA)" in headings
    # APA references sorted alphabetically by the rendered string: "Alpha" before "Zeta".
    alpha_i = text.index("Alpha study")
    zeta_i = text.index("Zeta study")
    assert alpha_i < zeta_i
    assert "https://doi.org/10.1/z" in text


def test_render_docx_empty_references_is_valid() -> None:
    bundle = SectionBundle(title="Empty", sections=[], references=[])
    doc = render_docx(bundle)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "No references." in text


def test_write_docx_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "manuscript.docx"
    returned = write_docx(_bundle(), out)
    assert returned == out
    assert out.exists() and out.stat().st_size > 0


def test_render_docx_keeps_intext_citations_in_prose() -> None:
    # Section prose carries APA in-text citations (baked in at generation time);
    # they must survive rendering and be backed by a matching reference entry.
    from lfx_insights.models import Author

    bundle = SectionBundle(
        title="Manuscript Draft",
        sections=[
            GeneratedSection(
                name="introduction",
                text="Message passing improves accuracy (Smith & Jones, 2020).",
                citations=["W1"],
            ),
        ],
        references=[
            Paper(
                id="W1",
                title="GNNs",
                year=2020,
                source="Nature",
                doi="10.1/z",
                authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
            ),
        ],
    )
    doc = render_docx(bundle)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "(Smith & Jones, 2020)" in text  # in-text citation preserved
    assert "Smith, A., & Jones, B. (2020)." in text  # matching APA reference entry


def test_render_docx_reference_list_disambiguates_collisions() -> None:
    # Two same-author/year references must carry a/b suffixes in the reference
    # list, matching the in-text citations baked into the prose.
    from lfx_insights.models import Author

    bundle = SectionBundle(
        title="Draft",
        sections=[
            GeneratedSection(
                name="introduction",
                text="Foo (Smith, 2020a) and bar (Smith, 2020b).",
                citations=["A", "B"],
            ),
        ],
        references=[
            Paper(id="A", title="Alpha", year=2020, authors=[Author(name="Alice Smith")], doi="10.1/a"),
            Paper(id="B", title="Beta", year=2020, authors=[Author(name="Alice Smith")], doi="10.1/b"),
        ],
    )
    doc = render_docx(bundle)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Smith, A. (2020a). Alpha." in text
    assert "Smith, A. (2020b). Beta." in text
