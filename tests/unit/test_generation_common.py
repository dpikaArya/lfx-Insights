from __future__ import annotations

import pytest

from lfx_insights.generation.common import (
    apply_intext_citations,
    disambiguation_suffixes,
    filter_verified_citations,
    format_apa,
    format_apa_intext,
    format_authors,
    ground_cited,
    ground_quote,
    grounded_evidence,
    has_output_leak,
    render_intext_group,
    verify_citation,
)
from lfx_insights.models import Author, Corpus, Paper

pytestmark = pytest.mark.unit


def test_format_authors_variants() -> None:
    assert format_authors(["Alice Smith"]) == "Smith, A."
    assert format_authors(["Smith, Alice"]) == "Smith, A."
    assert format_authors(["Alice Smith", "Bob Lee"]) == "Smith, A., & Lee, B."
    assert (
        format_authors(["Alice Smith", "Bob Lee", "Carla Diaz"]) == "Smith, A., Lee, B., & Diaz, C."
    )
    # surname particle retained
    assert format_authors(["Jan van Dijk"]) == "van Dijk, J."


def test_format_apa() -> None:
    p = Paper(
        id="W1",
        title="Graph neural networks for molecules.",
        doi="10.1/x",
        authors=[Author(name="Alice Smith")],
        year=2021,
        source="Journal of ML",
    )
    apa = format_apa(p)
    assert "Smith, A. (2021)." in apa
    assert "Graph neural networks for molecules." in apa
    assert "https://doi.org/10.1/x" in apa
    # missing year -> n.d.
    assert "(n.d.)" in format_apa(Paper(id="W2", title="T"))
    # falsy/zero year -> n.d. (not "(0).")
    apa_zero = format_apa(Paper(id="W3", title="T", year=0))
    assert "(n.d.)" in apa_zero
    assert "(0)" not in apa_zero
    # DOI already in full-URL form must not be double-prefixed
    apa_url = format_apa(Paper(id="W4", title="T", doi="https://doi.org/10.1/y"))
    assert "https://doi.org/10.1/y" in apa_url
    assert "https://doi.org/https://doi.org/" not in apa_url


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[Paper(id="W1", title="T", doi="10.1/x")],
    )


def test_verify_and_filter_citations() -> None:
    c = _corpus()
    assert verify_citation("W1", c) is True
    assert verify_citation("10.1/x", c) is True
    assert verify_citation("https://doi.org/10.1/x", c) is True
    assert verify_citation("W999", c) is False
    assert verify_citation("", c) is False
    assert filter_verified_citations(["W1", "W999", "W1", "10.1/x"], c) == ["W1", "10.1/x"]


def test_verify_citation_corpus_stores_full_url_doi() -> None:
    # Corpus stores DOIs in full-URL form; a bare-DOI ref must still resolve.
    c = Corpus(
        kb_id="kb",
        papers=[Paper(id="W1", title="T", doi="https://doi.org/10.1/x")],
    )
    assert verify_citation("10.1/x", c) is True
    assert verify_citation("https://doi.org/10.1/x", c) is True
    assert verify_citation("http://doi.org/10.1/x", c) is True
    assert verify_citation("10.1/MISSING", c) is False


def test_ground_quote_and_leak_detection() -> None:
    assert ground_quote("neural networks", ["we use neural networks here"]) is True
    assert ground_quote("teleportation", ["we use neural networks here"]) is False
    # snake_case template names and the standalone NaN cluster label are leaks
    assert has_output_leak("Theme {top_theme} is great") is True
    assert has_output_leak("Cluster Nan has 3 papers") is True
    assert has_output_leak("The {example} placeholder") is True
    assert has_output_leak("A clean sentence about transformers.") is False
    # legitimate prose containing a brace expression must NOT be flagged
    assert has_output_leak("The set {x} where x ranges over inputs.") is False
    assert has_output_leak("Define the singleton {a} for each element.") is False
    # 'nan' as a substring of a real word is not a leak (word-boundary)
    assert has_output_leak("The financier completed the analysis.") is False


def test_ground_cited_requires_a_grounding_quote() -> None:
    c = Corpus(
        kb_id="kb",
        papers=[Paper(id="W1", title="t", abstract="graph neural networks predict properties")],
    )
    # A quote that is verbatim-present in the paper text is kept.
    assert ground_cited([("W1", "neural networks")], c) == ["W1"]
    # An ungrounded quote (paraphrase/hallucination) is dropped.
    assert ground_cited([("W1", "quantum teleportation")], c) == []
    # A citation with no quote is dropped (id-membership alone is insufficient now).
    assert ground_cited([("W1", "")], c) == []
    # An unresolvable paper is dropped.
    assert ground_cited([("W404", "neural networks")], c) == []


def test_grounded_evidence_returns_verified_pairs() -> None:
    c = Corpus(
        kb_id="kb", papers=[Paper(id="W1", title="t", abstract="alphafold predicts structure")]
    )
    pairs = grounded_evidence([("W1", "predicts structure"), ("W1", "made up"), ("W404", "x")], c)
    assert pairs == [("W1", "predicts structure")]


# --- APA in-text citation rendering ---------------------------------------


def test_format_apa_intext_author_counts() -> None:
    one = Paper(id="W1", title="T", year=2020, authors=[Author(name="Alice Smith")])
    two = Paper(
        id="W2", title="T", year=2019, authors=[Author(name="Alice Smith"), Author(name="Bob Lee")]
    )
    three = Paper(
        id="W3",
        title="T",
        year=2018,
        authors=[Author(name="Alice Smith"), Author(name="Bob Lee"), Author(name="Carla Diaz")],
    )
    assert format_apa_intext(one) == "Smith, 2020"
    assert format_apa_intext(two) == "Smith & Lee, 2019"
    assert format_apa_intext(three) == "Smith et al., 2018"
    # surname particle retained
    assert format_apa_intext(
        Paper(id="W4", title="T", year=2017, authors=[Author(name="Jan van Dijk")])
    ) == "van Dijk, 2017"


def test_format_apa_intext_missing_year_and_authors() -> None:
    # No year -> n.d.
    assert format_apa_intext(
        Paper(id="W1", title="T", authors=[Author(name="Alice Smith")])
    ) == "Smith, n.d."
    # No authors -> short-title fallback
    no_auth = format_apa_intext(Paper(id="W2", title="Deep learning for proteins", year=2020))
    assert no_auth == '"Deep learning for", 2020'


def test_render_intext_group_multiple_papers() -> None:
    a = Paper(id="A", title="T", year=2020, authors=[Author(name="Alice Smith")])
    b = Paper(id="B", title="T", year=2019, authors=[Author(name="Bob Lee")])
    assert render_intext_group([a]) == "(Smith, 2020)"
    assert render_intext_group([a, b]) == "(Smith, 2020; Lee, 2019)"


def _intext_corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="GNNs", year=2020, authors=[Author(name="Alice Smith")]),
            Paper(id="W2", title="Generative design", year=2019, authors=[Author(name="Bob Lee")]),
        ],
    )


def test_apply_intext_citations_substitutes_grounded_marker() -> None:
    text, placed = apply_intext_citations(
        "Recent work advances modeling [W1].", ["W1"], _intext_corpus()
    )
    assert text == "Recent work advances modeling (Smith, 2020)."
    assert placed == ["W1"]


def test_apply_intext_citations_strips_ungrounded_marker_and_space() -> None:
    # W2 resolves to a corpus paper but is NOT grounded -> the marker is removed
    # along with its leading space, leaving clean prose.
    text, placed = apply_intext_citations(
        "Generative approaches help [W2]. Modeling improves [W1].", ["W1"], _intext_corpus()
    )
    assert text == "Generative approaches help. Modeling improves (Smith, 2020)."
    assert placed == ["W1"]


def test_apply_intext_citations_groups_multiple_ids() -> None:
    text, placed = apply_intext_citations(
        "Prior work [W1; W2] shows this.", ["W1", "W2"], _intext_corpus()
    )
    assert text == "Prior work (Smith, 2020; Lee, 2019) shows this."
    assert placed == ["W1", "W2"]


def test_apply_intext_citations_leaves_non_citation_brackets() -> None:
    # A bracket whose token does not resolve to any corpus paper is prose, untouched.
    text, placed = apply_intext_citations(
        "The result was wrong [sic] but cited [W1].", ["W1"], _intext_corpus()
    )
    assert "[sic]" in text
    assert text.endswith("cited (Smith, 2020).")
    assert placed == ["W1"]


# --- APA year-disambiguation (a/b/c) --------------------------------------


def test_format_apa_intext_with_suffix() -> None:
    p = Paper(id="A", title="T", year=2020, authors=[Author(name="Alice Smith")])
    assert format_apa_intext(p, "b") == "Smith, 2020b"
    # n.d. with a suffix uses the APA "n.d.-a" form
    nd = Paper(id="B", title="T", authors=[Author(name="Alice Smith")])
    assert format_apa_intext(nd, "a") == "Smith, n.d.-a"


def test_format_apa_reference_with_suffix() -> None:
    p = Paper(
        id="A", title="GNNs", year=2020, authors=[Author(name="Alice Smith")], doi="10.1/x"
    )
    assert "Smith, A. (2020b)." in format_apa(p, "b")


def test_disambiguation_suffixes_orders_collisions_by_title() -> None:
    beta = Paper(id="A", title="Beta methods", year=2020, authors=[Author(name="Alice Smith")])
    alpha = Paper(id="B", title="Alpha methods", year=2020, authors=[Author(name="Alice Smith")])
    solo = Paper(id="C", title="Solo", year=2019, authors=[Author(name="Alice Smith")])
    sx = disambiguation_suffixes([beta, alpha, solo])
    # Same author+year (Smith 2020): ordered by title -> "Alpha" 'a', "Beta" 'b'.
    assert sx["B"] == "a"
    assert sx["A"] == "b"
    # Unique author+year -> no suffix.
    assert sx["C"] == ""


def test_disambiguation_collides_on_et_al_label() -> None:
    # Two DISTINCT 3-author papers that both render "Smith et al., 2020" must
    # still be disambiguated (the in-text labels would otherwise be identical).
    p1 = Paper(
        id="P1",
        title="Zeta",
        year=2020,
        authors=[Author(name="Alice Smith"), Author(name="Bob Lee"), Author(name="Carla Diaz")],
    )
    p2 = Paper(
        id="P2",
        title="Alpha",
        year=2020,
        authors=[Author(name="Alice Smith"), Author(name="Dan Poe"), Author(name="Eve Ng")],
    )
    sx = disambiguation_suffixes([p1, p2])
    assert sx["P2"] == "a"  # "Alpha" before "Zeta"
    assert sx["P1"] == "b"


def test_disambiguation_spills_past_z() -> None:
    # 27 colliding references: the 27th (index 26, by title order) becomes "aa".
    papers = [
        Paper(id=f"P{i:02d}", title=f"T{i:02d}", year=2020, authors=[Author(name="Alice Smith")])
        for i in range(27)
    ]
    sx = disambiguation_suffixes(papers)
    assert sx["P00"] == "a"
    assert sx["P25"] == "z"
    assert sx["P26"] == "aa"


# --- Citation validation helpers ------------------------------------------


def _val_corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecules",
                doi="10.1/x",
                year=2020,
                authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
                abstract="Message passing networks predict molecular properties accurately.",
            ),
            Paper(
                id="W2",
                title="Generative models for de novo design",
                doi="10.1/y",
                year=2019,
                authors=[Author(name="Carol Lee")],
                abstract="Latent variable models generate novel drug-like molecules.",
            ),
        ],
    )


def test_extract_cited_paper_ids_resolves_apa() -> None:
    from lfx_insights.generation.common import extract_cited_paper_ids

    corpus = _val_corpus()
    text = "Recent work (Smith & Jones, 2020) advances modeling. Earlier (Lee, 2019) showed results."
    ids = extract_cited_paper_ids(text, corpus)
    assert "W1" in ids
    assert "W2" in ids


def test_extract_cited_paper_ids_dedupes() -> None:
    from lfx_insights.generation.common import extract_cited_paper_ids

    corpus = _val_corpus()
    text = "Work (Smith & Jones, 2020) and also (Smith & Jones, 2020) again."
    ids = extract_cited_paper_ids(text, corpus)
    assert ids.count("W1") == 1


def test_extract_cited_paper_ids_skips_unresolvable() -> None:
    from lfx_insights.generation.common import extract_cited_paper_ids

    corpus = _val_corpus()
    text = "Work by Unknown (2025) and also (Smith & Jones, 2020)."
    ids = extract_cited_paper_ids(text, corpus)
    assert ids == ["W1"]


def test_validate_citations_in_text_all_exist() -> None:
    from lfx_insights.generation.common import validate_citations_in_text

    corpus = _val_corpus()
    text = "Work (Smith & Jones, 2020) and (Lee, 2019) support this."
    result = validate_citations_in_text(text, corpus)
    assert result["all_exist"] is True
    assert len(result["cited_ids"]) == 2


def test_validate_citations_in_text_no_citations() -> None:
    from lfx_insights.generation.common import validate_citations_in_text

    corpus = _val_corpus()
    text = "No citations here, just plain text."
    result = validate_citations_in_text(text, corpus)
    assert result["cited_ids"] == []
    assert result["all_exist"] is True


def test_build_cited_reference_list_dedupes_by_doi() -> None:
    from lfx_insights.generation.common import build_cited_reference_list

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text [W1].", citations=["W1"]),
        GeneratedSection(name="methods", text="More [W1] and [W2].", citations=["W1", "W2"]),
    ]
    ref_list = build_cited_reference_list(sections, corpus)
    assert len(ref_list) == 2
    ids = [p.id for p in ref_list]
    assert ids == ["W1", "W2"]


def test_build_cited_reference_list_skips_missing() -> None:
    from lfx_insights.generation.common import build_cited_reference_list

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text.", citations=["W1", "W404"]),
    ]
    ref_list = build_cited_reference_list(sections, corpus)
    assert len(ref_list) == 1
    assert ref_list[0].id == "W1"


def test_format_reference_list_sorted() -> None:
    from lfx_insights.generation.common import format_reference_list

    corpus = _val_corpus()
    ref_list = format_reference_list(list(corpus.papers), corpus)
    lines = ref_list.strip().split("\n")
    assert len(lines) == 2
    # APA references are sorted alphabetically
    assert lines[0] < lines[1]


def test_validate_manuscript_citations_all_exist() -> None:
    from lfx_insights.generation.common import validate_manuscript_citations

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text.", citations=["W1", "W2"]),
    ]
    result = validate_manuscript_citations(sections, corpus)
    assert result["all_exist"] is True
    assert result["total_cited"] == 2
    assert result["reference_count"] == 2
    assert result["issues"] == []


def test_validate_manuscript_citations_missing_paper() -> None:
    from lfx_insights.generation.common import validate_manuscript_citations

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text.", citations=["W1", "W999"]),
    ]
    result = validate_manuscript_citations(sections, corpus)
    assert result["all_exist"] is False
    assert len(result["issues"]) == 1
    assert "W999" in result["issues"][0]


def test_build_evidence_chain_maps_papers() -> None:
    from lfx_insights.generation.common import build_evidence_chain

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text citing.", citations=["W1"]),
    ]
    chain = build_evidence_chain(sections, corpus)
    assert len(chain) == 1
    assert chain[0]["section"] == "intro"
    assert len(chain[0]["citations"]) == 1
    assert chain[0]["citations"][0]["paper_id"] == "W1"
    assert chain[0]["citations"][0]["title"] == "Graph neural networks for molecules"
    assert chain[0]["citations"][0]["doi"] == "10.1/x"


def test_build_evidence_chain_skips_missing() -> None:
    from lfx_insights.generation.common import build_evidence_chain

    from lfx_insights.models import GeneratedSection

    corpus = _val_corpus()
    sections = [
        GeneratedSection(name="intro", text="Text.", citations=["W1", "W404"]),
    ]
    chain = build_evidence_chain(sections, corpus)
    assert len(chain[0]["citations"]) == 1
    assert chain[0]["citations"][0]["paper_id"] == "W1"
