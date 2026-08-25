from __future__ import annotations

import pytest

from consilium.lifescience.datasets import discover_datasets
from consilium.models import Corpus, Paper

pytestmark = pytest.mark.unit


def _corpus(papers: list[Paper]) -> Corpus:
    return Corpus(kb_id="kb", papers=papers)


def test_empty_corpus_returns_empty() -> None:
    assert discover_datasets(_corpus([])) == []


def test_no_accession_returns_empty() -> None:
    paper = Paper(id="W1", title="A study", abstract="No data were deposited anywhere.")
    assert discover_datasets(_corpus([paper])) == []


def test_geo_and_pride_yield_two_insights() -> None:
    paper = Paper(
        id="W1",
        title="Multi-omics study",
        abstract="Data deposited in GEO under GSE12345 and PRIDE PXD000001.",
    )
    insights = discover_datasets(_corpus([paper]))
    assert len(insights) == 2

    # Order follows position in text: GSE12345 precedes PXD000001.
    geo, pride = insights
    assert geo.statement == "Dataset GSE12345 (GEO) referenced in 'Multi-omics study'."
    assert geo.tags == ["dataset", "GEO"]
    assert pride.statement == "Dataset PXD000001 (PRIDE) referenced in 'Multi-omics study'."
    assert pride.tags == ["dataset", "PRIDE"]


def test_insight_fields_are_correct() -> None:
    paper = Paper(id="W7", title="Repro paper", abstract="See GSE999.")
    [insight] = discover_datasets(_corpus([paper]))
    assert insight.reasoning == "public (accession present in abstract)"
    assert len(insight.evidence) == 1
    ev = insight.evidence[0]
    assert ev.paper_id == "W7"
    assert ev.quote == "GSE999"
    assert ev.location == "text"


def test_dedup_repeated_accession_across_papers_keeps_first() -> None:
    papers = [
        Paper(id="W1", title="First mention", abstract="We use GSE12345 here."),
        Paper(id="W2", title="Second mention", abstract="Reanalysis of GSE12345."),
    ]
    insights = discover_datasets(_corpus(papers))
    assert len(insights) == 1
    # First paper wins: title and evidence paper_id come from W1.
    assert insights[0].statement == "Dataset GSE12345 (GEO) referenced in 'First mention'."
    assert insights[0].evidence[0].paper_id == "W1"


def test_dedup_within_single_paper() -> None:
    paper = Paper(
        id="W1",
        title="Repeated",
        abstract="GSE12345 was reanalysed; GSE12345 again confirms the result.",
    )
    insights = discover_datasets(_corpus([paper]))
    assert len(insights) == 1


def test_all_repository_types_detected() -> None:
    paper = Paper(
        id="W1",
        title="Everything",
        abstract=(
            "Sequencing: SRR1234567, study SRP001, bioproject PRJNA000123. "
            "ENA PRJEB12345. Proteomics PXD000001. Metabolomics MTBLS42. "
            "Transcriptomics GSE100 and ArrayExpress E-MTAB-1234."
        ),
    )
    insights = discover_datasets(_corpus([paper]))
    repos = {insight.tags[1] for insight in insights}
    assert repos == {"SRA", "ENA", "PRIDE", "MetaboLights", "GEO", "ArrayExpress"}
    # Three distinct SRA accessions (SRR, SRP, PRJNA) all classified as SRA.
    sra = [i for i in insights if i.tags[1] == "SRA"]
    accs = {i.evidence[0].quote for i in sra}
    assert accs == {"SRR1234567", "SRP001", "PRJNA000123"}


def test_arrayexpress_accession_quote_is_verbatim() -> None:
    paper = Paper(id="W1", title="AE", abstract="Available at ArrayExpress E-MTAB-9999.")
    [insight] = discover_datasets(_corpus([paper]))
    assert insight.tags == ["dataset", "ArrayExpress"]
    assert insight.evidence[0].quote == "E-MTAB-9999"


def test_word_boundary_avoids_embedded_false_positives() -> None:
    # An accession-looking substring glued to other word characters must NOT match.
    paper = Paper(
        id="W1",
        title="Glued",
        abstract="The gene xGSE12345 and the label GSE12345x are not accessions.",
    )
    assert discover_datasets(_corpus([paper])) == []


def test_word_boundary_allows_punctuation_delimited_accession() -> None:
    paper = Paper(id="W1", title="Punct", abstract="Deposited (GSE12345); see also.")
    [insight] = discover_datasets(_corpus([paper]))
    assert insight.evidence[0].quote == "GSE12345"


def test_accession_requires_digits() -> None:
    # Bare prefixes with no number are not accessions.
    paper = Paper(id="W1", title="Bare", abstract="We discuss GEO, SRA, and PRIDE generally.")
    assert discover_datasets(_corpus([paper])) == []


def test_cross_paper_distinct_accessions_all_reported_in_order() -> None:
    papers = [
        Paper(id="W1", title="P1", abstract="GSE1 here."),
        Paper(id="W2", title="P2", abstract="PXD2 here."),
    ]
    insights = discover_datasets(_corpus(papers))
    assert len(insights) == 2
    assert insights[0].evidence[0].quote == "GSE1"
    assert insights[0].evidence[0].paper_id == "W1"
    assert insights[1].evidence[0].quote == "PXD2"
    assert insights[1].evidence[0].paper_id == "W2"


def test_full_texts_override_enables_accession_recall() -> None:
    from consilium.lifescience.datasets import discover_datasets
    from consilium.models import Corpus, Paper

    c = Corpus(kb_id="k", papers=[Paper(id="W1", title="t", abstract="no accession in abstract")])
    assert discover_datasets(c) == []  # abstract has none
    out = discover_datasets(c, full_texts={"W1": "Data available at GEO GSE12345."})
    assert len(out) == 1
    assert out[0].tags == ["dataset", "GEO"]
    assert "full text" in (out[0].reasoning or "")


@pytest.mark.parametrize(
    "accession,repo",
    [
        ("GSM9116755", "GEO"),  # GEO sample (was missed: only GSE matched before)
        ("GDS1234", "GEO"),
        ("GSE12345", "GEO"),
        ("phs000123", "dbGaP"),
        ("EGAS00001000123", "EGA"),
        ("EGAD00001000123", "EGA"),
    ],
)
def test_extended_accession_formats(accession: str, repo: str) -> None:
    from consilium.lifescience.datasets import discover_datasets
    from consilium.models import Corpus, Paper

    c = Corpus(kb_id="k", papers=[Paper(id="W1", title="t", abstract=f"data in {accession}.")])
    out = discover_datasets(c)
    assert len(out) == 1
    assert out[0].tags == ["dataset", repo]
    assert out[0].evidence[0].quote == accession
