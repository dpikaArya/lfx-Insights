from __future__ import annotations

import pytest

from lfx_insights.lifescience.bioinformatics import detect_omics
from lfx_insights.models import Corpus, Paper

pytestmark = pytest.mark.unit


def _corpus(*papers: Paper) -> Corpus:
    return Corpus(kb_id="kb", papers=list(papers))


def _omics_of(insights: list, type_: str):  # type: ignore[no-untyped-def]
    return next((i for i in insights if type_ in i.tags), None)


def test_rna_seq_is_transcriptomics_not_genomics() -> None:
    corpus = _corpus(
        Paper(
            id="T1",
            title="An RNA-seq study of liver",
            abstract="We performed RNA-seq to quantify gene expression.",
        )
    )
    insights = detect_omics(corpus)
    assert len(insights) == 1
    ins = insights[0]
    assert ins.tags == ["bioinformatics", "transcriptomics"]
    # CRITICAL: must NOT be tagged genomics.
    assert "genomics" not in ins.tags
    assert "GEO" in ins.statement
    assert ins.statement == (
        "Detected transcriptomics signal; suggested repositories: GEO, ArrayExpress, ENA."
    )
    assert ins.evidence[0].paper_id == "T1"
    assert ins.is_synthesized is True
    assert ins.reasoning is not None
    assert "EDAM:topic_3170 (transcriptomics)" in ins.reasoning
    assert "rna-seq" in ins.reasoning


def test_16s_amplicon_is_metagenomics() -> None:
    corpus = _corpus(
        Paper(
            id="M1",
            title="Gut 16S rRNA amplicon survey",
            abstract="16S rRNA amplicon sequencing of the gut microbiome.",
        )
    )
    insights = detect_omics(corpus)
    meta = _omics_of(insights, "metagenomics")
    assert meta is not None
    assert meta.tags == ["bioinformatics", "metagenomics"]
    assert "ENA" in meta.statement
    assert "MG-RAST" in meta.statement
    assert "genomics" not in meta.tags
    assert meta.reasoning is not None
    assert "16s" in meta.reasoning
    assert "microbiome" in meta.reasoning


def test_unrelated_text_yields_nothing() -> None:
    corpus = _corpus(
        Paper(
            id="U1",
            title="A history of impressionist painting",
            abstract="This essay discusses brushwork and colour in 19th century art.",
        )
    )
    assert detect_omics(corpus) == []


def test_empty_corpus_returns_empty() -> None:
    assert detect_omics(Corpus(kb_id="kb", papers=[])) == []


def test_word_boundary_no_false_substring_match() -> None:
    # "exome" must not fire on "axexome"-like words; "16s" must not fire on
    # "1116s" or "16software". "wes" must not fire on "western" or "weston".
    corpus = _corpus(
        Paper(
            id="N1",
            title="Western blotting in cell biology",
            abstract="We discuss the western blot and transcriptomewide caveats.",
        )
    )
    # "transcriptomewide" must NOT match the "transcriptome" term (boundary).
    assert detect_omics(corpus) == []


def test_genomics_detected_for_wgs_wes_exome() -> None:
    corpus = _corpus(
        Paper(
            id="G1",
            title="Whole-genome sequencing of a cohort",
            abstract="WGS and exome (WES) with variant calling and a GWAS.",
        )
    )
    insights = detect_omics(corpus)
    geno = _omics_of(insights, "genomics")
    assert geno is not None
    assert geno.tags == ["bioinformatics", "genomics"]
    assert "SRA" in geno.statement
    assert "dbGaP" in geno.statement
    assert geno.reasoning is not None
    assert "EDAM:topic_3673 (genomics)" in geno.reasoning
    # The matched terms should include the generic genomics triggers.
    for term in ("whole-genome", "wgs", "wes", "exome", "variant calling", "gwas"):
        assert term in geno.reasoning


def test_epigenomics_atac_and_chip_seq() -> None:
    corpus = _corpus(
        Paper(
            id="E1",
            title="ATAC-seq and ChIP-seq of chromatin",
            abstract="Bisulfite sequencing revealed DNA methylation changes.",
        )
    )
    epi = _omics_of(detect_omics(corpus), "epigenomics")
    assert epi is not None
    assert "GEO" in epi.statement
    assert epi.reasoning is not None
    assert "atac-seq" in epi.reasoning
    assert "chip-seq" in epi.reasoning
    assert "methylation" in epi.reasoning


def test_proteomics_and_metabolomics() -> None:
    corpus = _corpus(
        Paper(
            id="P1",
            title="Plasma proteome profiling",
            abstract="LC-MS/MS protein quantification of the proteome.",
        ),
        Paper(
            id="X1",
            title="Untargeted metabolomics of serum",
            abstract="The metabolome was profiled by LC-MS metabolite analysis.",
        ),
    )
    insights = detect_omics(corpus)
    prot = _omics_of(insights, "proteomics")
    metab = _omics_of(insights, "metabolomics")
    assert prot is not None
    assert "PRIDE" in prot.statement
    assert "ProteomeXchange" in prot.statement
    assert metab is not None
    assert "MetaboLights" in metab.statement
    assert "Metabolomics Workbench" in metab.statement


def test_one_insight_per_type_with_multiple_papers() -> None:
    corpus = _corpus(
        Paper(id="T1", title="RNA-seq one", abstract="rna-seq analysis"),
        Paper(id="T2", title="RNA-seq two", abstract="transcriptome profiling"),
    )
    insights = detect_omics(corpus)
    trans = [i for i in insights if "transcriptomics" in i.tags]
    assert len(trans) == 1
    # One EvidenceRef per matching paper, in corpus order.
    assert [e.paper_id for e in trans[0].evidence] == ["T1", "T2"]


def test_multi_omics_paper_gets_each_specific_insight() -> None:
    # A paper triggering several specific omics types reports each of them.
    # The generic genomics signal (WGS, variant calling) is *suppressed* here
    # because specific assays (RNA-seq, proteomics) also matched.
    corpus = _corpus(
        Paper(
            id="MULTI",
            title="Multi-omics integration",
            abstract="We combined RNA-seq, WGS variant calling, and proteomics.",
        )
    )
    insights = detect_omics(corpus)
    tags = {t for i in insights for t in i.tags if t != "bioinformatics"}
    assert {"transcriptomics", "proteomics"} <= tags
    # Assay-aware: generic genomics is dropped when specific omics types matched.
    assert "genomics" not in tags
    # Every insight points back at the single source paper.
    for ins in insights:
        assert [e.paper_id for e in ins.evidence] == ["MULTI"]


def test_genomics_suppressed_when_specific_omics_present() -> None:
    # The verified finding: a paper mentioning RNA-seq AND a generic genomics
    # trigger ("genome sequence" / "aligned to the genome" plus "genome
    # sequencing") must classify as transcriptomics ONLY, never genomics.
    corpus = _corpus(
        Paper(
            id="RG1",
            title="RNA-seq of tumour tissue",
            abstract=(
                "We performed RNA-seq and the reads were aligned to the genome. "
                "Genome sequencing of matched normals was used as reference."
            ),
        )
    )
    insights = detect_omics(corpus)
    # Exactly one insight, and it is transcriptomics.
    assert len(insights) == 1
    ins = insights[0]
    assert ins.tags == ["bioinformatics", "transcriptomics"]
    assert "genomics" not in ins.tags
    # No genomics insight is emitted at all.
    assert _omics_of(insights, "genomics") is None
    assert ins.evidence[0].paper_id == "RG1"


def test_genomics_kept_when_no_specific_omics() -> None:
    # Suppression only fires when a specific omics type co-occurs. A pure
    # generic-genomics paper still reports genomics.
    corpus = _corpus(
        Paper(
            id="GEN1",
            title="Population genomics",
            abstract="Whole-genome sequencing with variant calling across a cohort.",
        )
    )
    insights = detect_omics(corpus)
    geno = _omics_of(insights, "genomics")
    assert geno is not None
    assert geno.tags == ["bioinformatics", "genomics"]


def test_deterministic_and_stable_order() -> None:
    corpus = _corpus(
        Paper(id="A", title="metabolomics", abstract="untargeted metabolomics"),
        Paper(id="B", title="rna-seq", abstract="transcriptome"),
        Paper(id="C", title="wgs", abstract="exome variant calling"),
    )
    first = detect_omics(corpus)
    second = detect_omics(corpus)
    assert [i.statement for i in first] == [i.statement for i in second]
    # Emission follows rule-declaration order: transcriptomics before
    # metabolomics before genomics.
    order = [i.tags[1] for i in first]
    assert order.index("transcriptomics") < order.index("metabolomics")
    assert order.index("metabolomics") < order.index("genomics")
