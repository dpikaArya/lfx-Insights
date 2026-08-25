"""Assay-aware omics detection over a corpus.

Deterministic, no LLM. For each paper we scan its title+abstract with
word-boundary regexes for assay-specific signal terms and map matches to an
omics type plus the canonical public repositories where such data are deposited.

Correctness notes (the dangerous failure mode here is *misclassification*):

* Detection is **assay-aware** — specific assay terms win over generic
  sequencing terms. In particular ``RNA-seq`` is *transcriptomics*, not genomics,
  and ``16S`` is *metagenomics*, not generic genomics. We achieve this by
  collecting all matched terms per paper and then dropping the broad ``genomics``
  bucket whenever any specific omics type (transcriptomics/epigenomics/
  proteomics/metabolomics/metagenomics) also matched in the same paper — so a
  generic ``genome sequencing`` mention alongside ``RNA-seq`` yields
  transcriptomics only.
* Matching uses ``\\b`` word boundaries, never naive substring containment, so
  ``RNA-seq`` does not get mistaken for a bare ``RNA`` mention and ``mrna`` does
  not fire on unrelated text.

Each detected omics type yields exactly one :class:`~consilium.models.Insight`
listing the supporting papers and the matched terms, tagged with an EDAM topic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from consilium.models import EvidenceRef, Insight

if TYPE_CHECKING:
    from consilium.models import Corpus, Paper


class _OmicsRule:
    """A single omics type: its trigger terms, repositories, and EDAM label."""

    __slots__ = ("edam", "name", "patterns", "repos", "terms")

    def __init__(self, name: str, terms: list[str], repos: list[str], edam: str) -> None:
        self.name = name
        self.terms = terms
        self.repos = repos
        self.edam = edam
        self.patterns = [(t, re.compile(_term_regex(t), re.IGNORECASE)) for t in terms]

    def matches(self, text: str) -> list[str]:
        """Matched terms (in declaration order), de-duplicated."""
        return [term for term, pat in self.patterns if pat.search(text)]


def _term_regex(term: str) -> str:
    r"""Build a word-boundary regex for a (possibly hyphen/slash-bearing) term.

    We anchor with ``(?<!\w)`` / ``(?!\w)`` rather than a bare ``\b`` so that the
    boundary is correct even when the term begins or ends with a non-word
    character context (e.g. digits in ``16s`` or the slash in ``lc-ms/ms``).
    Internal whitespace is made flexible (any run of whitespace).
    """
    escaped = re.escape(term)
    # Allow any whitespace run where the term has a single space.
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


# Order is for deterministic emission only. Assay-aware suppression of the broad
# genomics bucket (see _classify_paper) is what guarantees an "RNA-seq" paper
# reports transcriptomics, never genomics — independent of declaration order.
_RULES: list[_OmicsRule] = [
    _OmicsRule(
        "transcriptomics",
        [
            "rna-seq",
            "rnaseq",
            "scrna",
            "single-cell rna",
            "transcriptome",
            "mrna expression",
        ],
        ["GEO", "ArrayExpress", "ENA"],
        "EDAM:topic_3170 (transcriptomics)",
    ),
    _OmicsRule(
        "epigenomics",
        ["chip-seq", "atac-seq", "bisulfite", "methylation", "dna methylation"],
        ["GEO", "ENA"],
        "EDAM:topic_3173 (epigenomics)",
    ),
    _OmicsRule(
        "proteomics",
        ["proteome", "proteomics", "mass spectrometry protein", "lc-ms/ms protein"],
        ["PRIDE", "ProteomeXchange"],
        "EDAM:topic_0121 (proteomics)",
    ),
    _OmicsRule(
        "metabolomics",
        ["metabolome", "metabolomics", "lc-ms metabolite", "untargeted metabolomics"],
        ["MetaboLights", "Metabolomics Workbench"],
        "EDAM:topic_3172 (metabolomics)",
    ),
    _OmicsRule(
        "metagenomics",
        ["16s", "metagenome", "metagenomics", "microbiome", "shotgun metagenomic"],
        ["ENA", "MG-RAST"],
        "EDAM:topic_3174 (metagenomics)",
    ),
    _OmicsRule(
        "genomics",
        [
            "whole-genome",
            "wgs",
            "wes",
            "exome",
            "variant calling",
            "gwas",
            "genome sequencing",
        ],
        ["SRA", "ENA", "dbGaP"],
        "EDAM:topic_3673 (genomics)",
    ),
]


_SPECIFIC_OMICS: frozenset[str] = frozenset(
    {"transcriptomics", "epigenomics", "proteomics", "metabolomics", "metagenomics"}
)


def _classify_paper(paper: Paper, rules: list[_OmicsRule]) -> dict[str, list[str]]:
    """Map each omics type the paper triggers to its matched terms.

    Assay-aware: the generic ``genomics`` bucket is dropped whenever any specific
    omics type (transcriptomics/epigenomics/proteomics/metabolomics/metagenomics)
    also matched in the same paper. So an ``RNA-seq`` + ``genome sequencing``
    paper classifies as transcriptomics only, never genomics — generic
    sequencing terms must not produce a spurious genomics tag once a specific
    assay is present.
    """
    text = paper.text()
    hits: dict[str, list[str]] = {}
    for rule in rules:
        matched = rule.matches(text)
        if matched:
            hits[rule.name] = matched
    # Suppress the broad genomics bucket when a specific omics type also matched.
    if "genomics" in hits and _SPECIFIC_OMICS.intersection(hits):
        del hits["genomics"]
    return hits


def detect_omics(corpus: Corpus) -> list[Insight]:
    """Detect omics signal across the corpus, one :class:`Insight` per omics type.

    For each paper, title+abstract is scanned with word-boundary regexes for
    assay-specific terms. Detection is assay-aware: ``RNA-seq`` classifies as
    transcriptomics (never genomics) and ``16S`` as metagenomics. The broad
    ``genomics`` bucket is dropped for any paper that also triggered a specific
    omics type, so generic sequencing wording never co-tags a specific assay.

    Each detected omics type produces one deterministic :class:`Insight`:

    * ``statement`` names the type and its suggested deposition repositories.
    * ``evidence`` is one :class:`EvidenceRef` per matching paper (corpus order).
    * ``tags`` is ``["bioinformatics", <type>]``.
    * ``reasoning`` lists the matched terms and the EDAM topic label.

    Empty corpus or no omics signal -> ``[]``.
    """
    if len(corpus) == 0:
        return []

    # Per omics type: ordered (paper_id -> matched terms), preserving corpus order.
    per_type: dict[str, list[tuple[str, list[str]]]] = {}
    for paper in corpus.papers:
        hits = _classify_paper(paper, _RULES)
        for omics_type, terms in hits.items():
            per_type.setdefault(omics_type, []).append((paper.id, terms))

    insights: list[Insight] = []
    # Emit in rule-declaration order for deterministic, stable output.
    for rule in _RULES:
        entries = per_type.get(rule.name)
        if not entries:
            continue
        evidence = [EvidenceRef(paper_id=pid) for pid, _ in entries]
        matched_terms: list[str] = []
        for _, terms in entries:
            for term in terms:
                if term not in matched_terms:
                    matched_terms.append(term)
        repos = ", ".join(rule.repos)
        statement = f"Detected {rule.name} signal; suggested repositories: {repos}."
        reasoning = f"Matched terms: {', '.join(matched_terms)}. Classified as {rule.edam}."
        insights.append(
            Insight(
                statement=statement,
                evidence=evidence,
                is_synthesized=True,
                tags=["bioinformatics", rule.name],
                reasoning=reasoning,
            )
        )
    return insights
