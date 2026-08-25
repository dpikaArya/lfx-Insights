"""Public dataset accession discovery (deterministic, no LLM).

Scans each paper's ``text()`` for repository accession identifiers and emits one
:class:`~consilium.models.Insight` per *distinct* accession found across the corpus.
This surfaces the public data backing a literature set — a reproducibility signal —
without any model call: the accession itself is the evidence quote.

Recognised repositories:

* **GEO** — Gene Expression Omnibus series/samples/datasets (``GSE``/``GSM``/``GDS`` ``\\d+``)
* **SRA** — Sequence Read Archive runs/studies and BioProjects (``SRR``/``SRP``/``PRJNA``)
* **PRIDE** — proteomics datasets (``PXD\\d+``)
* **MetaboLights** — metabolomics studies (``MTBLS\\d+``)
* **ArrayExpress** — functional genomics (``E-XXXX-\\d+``)
* **ENA** — European Nucleotide Archive projects (``PRJEB\\d+``)
* **dbGaP** — controlled-access studies (``phs\\d+``)
* **EGA** — European Genome-phenome Archive studies/datasets (``EGAS``/``EGAD`` ``\\d+``)

Detection uses anchored, word-boundary patterns so accessions are matched as whole
tokens (e.g. ``GSE12345`` is found, but ``xGSE12345`` or ``GSE12345x`` is not).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from consilium.models import EvidenceRef, Insight

if TYPE_CHECKING:
    from consilium.models import Corpus


class _Repo(NamedTuple):
    """A repository name paired with its accession-matching pattern."""

    name: str
    pattern: re.Pattern[str]


def _accession(body: str) -> re.Pattern[str]:
    """Compile a word-boundary-anchored accession pattern.

    ``\\b`` on both sides keeps matches to whole tokens. The ArrayExpress form
    contains hyphens (not word characters) so its body supplies its own anchoring
    where ``\\b`` would not fire; the leading ``\\bE`` still guards the prefix.
    """
    return re.compile(rf"\b{body}\b")


# Order matters only for readability; patterns are mutually exclusive by prefix.
_REPOS: tuple[_Repo, ...] = (
    _Repo("GEO", _accession(r"G(?:SE|SM|DS)\d+")),
    _Repo("SRA", _accession(r"(?:SRR|SRP|PRJNA)\d+")),
    _Repo("PRIDE", _accession(r"PXD\d+")),
    _Repo("MetaboLights", _accession(r"MTBLS\d+")),
    _Repo("ArrayExpress", _accession(r"E-[A-Z]{4}-\d+")),
    _Repo("ENA", _accession(r"PRJEB\d+")),
    _Repo("dbGaP", _accession(r"phs\d+")),
    _Repo("EGA", _accession(r"EGA[SD]\d+")),
)


def _scan_text(text: str) -> list[tuple[str, str]]:
    """Return ``(accession, repo)`` pairs found in ``text``, in order of appearance.

    An accession is reported once (first hit wins) even if repeated; ordering follows
    the position of each accession's first occurrence so output is stable.
    """
    hits: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for repo in _REPOS:
        for match in repo.pattern.finditer(text):
            acc = match.group(0)
            if acc in seen:
                continue
            seen.add(acc)
            hits.append((match.start(), acc, repo.name))
    hits.sort(key=lambda h: h[0])
    return [(acc, repo) for _, acc, repo in hits]


def discover_datasets(corpus: Corpus, full_texts: dict[str, str] | None = None) -> list[Insight]:
    """Discover public dataset accessions referenced across the corpus.

    For each distinct accession (deduplicated across all papers, keeping the first
    paper in which it appears) emit one grounded :class:`Insight`. The accession
    string is recorded verbatim as the evidence quote.

    By default each paper's ``title + abstract`` is scanned; accessions usually live
    in a *Data Availability* section, so pass ``full_texts`` (``{paper_id: full text}``,
    e.g. fetched from Perspicacité) for meaningful recall.

    Returns ``[]`` for an empty corpus or when no accession is present.
    """
    texts = full_texts or {}
    insights: list[Insight] = []
    seen: set[str] = set()
    for paper in corpus.papers:
        scanned = texts.get(paper.id) or paper.text()
        source = "full text" if texts.get(paper.id) else "abstract"
        for acc, repo in _scan_text(scanned):
            if acc in seen:
                continue
            seen.add(acc)
            insights.append(
                Insight(
                    statement=f"Dataset {acc} ({repo}) referenced in '{paper.title}'.",
                    evidence=[
                        EvidenceRef(paper_id=paper.id, quote=acc, location="text"),
                    ],
                    tags=["dataset", repo],
                    reasoning=f"public (accession present in {source})",
                )
            )
    return insights
