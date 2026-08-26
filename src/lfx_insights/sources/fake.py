"""In-memory backend for tests and ``--offline`` runs (no network)."""

from __future__ import annotations

from lfx_insights.models import Author, Corpus, Paper, Passage

_DEFAULT_PAPERS = [
    Paper(
        id="W1",
        title="Graph neural networks for molecular property prediction",
        doi="10.1000/gnn",
        authors=[Author(name="A Smith")],
        year=2021,
        abstract="We use graph neural networks to predict molecular properties for drug discovery.",
        source="fake",
    ),
    Paper(
        id="W2",
        title="Deep generative models for de novo drug design",
        doi="10.1000/gen",
        authors=[Author(name="B Lee")],
        year=2022,
        abstract="Generative deep learning proposes novel molecules with desired properties.",
        source="fake",
    ),
    Paper(
        id="W3",
        title="Transformer architectures for protein structure prediction",
        doi="10.1000/prot",
        authors=[Author(name="C Diaz")],
        year=2021,
        abstract="Transformers predict protein structure from sequence with high accuracy.",
        source="fake",
    ),
    Paper(
        id="W4",
        title="Self-supervised learning of protein language models",
        doi="10.1000/plm",
        authors=[Author(name="D Kim")],
        year=2023,
        abstract="Protein language models learn representations from unlabeled sequences.",
        source="fake",
    ),
]


class FakeBackend:
    """Returns canned data; satisfies :class:`RetrievalBackend`."""

    def __init__(self, corpus: Corpus | None = None, passages: list[Passage] | None = None) -> None:
        self._corpus = corpus or Corpus(kb_id="fake-kb", papers=list(_DEFAULT_PAPERS))
        self._passages = passages

    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        return Corpus(kb_id=self._corpus.kb_id, papers=self._corpus.papers[:max_papers])

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        if self._passages is not None:
            return self._passages[:k]
        return [
            Passage(paper_id=p.id, text=p.abstract or p.title, location="abstract")
            for p in self._corpus.papers[:k]
        ]

    def paper_content(self, paper_id: str) -> str:
        paper = self._corpus.by_id(paper_id)
        return paper.text() if paper else ""
