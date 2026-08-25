"""Retrieval-ablation backends for the eval harness.

Three backends share the :class:`~consilium.sources.base.RetrievalBackend`
protocol so the runner can swap them per condition:

* :class:`NullBackend` — closed-book (no retrieval); the floor condition.
* :class:`TfidfBackend` — a generic lexical retriever over a *fixed* candidate
  pool (a vanilla TF-IDF baseline, deliberately **not** Perspicacité).
* :class:`OracleBackend` — perfect retrieval: returns a case's own gold contexts.
  The ceiling condition; it isolates *generation/citation* quality from *retrieval*
  quality (if oracle is high but perspicacite is low, the gap is retrieval).
* the Perspicacité backend (built lazily by :func:`build_eval_backend`) — the
  real, Consilium-owned literature backend under test.

The ablation answers "does Perspicacité retrieval lift Consilium's grounded
answers above a closed-book floor and a generic lexical baseline, and how close is
it to the oracle ceiling?".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from consilium.models import Corpus, Passage

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix

    from consilium.models import Paper
    from consilium.sources.base import RetrievalBackend


class NullBackend:
    """Closed-book backend: retrieves nothing.

    Establishes the floor condition for the ablation — answers are generated with
    no supporting corpus, so any grounded citation will fail verification.
    """

    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        """Return an empty corpus (no KB is built)."""
        return Corpus(kb_id="null", papers=[])

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        """Return no passages."""
        return []

    def paper_content(self, paper_id: str) -> str:
        """Return empty content (no corpus to read from)."""
        return ""


class OracleBackend:
    """Perfect-retrieval backend: returns a fixed set of gold documents as the corpus.

    Constructed per case with that case's own gold contexts, so generation sees exactly
    the right papers. The ablation ceiling — it decouples generation/citation quality
    from retrieval quality.
    """

    def __init__(self, docs: list[Paper]) -> None:
        self.docs = docs

    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        """Return the gold documents (truncated to ``max_papers``) as the corpus."""
        return Corpus(kb_id="oracle", papers=self.docs[:max_papers])

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        """Return up to ``k`` abstract-level passages from the gold documents."""
        return [
            Passage(paper_id=p.id, text=p.abstract or p.title, location="abstract")
            for p in self.docs[:k]
        ]

    def paper_content(self, paper_id: str) -> str:
        """Return the gold paper's ``text()`` (title + abstract), or ``""``."""
        paper = next((p for p in self.docs if p.id == paper_id), None)
        return paper.text() if paper is not None else ""


class TfidfBackend:
    """Generic lexical retriever over a fixed candidate pool (a TF-IDF baseline).

    This is intentionally a vanilla baseline, **not** Perspicacité: it ranks a
    fixed pool of :class:`~consilium.models.Paper` by cosine similarity of TF-IDF
    vectors. The fitted vectorizer and document matrix are computed lazily on the
    first ranking call and cached for the lifetime of the instance.
    """

    def __init__(self, pool: list[Paper]) -> None:
        """Store the candidate ``pool``; fitting is deferred until first use."""
        self.pool = pool
        self._vectorizer: object | None = None
        self._matrix: csr_matrix | None = None
        self._fitted = False

    def _ensure_fitted(self) -> None:
        """Lazily fit the TF-IDF vectorizer + document matrix over the pool.

        An empty pool leaves the backend unfitted so that ranking calls
        short-circuit to empty results instead of raising.
        """
        if self._fitted:
            return
        if not self.pool:
            self._fitted = True
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english")
        # L2-normalized by default, so cosine similarity == linear kernel.
        matrix = vectorizer.fit_transform(p.text() for p in self.pool)
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._fitted = True

    def _rank(self, query: str) -> list[tuple[Paper, float]]:
        """Rank the pool by descending cosine similarity of ``query`` to each paper.

        Returns ``(paper, score)`` pairs; an empty pool or empty query yields no
        results without raising.
        """
        self._ensure_fitted()
        if self._vectorizer is None or self._matrix is None or not query.strip():
            return []
        from sklearn.metrics.pairwise import linear_kernel

        query_vec = self._vectorizer.transform([query])  # type: ignore[attr-defined]
        sims: np.ndarray = linear_kernel(query_vec, self._matrix).ravel()
        order = sims.argsort()[::-1]
        return [(self.pool[i], float(sims[i])) for i in order]

    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        """Return the top-``max_papers`` pool papers most similar to ``topic``."""
        ranked = self._rank(topic)
        papers = [p for p, _ in ranked[:max_papers]]
        return Corpus(kb_id="tfidf", papers=papers)

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        """Return up to ``k`` abstract-level passages most relevant to ``query``."""
        ranked = self._rank(query)
        return [
            Passage(paper_id=p.id, text=p.abstract or p.title, location="abstract")
            for p, _ in ranked[:k]
        ]

    def paper_content(self, paper_id: str) -> str:
        """Return the pool paper's ``text()`` (title + abstract), or ``""``."""
        paper = next((p for p in self.pool if p.id == paper_id), None)
        return paper.text() if paper is not None else ""


def build_eval_backend(
    condition: str,
    *,
    pool: list[Paper] | None = None,
    case_docs: list[Paper] | None = None,
    url: str = "http://localhost:8002/mcp",
    timeout: int = 60,
) -> RetrievalBackend:
    """Construct the retrieval backend for an ablation ``condition``.

    Parameters
    ----------
    condition:
        One of ``"null"``, ``"tfidf"``, ``"oracle"``, or ``"perspicacite"``.
    pool:
        Candidate papers for the ``"tfidf"`` baseline (ignored otherwise).
    case_docs:
        The current case's gold contexts for the ``"oracle"`` condition (perfect
        retrieval); ignored otherwise.
    url, timeout:
        Perspicacité MCP endpoint settings (used only for ``"perspicacite"``).

    Raises
    ------
    ValueError
        If ``condition`` is not a recognized ablation condition.
    """
    if condition == "null":
        return NullBackend()
    if condition == "tfidf":
        return TfidfBackend(pool or [])
    if condition == "oracle":
        return OracleBackend(case_docs or [])
    if condition == "perspicacite":
        from consilium.sources.perspicacite import PerspicaciteBackend

        return PerspicaciteBackend(url, timeout)
    raise ValueError(f"unknown eval retrieval condition: {condition!r}")
