"""Unit tests for the eval retrieval-ablation backends.

Offline by design: the Perspicacité backend is only constructed (never called), so
no network access occurs.
"""

from __future__ import annotations

import pytest

from consilium.eval.retrieval import (
    NullBackend,
    TfidfBackend,
    build_eval_backend,
)
from consilium.models import Paper

pytestmark = pytest.mark.unit


def _pool() -> list[Paper]:
    """A small pool of papers with deliberately distinct topics."""
    return [
        Paper(
            id="astro",
            title="Galaxy formation in the early universe",
            abstract="We study dark matter halos, stellar nucleosynthesis, and cosmic redshift.",
        ),
        Paper(
            id="micro",
            title="Gut microbiome and host metabolism",
            abstract="Bacterial taxa in the intestine modulate short-chain fatty acid production.",
        ),
        Paper(
            id="ml",
            title="Transformer attention for language modeling",
            abstract="Self-attention layers and gradient descent train deep neural networks.",
        ),
        Paper(
            id="chem",
            title="Catalysis of organic reactions",
            abstract="Transition-metal catalysts accelerate carbon-carbon bond formation.",
        ),
    ]


def test_null_backend_returns_empty() -> None:
    backend = NullBackend()
    corpus = backend.build_or_select_kb("anything", max_papers=5)
    assert corpus.kb_id == "null"
    assert corpus.papers == []
    assert backend.relevant_passages("anything", "null", k=10) == []
    assert backend.paper_content("astro") == ""


def test_tfidf_ranks_on_topic_paper_first() -> None:
    backend = TfidfBackend(_pool())

    corpus = backend.build_or_select_kb("dark matter halos and cosmic redshift")
    assert corpus.kb_id == "tfidf"
    assert corpus.papers[0].id == "astro"

    passages = backend.relevant_passages("bacterial taxa in the intestine", "tfidf", k=4)
    assert passages[0].paper_id == "micro"
    assert passages[0].location == "abstract"
    # The abstract is used as the passage text when present.
    assert "intestine" in passages[0].text


def test_tfidf_respects_max_papers_and_k() -> None:
    backend = TfidfBackend(_pool())

    corpus = backend.build_or_select_kb("neural networks and attention", max_papers=2)
    assert len(corpus.papers) == 2
    assert corpus.papers[0].id == "ml"

    passages = backend.relevant_passages("transition-metal catalysts", "tfidf", k=1)
    assert len(passages) == 1
    assert passages[0].paper_id == "chem"


def test_tfidf_paper_content_falls_back_to_empty() -> None:
    backend = TfidfBackend(_pool())
    assert "Galaxy formation" in backend.paper_content("astro")
    assert backend.paper_content("does-not-exist") == ""


def test_tfidf_empty_pool_no_crash() -> None:
    backend = TfidfBackend([])
    assert backend.build_or_select_kb("anything").papers == []
    assert backend.relevant_passages("anything", "tfidf", k=5) == []
    assert backend.paper_content("anything") == ""


def test_tfidf_empty_query_no_results() -> None:
    backend = TfidfBackend(_pool())
    assert backend.relevant_passages("   ", "tfidf", k=5) == []


def test_build_eval_backend_routing() -> None:
    null = build_eval_backend("null")
    assert isinstance(null, NullBackend)

    tfidf = build_eval_backend("tfidf", pool=_pool())
    assert isinstance(tfidf, TfidfBackend)
    assert tfidf.build_or_select_kb("dark matter").papers[0].id == "astro"

    # tfidf with no pool still constructs and yields empty results.
    empty_tfidf = build_eval_backend("tfidf")
    assert isinstance(empty_tfidf, TfidfBackend)
    assert empty_tfidf.build_or_select_kb("anything").papers == []


def test_build_eval_backend_perspicacite_constructs_without_network() -> None:
    # Construct only; never call its methods (which would hit the network).
    backend = build_eval_backend("perspicacite")
    assert hasattr(backend, "build_or_select_kb")
    assert hasattr(backend, "relevant_passages")
    assert hasattr(backend, "paper_content")


def test_build_eval_backend_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="unknown eval retrieval condition"):
        build_eval_backend("bm25")


def test_oracle_backend_returns_case_docs() -> None:
    from consilium.eval.retrieval import OracleBackend

    docs = [Paper(id="g1", title="Gold one", abstract="a"), Paper(id="g2", title="Gold two")]
    oracle = OracleBackend(docs)
    corpus = oracle.build_or_select_kb("anything", max_papers=10)
    assert [p.id for p in corpus.papers] == ["g1", "g2"]
    assert oracle.paper_content("g1") and oracle.paper_content("missing") == ""
    assert len(oracle.relevant_passages("q", "oracle", k=1)) == 1


def test_build_eval_backend_oracle() -> None:
    from consilium.eval.retrieval import OracleBackend

    docs = [Paper(id="g1", title="Gold", abstract="x")]
    backend = build_eval_backend("oracle", case_docs=docs)
    assert isinstance(backend, OracleBackend)
    assert [p.id for p in backend.build_or_select_kb("q").papers] == ["g1"]
