"""Intrinsic retrieval metrics (LitSearch-style): did the retriever fetch the gold papers.

ScholarQABench scores retrieval only *extrinsically* (through downstream citation/answer
quality). LitSearch (Ajith et al., EMNLP 2024) instead gives each query a set of gold
relevant corpus papers and reports recall@k. This module adds that intrinsic measure â€”
recall@k and nDCG@k under binary relevance â€” so a retrieval condition (especially
PerspicacitÃ©) can be scored directly, not just through the answer it enables.

Matching is identifier-agnostic: a retrieved paper is relevant if any of its keys (corpus
id, DOI, normalised title) intersects the gold set, so it works across corpora/embedders
that expose different identifiers.
"""

from __future__ import annotations

import math

from lfx_insights.eval.models import RetrievalScore


def relevance_flags(retrieved_keys: list[set[str]], gold: set[str]) -> list[bool]:
    """Per-rank relevance: a retrieved paper is relevant if any of its keys is gold."""
    return [bool(keys & gold) for keys in retrieved_keys]


def recall_at_k(relevances: list[bool], n_gold: int, k: int) -> float:
    """Fraction of gold papers found in the top-``k`` (0.0 when there is no gold)."""
    if n_gold <= 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r)
    return min(1.0, hits / n_gold)


def ndcg_at_k(relevances: list[bool], n_gold: int, k: int) -> float:
    """nDCG@k under binary relevance; ideal ranking puts all gold hits first."""
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(relevances[:k]) if r)
    ideal_hits = min(n_gold, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def score_retrieval(retrieved_keys: list[set[str]], gold: set[str], k: int) -> RetrievalScore:
    """Score a rank-ordered retrieval against the gold set.

    ``retrieved_keys`` is the retrieved papers in rank order, each as the set of its
    identifiers (id/DOI/normalised title); ``gold`` is the gold identifier set.
    """
    rels = relevance_flags(retrieved_keys, gold)
    n_gold = len(gold)
    return RetrievalScore(
        recall=recall_at_k(rels, n_gold, k),
        ndcg=ndcg_at_k(rels, n_gold, k),
        k=k,
        n_gold=n_gold,
        n_retrieved=len(retrieved_keys),
    )
