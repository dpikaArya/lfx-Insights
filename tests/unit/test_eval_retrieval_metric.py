"""Golden-value tests for the intrinsic retrieval metric (recall@k / nDCG@k)."""

from __future__ import annotations

import math

import pytest

from lfx_insights.eval.metrics.retrieval import (
    ndcg_at_k,
    recall_at_k,
    relevance_flags,
    score_retrieval,
)

pytestmark = pytest.mark.unit


def test_relevance_flags_match_by_any_key() -> None:
    keys = [{"a", "alpha"}, {"b"}, {"c"}]
    assert relevance_flags(keys, {"alpha", "c"}) == [True, False, True]


def test_recall_at_k() -> None:
    rels = [True, False, True]
    assert recall_at_k(rels, n_gold=2, k=3) == 1.0
    assert recall_at_k(rels, n_gold=2, k=1) == 0.5  # only first (relevant) counted
    assert recall_at_k([], n_gold=0, k=3) == 0.0  # no gold -> 0, no crash


def test_ndcg_at_k_golden() -> None:
    # rel=[T,F,T], 2 gold: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5;
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093; nDCG = 0.91973.
    rels = [True, False, True]
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(rels, n_gold=2, k=3) == pytest.approx(expected)
    # Perfect ranking -> 1.0.
    assert ndcg_at_k([True, True], n_gold=2, k=2) == pytest.approx(1.0)
    # No gold -> 0.0.
    assert ndcg_at_k([True], n_gold=0, k=3) == 0.0


def test_score_retrieval_assembles_scores() -> None:
    score = score_retrieval([{"a"}, {"x"}, {"c"}], {"a", "c"}, k=3)
    assert score.recall == 1.0
    assert score.ndcg == pytest.approx(1.5 / (1.0 + 1.0 / math.log2(3)))
    assert score.k == 3 and score.n_gold == 2 and score.n_retrieved == 3


def test_score_retrieval_misses() -> None:
    score = score_retrieval([{"x"}, {"y"}], {"a", "b"}, k=2)
    assert score.recall == 0.0 and score.ndcg == 0.0
