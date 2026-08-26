"""Unit tests for reference-based correctness metrics."""

from __future__ import annotations

import pytest

from lfx_insights.eval.metrics.correctness import match_score, remove_citations, rouge_l

pytestmark = pytest.mark.unit


def test_remove_citations_strips_markers_and_collapses_whitespace() -> None:
    assert remove_citations("the answer is yes [3]") == "the answer is yes"
    assert remove_citations("a [1] b [22]  c") == "a b c"
    assert remove_citations("  spaced\n\nout  text  ") == "spaced out text"
    assert remove_citations("no markers here") == "no markers here"


def test_match_score_true() -> None:
    assert match_score("the answer is definitely YES.", "yes") == 1.0


def test_match_score_true_after_citation_stripping() -> None:
    # The gold token sits right next to a citation marker; stripping must expose it.
    assert match_score("the answer is yes [3]", "yes") == 1.0


def test_match_score_case_and_whitespace_insensitive() -> None:
    assert match_score("CRISPR-Cas9   is a   gene editing tool", "gene editing") == 1.0


def test_match_score_false() -> None:
    assert match_score("the answer is no", "yes") == 0.0


def test_match_score_empty_gold_is_zero() -> None:
    assert match_score("any answer at all", "") == 0.0
    assert match_score("any answer at all", "   ") == 0.0


def test_rouge_l_identical_is_one() -> None:
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_rouge_l_is_case_insensitive() -> None:
    assert rouge_l("The Cat", "the cat") == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero() -> None:
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_l_partial_overlap_golden() -> None:
    # pred  = the cat sat on the mat   (6 tokens)
    # ref   = the cat is on a mat      (6 tokens)
    # LCS   = the cat on mat           (length 4)
    # P = 4/6, R = 4/6, F = 2PR/(P+R) = 4/6 = 0.6666...
    assert rouge_l("the cat sat on the mat", "the cat is on a mat") == pytest.approx(4 / 6)


def test_rouge_l_asymmetric_lengths_golden() -> None:
    # pred  = a b c d            (4 tokens)
    # ref   = a c e              (3 tokens)
    # LCS   = a c                (length 2)
    # P = 2/4 = 0.5, R = 2/3, F = 2*0.5*(2/3)/(0.5 + 2/3) = (2/3)/(7/6) = 4/7
    assert rouge_l("a b c d", "a c e") == pytest.approx(4 / 7)


def test_rouge_l_empty_prediction_is_zero() -> None:
    assert rouge_l("", "the cat sat on the mat") == 0.0


def test_rouge_l_empty_reference_is_zero() -> None:
    assert rouge_l("the cat sat on the mat", "") == 0.0


def test_rouge_l_both_empty_is_zero() -> None:
    assert rouge_l("", "") == 0.0
