"""Reference-based correctness metrics (deterministic, no LLM).

Faithful re-implementations of ScholarQABench's reference-based scorers
(Asai et al. 2024, arXiv:2411.14199):

* :func:`match_score` — ScholarQABench ``compute_match``: 1.0 if the normalized
  gold string is a substring of the normalized, citation-stripped answer.
* :func:`rouge_l` — ScholarQABench ``compute_rouge``: ROUGE-L F-measure over
  lowercased whitespace tokens, computed via a longest-common-subsequence (LCS)
  dynamic program. No external ROUGE dependency.

Both are case-insensitive and operate on whitespace-normalized text, so they are
robust to inline ``[n]`` citation markers and incidental spacing differences.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[\d+\]")
_WS = re.compile(r"\s+")


def remove_citations(text: str) -> str:
    """Strip inline ``[n]`` citation markers and collapse whitespace.

    Removes every ``[<digits>]`` marker (the scheme inline answers use to point at
    numbered docs) and normalizes runs of whitespace to single spaces, trimming the
    ends. This mirrors ScholarQABench's pre-processing so that citation markers never
    affect substring or token-overlap correctness.
    """
    return _WS.sub(" ", _CITATION.sub("", text)).strip()


def match_score(answer_text: str, gold: str) -> float:
    """Substring match score (ScholarQABench ``compute_match``).

    Returns 1.0 when the normalized gold answer (lowercased, stripped) appears as a
    substring of the normalized, citation-stripped answer text, else 0.0. An empty
    gold answer scores 0.0 (there is nothing to match against).
    """
    gold_norm = gold.lower().strip()
    if not gold_norm:
        return 0.0
    answer_norm = remove_citations(answer_text).lower().strip()
    return 1.0 if gold_norm in answer_norm else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token sequences.

    Standard O(n*m) dynamic program with an O(min(n, m)) rolling row.
    """
    if not a or not b:
        return 0
    # Keep the shorter sequence on the inner axis to bound memory at O(min(n, m)).
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (ScholarQABench ``compute_rouge``).

    Tokenizes both sides on whitespace after lowercasing, computes the LCS length
    ``L``, then ``precision = L / |pred|``, ``recall = L / |ref|`` and the harmonic
    F-measure ``F = 2*P*R / (P + R)``. Returns 0.0 if either side has no tokens.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
