"""Pluggable entailment judges for the citation faithfulness metric.

The citation metric asks, for each cited sentence, whether the cited passage(s)
*support* (entail) the sentence. That judgement is delegated to an
:class:`Entailer`: a small, swappable strategy so the harness can trade speed
for fidelity. Three implementations are provided:

- :class:`LexicalEntailer` — fast, deterministic token-overlap heuristic.
- :class:`GroundingEntailer` — reuses indicium's verbatim grounding gate
  (verbatim/near-verbatim support only; no paraphrase).
- :class:`LLMEntailer` — a single structured NLI-style LLM call (highest
  fidelity, requires a provider).

Use :func:`build_entailer` to construct one by name.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from consilium.standards.grounding import verify_quote_in

if TYPE_CHECKING:
    from consilium.llm.client import LLMClient

# A small, deliberately conservative English stopword set. Kept local so the
# lexical judge has no external/data dependency and is fully deterministic.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "has",
        "have",
        "her",
        "his",
        "its",
        "our",
        "out",
        "was",
        "were",
        "with",
        "that",
        "this",
        "from",
        "they",
        "them",
        "then",
        "than",
        "into",
        "such",
        "been",
        "their",
        "which",
        "while",
        "would",
        "could",
        "should",
        "about",
        "these",
        "those",
        "there",
        "here",
        "when",
        "what",
        "your",
    }
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, dropping stopwords and tokens of len < 3."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) >= 3 and t not in _STOPWORDS]


@runtime_checkable
class Entailer(Protocol):
    """A strategy deciding whether a premise supports (entails) a hypothesis."""

    name: str

    def entails(self, premise: str, hypothesis: str) -> bool:
        """Return True iff ``premise`` supports the claim in ``hypothesis``."""
        ...


class LexicalEntailer:
    """Token-overlap heuristic: cheap, deterministic, no model required.

    Entails iff the fraction of the hypothesis's content tokens that also appear
    in the premise's token set is at least ``threshold``. An empty hypothesis
    (no content tokens) never entails.
    """

    name = "lexical"

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    def entails(self, premise: str, hypothesis: str) -> bool:
        """Return True iff hypothesis content-token coverage by premise >= threshold."""
        hyp_tokens = _content_tokens(hypothesis)
        if not hyp_tokens:
            return False
        premise_tokens = set(_content_tokens(premise))
        covered = sum(1 for t in hyp_tokens if t in premise_tokens)
        return covered / len(hyp_tokens) >= self.threshold


class GroundingEntailer:
    """Verbatim grounding gate: reuses indicium's ``verify_quote`` kernel.

    Entails iff the hypothesis is verbatim (or near-verbatim) present in the
    premise. Strict by design — paraphrase does not count as support.
    """

    name = "grounding"

    def entails(self, premise: str, hypothesis: str) -> bool:
        """Return True iff hypothesis is grounded verbatim in premise."""
        return verify_quote_in(hypothesis, [premise])


class EntailVerdict(BaseModel):
    """Structured NLI verdict returned by the LLM judge."""

    entailed: bool


class LLMEntailer:
    """LLM-backed NLI judge: one structured call per (premise, hypothesis) pair."""

    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def entails(self, premise: str, hypothesis: str) -> bool:
        """Return True iff the LLM judges the premise to entail the claim."""
        prompt = (
            "You are a strict natural-language-inference judge.\n"
            "Decide whether the PREMISE supports (entails) the CLAIM. The premise "
            "entails the claim only if a reader of the premise could conclude the "
            "claim is true; otherwise it does not.\n\n"
            f"PREMISE:\n{premise}\n\n"
            f"CLAIM:\n{hypothesis}\n\n"
            "Return entailed=true if the premise supports the claim, else entailed=false."
        )
        verdict = self.llm.complete_structured(prompt, EntailVerdict)
        return verdict.entailed


def build_entailer(
    judge: str, *, threshold: float = 0.55, llm: LLMClient | None = None
) -> Entailer:
    """Construct an :class:`Entailer` by name.

    Args:
        judge: One of ``"lexical"``, ``"grounding"``, ``"llm"``.
        threshold: Coverage threshold for the lexical judge.
        llm: Required when ``judge == "llm"``; ignored otherwise.

    Raises:
        ValueError: For an unknown judge, or for ``"llm"`` without an ``llm``.
    """
    if judge == "lexical":
        return LexicalEntailer(threshold)
    if judge == "grounding":
        return GroundingEntailer()
    if judge == "llm":
        if llm is None:
            raise ValueError("LLMEntailer requires an llm client (judge='llm')")
        return LLMEntailer(llm)
    raise ValueError(f"Unknown entailer judge: {judge!r}")
