from __future__ import annotations

import pytest
from pydantic import BaseModel

from consilium.eval.entailment import (
    Entailer,
    EntailVerdict,
    GroundingEntailer,
    LexicalEntailer,
    LLMEntailer,
    build_entailer,
)
from consilium.llm.client import MockLLM

pytestmark = pytest.mark.unit


# --- LexicalEntailer ---------------------------------------------------------


def test_lexical_full_containment_entails() -> None:
    entailer = LexicalEntailer()
    premise = "Mitochondria are the powerhouse of the cell and produce ATP energy."
    hypothesis = "Mitochondria produce ATP energy."
    assert entailer.entails(premise, hypothesis) is True


def test_lexical_disjoint_does_not_entail() -> None:
    entailer = LexicalEntailer()
    premise = "Photosynthesis converts sunlight into chemical energy in plants."
    hypothesis = "Volcanic eruptions release sulfur dioxide gas."
    assert entailer.entails(premise, hypothesis) is False


def test_lexical_partial_above_threshold() -> None:
    # 3 of 4 content tokens covered (0.75) >= 0.55 -> entails.
    entailer = LexicalEntailer(threshold=0.55)
    premise = "Aspirin reduces inflammation and pain effectively."
    hypothesis = "Aspirin reduces inflammation magnificently."
    assert entailer.entails(premise, hypothesis) is True


def test_lexical_partial_below_threshold() -> None:
    # Only 1 of 4 content tokens covered (0.25) < 0.55 -> does not entail.
    entailer = LexicalEntailer(threshold=0.55)
    premise = "Aspirin reduces inflammation and pain effectively."
    hypothesis = "Aspirin causes serious gastric bleeding."
    assert entailer.entails(premise, hypothesis) is False


def test_lexical_threshold_boundary_is_inclusive() -> None:
    # Exactly half covered; threshold 0.5 -> entails (>= is inclusive).
    entailer = LexicalEntailer(threshold=0.5)
    premise = "Caffeine boosts alertness."
    # Content tokens: {caffeine, glucose}; only "caffeine" overlaps -> 1/2 = 0.5.
    hypothesis = "Caffeine glucose."
    assert entailer.entails(premise, hypothesis) is True


def test_lexical_empty_hypothesis_does_not_entail() -> None:
    entailer = LexicalEntailer()
    assert entailer.entails("anything at all here", "") is False
    # Hypothesis with only stopwords/short tokens has no content tokens.
    assert entailer.entails("anything at all here", "the and a of") is False


def test_lexical_name() -> None:
    assert LexicalEntailer().name == "lexical"


# --- GroundingEntailer -------------------------------------------------------


def test_grounding_verbatim_substring_entails() -> None:
    entailer = GroundingEntailer()
    premise = "The treatment improved survival by 30% in the cohort study."
    hypothesis = "improved survival by 30%"
    assert entailer.entails(premise, hypothesis) is True


def test_grounding_absent_hypothesis_does_not_entail() -> None:
    entailer = GroundingEntailer()
    premise = "The treatment improved survival by 30% in the cohort study."
    hypothesis = "the drug had no measurable effect whatsoever on outcomes"
    assert entailer.entails(premise, hypothesis) is False


def test_grounding_name() -> None:
    assert GroundingEntailer().name == "grounding"


# --- LLMEntailer -------------------------------------------------------------


def _verdict_responder(entailed: bool):
    def responder(prompt: str, model: type[BaseModel]) -> BaseModel:
        return EntailVerdict(entailed=entailed)

    return responder


def test_llm_entailer_true() -> None:
    llm = MockLLM(responder=_verdict_responder(True))
    entailer = LLMEntailer(llm)
    assert entailer.entails("premise text", "claim text") is True
    assert len(llm.calls) == 1  # exactly one structured call


def test_llm_entailer_false() -> None:
    llm = MockLLM(responder=_verdict_responder(False))
    entailer = LLMEntailer(llm)
    assert entailer.entails("premise text", "claim text") is False


def test_llm_entailer_name() -> None:
    assert LLMEntailer(MockLLM()).name == "llm"


# --- build_entailer routing --------------------------------------------------


def test_build_lexical_passes_threshold() -> None:
    entailer = build_entailer("lexical", threshold=0.9)
    assert isinstance(entailer, LexicalEntailer)
    assert entailer.threshold == 0.9
    assert isinstance(entailer, Entailer)


def test_build_grounding() -> None:
    assert isinstance(build_entailer("grounding"), GroundingEntailer)


def test_build_llm() -> None:
    llm = MockLLM(responder=_verdict_responder(True))
    entailer = build_entailer("llm", llm=llm)
    assert isinstance(entailer, LLMEntailer)


def test_build_llm_without_client_raises() -> None:
    with pytest.raises(ValueError, match="requires an llm"):
        build_entailer("llm")


def test_build_unknown_judge_raises() -> None:
    with pytest.raises(ValueError, match="Unknown entailer judge"):
        build_entailer("bogus")
