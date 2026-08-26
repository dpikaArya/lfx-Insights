"""LLM-judge answer-quality metric (approx. ScholarQABench's Prometheus rubric).

ScholarQABench (Asai et al. 2024, arXiv:2411.14199) scores long-form answers with
an ensemble of fine-tuned Prometheus 8x7B judges on three rubric aspects:

- **organization & coherence** â€” is the answer well structured and internally consistent;
- **coverage & amount of information** â€” does it address the question thoroughly;
- **relevance & focus** â€” does it stay on-topic without padding.

This module is an *approximation*, not a bit-identical reproduction of those 8x7B
judges. We send ONE prompt to whatever :class:`~lfx_insights.llm.client.LLMClient` is
configured (a general instruction-following model, or a deterministic mock offline)
and ask for a 1-5 integer per aspect. The wording, the model, and the score
distribution will differ from the released Prometheus checkpoints, so results are
directionally comparable â€” not numerically interchangeable â€” with the paper.

**Reference-guided scoring.** When a ``reference`` answer is supplied it is given to
the judge as the implicit "Score 5" anchor (the rubric is graded relative to it),
matching ScholarQABench's reference-guided setup. When ``reference`` is ``None`` the
judge grades the answer on its own merits.

**Normalization.** ScholarQABench reports the raw 1-5 rubric scores divided by 5
(so a perfect answer reads as 1.0). We follow that convention: each aspect is clamped
to the valid ``[1, 5]`` range and divided by 5, giving ``5 -> 1.0`` and ``1 -> 0.2``.
We deliberately do *not* use ``(x - 1) / 4`` (which would map the 1-5 floor to 0.0):
the goal here is to mirror the paper's ``/5`` reporting, where the worst legal score
still carries a non-zero floor of 0.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lfx_insights.eval.models import QualityScore

if TYPE_CHECKING:
    from lfx_insights.eval.models import GeneratedAnswer
    from lfx_insights.llm.client import LLMClient

# Rubric bounds and the divisor used for normalization (see module docstring).
_MIN_SCORE = 1
_MAX_SCORE = 5
_DIVISOR = 5.0


class QualityRubric(BaseModel):
    """The judge's raw 1-5 integer score per Prometheus aspect."""

    organization: int = Field(description="Organization & coherence, 1 (poor) to 5 (excellent).")
    coverage: int = Field(description="Coverage & amount of info, 1 (poor) to 5 (excellent).")
    relevance: int = Field(description="Relevance & focus, 1 (poor) to 5 (excellent).")


def _clamp(value: int) -> int:
    """Clamp a possibly out-of-range model score into the valid ``[1, 5]`` band."""
    return max(_MIN_SCORE, min(_MAX_SCORE, value))


def _normalize(value: int) -> float:
    """Clamp to ``[1, 5]`` then divide by 5 (``5 -> 1.0``, ``1 -> 0.2``)."""
    return _clamp(value) / _DIVISOR


def _build_prompt(answer_text: str, reference: str | None) -> str:
    """Compose the single 3-aspect rubric prompt (reference-guided if available)."""
    rubric = (
        "Score the ANSWER on a 1-5 integer scale for each of three aspects, where "
        "1 is poor and 5 is excellent:\n"
        "- organization: organization & coherence (clear structure, internally consistent).\n"
        "- coverage: coverage & amount of information (thoroughly addresses the topic).\n"
        "- relevance: relevance & focus (on-topic, no padding or digressions)."
    )
    if reference is not None:
        anchor = (
            "A REFERENCE answer is provided as the gold standard; treat an answer that "
            "matches its quality as a 5 on every aspect.\n\n"
            f"REFERENCE answer (Score 5 anchor):\n{reference}\n\n"
        )
    else:
        anchor = ""
    return (
        "You are a strict scientific-writing judge approximating the ScholarQABench "
        "Prometheus rubric.\n\n"
        f"{rubric}\n\n"
        f"{anchor}"
        f"ANSWER to score:\n{answer_text}\n\n"
        "Return one integer (1-5) per aspect."
    )


def judge_quality(answer: GeneratedAnswer, reference: str | None, llm: LLMClient) -> QualityScore:
    """Score an answer's quality with an LLM judge, approximating Prometheus.

    Sends a single 1-5-per-aspect rubric prompt to ``llm`` (reference-guided when
    ``reference`` is not ``None``) and normalizes each returned score to ``[0, 1]``
    by clamping to ``[1, 5]`` and dividing by 5 (so ``5 -> 1.0``, ``1 -> 0.2``).

    Args:
        answer: The generated long-form answer; only its text is judged.
        reference: Optional gold answer used as the "Score 5" anchor.
        llm: Any structured-output client (real or :class:`MockLLM`).

    Returns:
        A :class:`~lfx_insights.eval.models.QualityScore` with ``judge="llm"``.
    """
    prompt = _build_prompt(answer.text, reference)
    rubric = llm.complete_structured(prompt, QualityRubric)
    return QualityScore(
        organization=_normalize(rubric.organization),
        coverage=_normalize(rubric.coverage),
        relevance=_normalize(rubric.relevance),
        judge="llm",
    )
