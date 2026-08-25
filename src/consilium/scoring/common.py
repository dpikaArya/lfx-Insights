"""Shared honest-scoring kernel used by all scoring modules.

These helpers enforce the project rule that a composite score always carries its
components and an interpretation band — there are no bare magic numbers.
"""

from __future__ import annotations

import math

from consilium.models import Score, ScoreComponent

# (threshold, label) in descending order.
_BANDS: list[tuple[float, str]] = [
    (0.8, "very high"),
    (0.6, "high"),
    (0.4, "moderate"),
    (0.2, "low"),
    (0.0, "very low"),
]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def band(value: float) -> str:
    """Map a 0..1 value to an interpretation band."""
    for threshold, label in _BANDS:
        if value >= threshold:
            return label
    return "very low"


def minmax_normalize(values: list[float]) -> list[float]:
    """Scale to 0..1. With no spread, returns a neutral 0.5 for each (honest:
    we do not invent rank structure where none exists).
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def weighted_mean(components: list[ScoreComponent]) -> float:
    total_w = sum(c.weight for c in components)
    if total_w == 0:
        return 0.0
    return clamp01(sum(c.value * c.weight for c in components) / total_w)


def sample_uncertainty(n: int) -> float:
    """Crude uncertainty that shrinks with sample size (1/sqrt(n), capped at 1)."""
    if n <= 0:
        return 1.0
    return clamp01(1.0 / math.sqrt(n))


def make_score(
    components: list[ScoreComponent],
    *,
    method: str = "weighted_mean",
    uncertainty: float | None = None,
) -> Score:
    """Build a Score from components (value = weighted mean, band = interpretation)."""
    value = weighted_mean(components)
    return Score(
        value=value,
        components=components,
        method=method,
        interpretation=band(value),
        uncertainty=uncertainty,
    )


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 when either is zero)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return clamp01(dot / (na * nb))
