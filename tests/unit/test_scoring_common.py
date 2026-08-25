from __future__ import annotations

import pytest

from consilium.models import ScoreComponent
from consilium.scoring.common import (
    band,
    cosine,
    make_score,
    minmax_normalize,
    sample_uncertainty,
    weighted_mean,
)

pytestmark = pytest.mark.unit


def test_band_thresholds() -> None:
    assert band(0.95) == "very high"
    assert band(0.7) == "high"
    assert band(0.5) == "moderate"
    assert band(0.25) == "low"
    assert band(0.0) == "very low"


def test_minmax_neutral_without_spread() -> None:
    assert minmax_normalize([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert minmax_normalize([0.0, 10.0]) == [0.0, 1.0]
    assert minmax_normalize([]) == []


def test_weighted_mean_and_make_score() -> None:
    comps = [
        ScoreComponent(name="a", value=1.0, weight=1.0),
        ScoreComponent(name="b", value=0.0, weight=1.0),
    ]
    assert weighted_mean(comps) == 0.5
    s = make_score(comps, uncertainty=0.2)
    assert s.value == 0.5
    assert s.interpretation == "moderate"
    assert s.uncertainty == 0.2
    assert len(s.components) == 2


def test_sample_uncertainty_shrinks() -> None:
    assert sample_uncertainty(0) == 1.0
    assert sample_uncertainty(1) == 1.0
    assert sample_uncertainty(100) < sample_uncertainty(4)


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
