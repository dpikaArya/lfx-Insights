from __future__ import annotations

import pytest

from consilium.lifescience.statistics import post_hoc_power, recommend_sample_size
from consilium.models import StatRecommendation

pytestmark = pytest.mark.unit


# --- recommend_sample_size: golden values --------------------------------------------------


def test_two_sample_t_classic_cohen_value() -> None:
    # Cohen's classic: medium effect d=0.5, alpha=0.05, power=0.80 -> 64 per group.
    rec = recommend_sample_size("two_sample_t", 0.5)
    assert isinstance(rec, StatRecommendation)
    assert rec.n_per_group == 64
    assert rec.total_n == 128
    assert rec.method == "two-sample t-test"
    assert rec.stato_term is not None
    assert rec.effect_size == 0.5
    assert rec.alpha == 0.05
    assert rec.power == 0.80


def test_correlation_sample_size_golden() -> None:
    # Fisher z: r=0.3, alpha=0.05, power=0.80 -> n == 85 (±1 acceptable).
    rec = recommend_sample_size("correlation", 0.3)
    assert rec.total_n is not None
    assert 84 <= rec.total_n <= 85
    assert rec.n_per_group == rec.total_n


def test_paired_t_returns_total_n() -> None:
    rec = recommend_sample_size("paired_t", 0.5)
    assert rec.total_n == 34
    assert rec.n_per_group == 34  # paired: total == pairs
    assert rec.method == "paired t-test"


def test_one_way_anova_sensible_per_group() -> None:
    rec = recommend_sample_size("one_way_anova", 0.25, groups=4)
    assert rec.total_n is not None and rec.total_n > 0
    assert rec.n_per_group is not None and rec.n_per_group > 0
    # 4 groups balanced -> per group * groups covers the total.
    assert rec.n_per_group * 4 >= rec.total_n
    assert rec.method == "one-way ANOVA"


def test_two_proportions_uses_cohen_h() -> None:
    rec = recommend_sample_size("two_proportions", 0.5)
    assert rec.n_per_group == 63
    assert rec.total_n == 126
    assert rec.method == "two-proportion z-test"


def test_alpha_and_power_propagate() -> None:
    rec = recommend_sample_size("two_sample_t", 0.5, alpha=0.01, power=0.90)
    assert rec.alpha == 0.01
    assert rec.power == 0.90
    # Stricter alpha and higher power demand a larger sample than the classic 64.
    assert rec.n_per_group is not None and rec.n_per_group > 64


# --- post_hoc_power ------------------------------------------------------------------------


def test_post_hoc_power_two_sample_matches_target() -> None:
    p = post_hoc_power("two_sample_t", 64, 0.5)
    assert p == pytest.approx(0.80, abs=0.02)


def test_post_hoc_power_correlation_matches_target() -> None:
    p = post_hoc_power("correlation", 85, 0.3)
    assert p == pytest.approx(0.80, abs=0.02)


def test_post_hoc_power_two_proportions() -> None:
    p = post_hoc_power("two_proportions", 64, 0.5)
    assert p == pytest.approx(0.81, abs=0.02)


def test_post_hoc_power_anova_total_n() -> None:
    p = post_hoc_power("one_way_anova", 180, 0.25, groups=4)
    assert p == pytest.approx(0.80, abs=0.02)


def test_post_hoc_power_monotonic_in_n() -> None:
    low = post_hoc_power("two_sample_t", 20, 0.5)
    high = post_hoc_power("two_sample_t", 100, 0.5)
    assert high > low


# --- domain guards & errors ----------------------------------------------------------------


def test_correlation_r_zero_raises() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recommend_sample_size("correlation", 0.0)


def test_correlation_r_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"\|r\| < 1"):
        recommend_sample_size("correlation", 1.2)


def test_correlation_post_hoc_r_zero_raises() -> None:
    with pytest.raises(ValueError, match="undefined"):
        post_hoc_power("correlation", 85, 0.0)


def test_correlation_post_hoc_n_too_small_raises() -> None:
    # n < 4 makes sqrt(n - 3) undefined; expect an explicit ValueError, not a
    # bare math domain error.
    for n in (1, 2, 3):
        with pytest.raises(ValueError, match=r"correlation post-hoc power requires n >= 4"):
            post_hoc_power("correlation", n, 0.3)


def test_unknown_design_raises() -> None:
    with pytest.raises(ValueError, match="Unknown design"):
        recommend_sample_size("nonparametric_magic", 0.5)


def test_unknown_design_post_hoc_raises() -> None:
    with pytest.raises(ValueError, match="Unknown design"):
        post_hoc_power("nonparametric_magic", 30, 0.5)


def test_bad_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        recommend_sample_size("two_sample_t", 0.5, alpha=1.5)


def test_bad_power_raises() -> None:
    with pytest.raises(ValueError, match="power"):
        recommend_sample_size("two_sample_t", 0.5, power=0.0)


def test_anova_too_few_groups_raises() -> None:
    with pytest.raises(ValueError, match="groups >= 2"):
        recommend_sample_size("one_way_anova", 0.25, groups=1)


def test_post_hoc_n_too_small_raises() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        post_hoc_power("two_sample_t", 0, 0.5)


@pytest.mark.parametrize(
    "design,n,effect,kwargs",
    [
        ("paired_t", 34, 0.5, {}),
        ("one_way_anova", 180, 0.25, {"groups": 4}),
        ("two_proportions", 64, 0.5, {}),
    ],
)
def test_post_hoc_power_across_designs(design: str, n: int, effect: float, kwargs: dict) -> None:
    power = post_hoc_power(design, n, effect, **kwargs)
    assert 0.0 <= power <= 1.0
    assert power > 0.5  # these (n, effect) pairs are adequately powered


def test_post_hoc_anova_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="groups"):
        post_hoc_power("one_way_anova", 100, 0.25, groups=1)
