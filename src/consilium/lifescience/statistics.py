"""Deterministic statistical power / sample-size advisory.

Correctness is paramount: wrong sample-size or power advice is the most dangerous
failure mode of this module. We therefore delegate every numeric computation to
``statsmodels.stats.power`` (t/F/normal power solvers) and ``scipy.stats`` (Fisher
z transform for correlations) and never hand-roll z-tables or hardcode critical
values for a single (alpha, power) pair.

Every recommendation names its method and a best-effort STATO term so the advice
is auditable.
"""

from __future__ import annotations

import math

from scipy.stats import norm
from statsmodels.stats.power import (
    FTestAnovaPower,
    NormalIndPower,
    TTestIndPower,
    TTestPower,
)

from consilium.models import StatRecommendation

# Human-readable method label + best-effort STATO/OBI term per supported design.
# STATO terms are the closest standardized labels for the named test.
_DESIGN_LABELS: dict[str, tuple[str, str]] = {
    "two_sample_t": ("two-sample t-test", "STATO:0000304"),
    "paired_t": ("paired t-test", "STATO:0000305"),
    "one_way_anova": ("one-way ANOVA", "STATO:0000208"),
    "two_proportions": ("two-proportion z-test", "STATO:0000345"),
    "correlation": ("Pearson correlation (Fisher z)", "STATO:0000142"),
}


def _check_design(design: str) -> None:
    if design not in _DESIGN_LABELS:
        supported = ", ".join(sorted(_DESIGN_LABELS))
        raise ValueError(f"Unknown design {design!r}; supported designs are: {supported}.")


def _check_common(alpha: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1); got {power}.")


def _check_correlation_r(r: float) -> None:
    if r == 0.0:
        raise ValueError("correlation effect_size r == 0 is undefined (Fisher z diverges).")
    if abs(r) >= 1.0:
        raise ValueError(f"correlation effect_size r must satisfy |r| < 1; got {r}.")


def recommend_sample_size(
    design: str,
    effect_size: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    groups: int = 2,
) -> StatRecommendation:
    """Recommend a sample size for a target power.

    Args:
        design: one of the supported designs (see module docstring).
        effect_size: the design's natural effect size — Cohen's d (t-tests),
            Cohen's f (one-way ANOVA), Cohen's h (two proportions), or the
            correlation coefficient r (correlation).
        alpha: two-sided significance level.
        power: target statistical power.
        groups: number of groups (one-way ANOVA only).

    Returns:
        A :class:`StatRecommendation` with method, STATO term, and the solved
        ``n_per_group`` / ``total_n``.

    Raises:
        ValueError: for an unknown design, out-of-range alpha/power/groups, or a
            domain-invalid correlation effect size.
    """
    _check_design(design)
    _check_common(alpha, power)
    method, stato = _DESIGN_LABELS[design]

    n_per_group: int
    total_n: int
    notes = ""

    if design == "two_sample_t":
        raw = TTestIndPower().solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
        )
        n_per_group = math.ceil(float(raw))
        total_n = 2 * n_per_group
        notes = "n is per group (equal allocation, two-sided)."
    elif design == "paired_t":
        raw = TTestPower().solve_power(effect_size=effect_size, alpha=alpha, power=power)
        total_n = math.ceil(float(raw))
        n_per_group = total_n
        notes = "total_n is the number of paired observations."
    elif design == "one_way_anova":
        if groups < 2:
            raise ValueError(f"one_way_anova requires groups >= 2; got {groups}.")
        raw = FTestAnovaPower().solve_power(
            effect_size=effect_size, alpha=alpha, power=power, k_groups=groups
        )
        total_n = math.ceil(float(raw))
        n_per_group = math.ceil(total_n / groups)
        notes = f"total_n across {groups} groups; n_per_group balances the design."
    elif design == "two_proportions":
        raw = NormalIndPower().solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
        )
        n_per_group = math.ceil(float(raw))
        total_n = 2 * n_per_group
        notes = "effect_size is Cohen's h; n is per group (two-sided)."
    else:  # correlation
        _check_correlation_r(effect_size)
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)
        total_n = math.ceil(((z_alpha + z_beta) / math.atanh(effect_size)) ** 2 + 3)
        n_per_group = total_n
        notes = "Fisher z approximation; total_n is the number of paired observations."

    return StatRecommendation(
        design=design,
        method=method,
        stato_term=stato,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        n_per_group=n_per_group,
        total_n=total_n,
        notes=notes,
    )


def post_hoc_power(
    design: str,
    n: int,
    effect_size: float,
    *,
    alpha: float = 0.05,
    groups: int = 2,
) -> float:
    """Compute the achieved (post-hoc) power for an observed sample size.

    ``n`` is interpreted per the design's natural unit: per-group n for
    ``two_sample_t`` and ``two_proportions``; total observations for ``paired_t``,
    ``one_way_anova`` (across all groups), and ``correlation``.

    Raises:
        ValueError: for an unknown design, out-of-range alpha/groups, n < 1, or a
            domain-invalid correlation effect size.
    """
    _check_design(design)
    _check_common(alpha)
    if n < 1:
        raise ValueError(f"n must be a positive integer; got {n}.")

    if design == "two_sample_t":
        result = TTestIndPower().solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0
        )
    elif design == "paired_t":
        result = TTestPower().solve_power(effect_size=effect_size, nobs=n, alpha=alpha)
    elif design == "one_way_anova":
        if groups < 2:
            raise ValueError(f"one_way_anova requires groups >= 2; got {groups}.")
        result = FTestAnovaPower().solve_power(
            effect_size=effect_size, nobs=n, alpha=alpha, k_groups=groups
        )
    elif design == "two_proportions":
        result = NormalIndPower().solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0
        )
    else:  # correlation
        _check_correlation_r(effect_size)
        if n < 4:
            raise ValueError("correlation post-hoc power requires n >= 4")
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z = math.atanh(effect_size) * math.sqrt(n - 3)
        result = float(norm.cdf(z - z_alpha) + norm.cdf(-z - z_alpha))

    return float(result)
