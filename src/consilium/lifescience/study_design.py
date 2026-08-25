"""Study-design recommendation per theme, driven by a transparent maturity rule.

Correctness is paramount: recommending a confirmatory design (cohort / RCT) for an
immature, sparse, or unsettled literature is a dangerous methodological error, and
so is forcing a heavily-studied, convergent field back into purely exploratory work.

The rule here is therefore *explicit and conjunctive*. Field **maturity** is a
weighted blend of THREE inputs that must hold together — we never take an OR over
them, and in particular we never equate "many papers" with "mature":

  * ``breadth``      — how much evidence exists (normalized across themes AND against
                       an absolute target, so a single tiny theme still reads as low).
  * ``recency``      — share of dated papers within a recency window of the corpus
                       max year (a stale literature is not confirmatory-ready, and a
                       literature with no dated papers — unknown recency — fails the
                       gate too: we cannot vouch for what we cannot date).
  * ``consistency``  — keyword homogeneity across the theme's papers, a proxy for
                       methodological convergence (a fragmented field is immature).

The blended maturity (0..1) is mapped onto the five-rung evidence ladder via
explicit thresholds, low -> exploratory/cross-sectional (hypothesis-generating),
high -> prospective cohort / RCT (confirmatory). Each design carries a best-effort
OBI (Ontology for Biomedical Investigations) label/term in the reasoning.

Honest by construction: every :class:`~consilium.models.Insight` carries a
:class:`~consilium.models.Score` built from explicit, weighted components via the
shared :func:`~consilium.scoring.common.make_score` kernel — never a bare number.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from consilium.corpus_features import max_year as corpus_max_year
from consilium.corpus_features import theme_papers
from consilium.models import EvidenceRef, Insight, ScoreComponent
from consilium.scoring.common import (
    clamp01,
    make_score,
    minmax_normalize,
    sample_uncertainty,
)

if TYPE_CHECKING:
    from consilium.models import Corpus, Paper, Theme

# --- Design vocabulary: rung -> (design id, best-effort OBI label/term) ---------
# OBI = Ontology for Biomedical Investigations. Terms are best-effort labels; the
# generic "study design" class is OBI:0500000. RCT has a stable term (OBI:0000471);
# the observational designs do not all have crisp leaf classes, so we anchor them
# to the parent study-design class and name the design explicitly.
_DESIGN_OBI: dict[str, str] = {
    "exploratory_observational": "OBI:0500000 (study design; exploratory observational)",
    "cross_sectional": "OBI:0500000 (study design; cross-sectional study)",
    "case_control": "OBI:0500000 (study design; case-control study)",
    "prospective_cohort": "OBI:0500000 (study design; prospective cohort study)",
    "randomized_controlled_trial": "OBI:0000471 (randomized controlled trial)",
}

# Maturity thresholds onto the five-rung ladder. Ascending; a maturity value is
# matched against the highest rung whose threshold it meets. The lowest rung has a
# 0.0 floor so every value lands somewhere.
_LADDER: list[tuple[float, str]] = [
    (0.80, "randomized_controlled_trial"),
    (0.60, "prospective_cohort"),
    (0.40, "case_control"),
    (0.20, "cross_sectional"),
    (0.00, "exploratory_observational"),
]

# Confirmatory designs require comparative groups / temporality and only become
# defensible once the field is mature.
_CONFIRMATORY: frozenset[str] = frozenset({"prospective_cohort", "randomized_controlled_trial"})
# Top confirmatory rung the recency gate may demote a stale field down to. A large,
# convergent, but STALE literature must not be recommended a confirmatory design on
# breadth+consistency alone — recency is a true gate, not a mean tie-breaker.
_RECENCY_GATE = 0.5
_GATED_DESIGN = "case_control"

# Component weights for maturity (sum = 1.0). Breadth leads (you cannot run a
# confirmatory study on almost no prior evidence), but recency and consistency are
# genuine gates, not tie-breakers.
_W_BREADTH = 0.4
_W_RECENCY = 0.3
_W_CONSISTENCY = 0.3

# Absolute evidence target: a theme at/above this many papers is "broad" on the
# absolute axis regardless of how it ranks against its siblings.
_BREADTH_TARGET = 8.0
# Cross-theme rank weight vs. absolute-saturation weight inside breadth.
_BREADTH_RANK_W = 0.5

_RECENCY_WINDOW = 3
_NEUTRAL = 0.5
_EVIDENCE_CAP = 10


def _breadth(size_rank: float, size: int) -> float:
    """Blend cross-theme rank with absolute saturation against a target.

    Using rank alone is unstable (a single theme always ranks 0.5); using the
    absolute count alone ignores the comparative landscape. We combine both so a
    tiny theme reads low whether judged against siblings or against the target.
    """
    saturation = clamp01(size / _BREADTH_TARGET)
    return clamp01(_BREADTH_RANK_W * size_rank + (1.0 - _BREADTH_RANK_W) * saturation)


def _recency(papers: list[Paper], max_year: int | None) -> float:
    """Share of *dated* papers within ``_RECENCY_WINDOW`` years of the corpus max.

    Undated papers are excluded from the denominator (we do not punish or reward
    missing years). With no dated papers the recency *signal* is absent, so the
    score component reads neutral (0.5) rather than inventing a value; the
    recency gate (see :func:`_design_for`) treats that unknown as *failing*, so an
    entirely-undated theme can never reach a confirmatory rung.
    """
    years = [p.year for p in papers if p.year is not None]
    if not years or max_year is None:
        return _NEUTRAL
    cutoff = max_year - _RECENCY_WINDOW
    recent = sum(1 for y in years if y >= cutoff)
    return clamp01(recent / len(years))


def _recency_known(papers: list[Paper], max_year: int | None) -> bool:
    """Whether the recency signal exists (at least one dated paper + corpus max)."""
    return max_year is not None and any(p.year is not None for p in papers)


def _consistency(theme: Theme, papers: list[Paper]) -> float:
    """Keyword homogeneity: mean share of theme keywords present per paper.

    A proxy for methodological/topical convergence. With no keywords or no papers
    there is nothing to read, so we return a neutral 0.5 rather than inventing
    convergence.
    """
    keywords = [kw.lower() for kw in theme.keywords if kw]
    if not keywords or not papers:
        return _NEUTRAL
    # Word-boundary matching (not substring): a keyword embedded inside a larger
    # word (e.g. "meta" inside "metabolism") must not count as present.
    patterns = [re.compile(rf"\b{re.escape(kw)}\b") for kw in keywords]
    shares: list[float] = []
    for paper in papers:
        text = paper.text().lower()
        present = sum(1 for pat in patterns if pat.search(text))
        shares.append(present / len(keywords))
    return clamp01(sum(shares) / len(shares))


def _design_for(maturity: float, recency: float, recency_known: bool) -> tuple[str, bool]:
    """Map maturity to a design, with a hard recency gate on confirmatory rungs.

    The weighted-mean maturity sets the un-gated rung, but a stale field (recency
    below the gate) *or one whose recency is unknown* (no dated papers) is *demoted*
    off any confirmatory rung down to ``case_control``. This keeps the rule
    conjunctive at the dangerous boundary: high breadth + high consistency can never,
    on their own, buy a confirmatory recommendation, and an undated corpus — whose
    recency we cannot vouch for — must not read as confirmatory-ready either.

    Returns the ``(design, was_demoted)`` pair, where ``was_demoted`` is true only
    when the un-gated ladder design was confirmatory and the recency gate forced the
    downgrade to ``case_control`` (so callers can report a real demotion, never a
    spurious one for themes that simply landed on ``case_control``).
    """
    design = "exploratory_observational"
    for threshold, candidate in _LADDER:
        if maturity >= threshold:
            design = candidate
            break
    gate_fails = (not recency_known) or recency < _RECENCY_GATE
    if design in _CONFIRMATORY and gate_fails:
        return _GATED_DESIGN, True
    return design, False


def _reasoning(
    label: str,
    design: str,
    components: list[ScoreComponent],
    maturity: float,
    band: str,
    was_demoted: bool,
    recency_known: bool,
) -> str:
    obi = _DESIGN_OBI[design]
    parts = {c.name: c.value for c in components}
    arm = "confirmatory" if design in _CONFIRMATORY else "exploratory / hypothesis-generating"
    gate_note = ""
    if was_demoted:
        if recency_known:
            cause = f"Recency ({parts['recency']:.2f}) is below the gate ({_RECENCY_GATE})"
        else:
            cause = "Recency is unknown (no dated papers), which fails the gate"
        gate_note = (
            f" {cause}; a literature that is stale or of unverifiable recency is demoted "
            f"off any confirmatory rung even when breadth and consistency are high (the "
            f"rule is conjunctive, not an OR)."
        )
    return (
        f"Maturity for theme '{label}' is {band} ({maturity:.2f}), blended (weighted "
        f"mean, NOT an OR) from breadth={parts['breadth']:.2f} (w={_W_BREADTH}), "
        f"recency={parts['recency']:.2f} (w={_W_RECENCY}), and consistency="
        f"{parts['consistency']:.2f} (w={_W_CONSISTENCY}). Low maturity maps to "
        f"exploratory/cross-sectional designs and high maturity to confirmatory "
        f"cohort/RCT designs; this theme falls in the {arm} band, so the recommended "
        f"design is {design} [{obi}].{gate_note}"
    )


def recommend_designs(themes: list[Theme], corpus: Corpus) -> list[Insight]:
    """Recommend a study design for each theme from its field maturity.

    Maturity is a weighted mean (never an OR) of breadth (normalized paper count,
    rank + absolute), recency (share of recent dated papers), and consistency
    (keyword homogeneity). It is mapped onto the evidence ladder
    {exploratory_observational, cross_sectional, case_control, prospective_cohort,
    randomized_controlled_trial} via explicit thresholds: low maturity -> exploratory
    / cross-sectional, high maturity -> confirmatory cohort / RCT.

    Returns one :class:`Insight` per theme (tagged ``study_design``,
    ``is_synthesized=True``) whose reasoning names the maturity inputs and the OBI
    label of the chosen design, and whose :class:`Score` carries the maturity
    components. Deterministic. Empty input -> ``[]``.
    """
    if not themes or len(corpus) == 0:
        return []

    max_year = corpus_max_year(corpus)
    size_ranks = minmax_normalize([float(t.size()) for t in themes])

    insights: list[Insight] = []
    for theme, size_rank in zip(themes, size_ranks, strict=True):
        papers = theme_papers(theme, corpus)
        components = [
            ScoreComponent(
                name="breadth",
                value=_breadth(size_rank, theme.size()),
                weight=_W_BREADTH,
            ),
            ScoreComponent(
                name="recency",
                value=_recency(papers, max_year),
                weight=_W_RECENCY,
            ),
            ScoreComponent(
                name="consistency",
                value=_consistency(theme, papers),
                weight=_W_CONSISTENCY,
            ),
        ]
        score = make_score(components, uncertainty=sample_uncertainty(theme.size()))
        recency_value = next(c.value for c in components if c.name == "recency")
        recency_known = _recency_known(papers, max_year)
        design, was_demoted = _design_for(score.value, recency_value, recency_known)
        label = theme.label or f"theme {theme.id}"
        reasoning = _reasoning(
            label,
            design,
            components,
            score.value,
            score.interpretation,
            was_demoted,
            recency_known,
        )
        evidence = [EvidenceRef(paper_id=p.id) for p in papers[:_EVIDENCE_CAP]]
        insights.append(
            Insight(
                statement=f"Recommended design for theme '{label}': {design}.",
                evidence=evidence,
                is_synthesized=True,
                reasoning=reasoning,
                tags=["study_design"],
                score=score,
            )
        )
    return insights
