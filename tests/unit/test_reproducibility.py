from __future__ import annotations

import pytest

from consilium.lifescience.reproducibility import audit_reproducibility
from consilium.models import Corpus, Paper

pytestmark = pytest.mark.unit


def _paper(pid: str, title: str, abstract: str) -> Paper:
    return Paper(id=pid, title=title, abstract=abstract, source="test")


_RICH_ABSTRACT = (
    "Data available at GEO accession GSE1. Code on github at our repository. "
    "We enrolled n=120 participants. We report p<0.05 with 95% CI and an effect size. "
    "We used 5-fold cross-validation on a held-out set. A negative control was included."
)

_BARREN_ABSTRACT = (
    "We explored a phenomenon and discussed our impressions in a narrative review. "
    "No quantitative analysis was performed."
)


def _rich() -> Paper:
    return _paper("RICH", "A rigorous study", _RICH_ABSTRACT)


def _barren() -> Paper:
    return _paper("BARE", "A loose essay", _BARREN_ABSTRACT)


def test_empty_corpus_returns_empty() -> None:
    assert audit_reproducibility(Corpus(kb_id="kb", papers=[])) == []


def test_one_insight_per_paper_in_order() -> None:
    corpus = Corpus(kb_id="kb", papers=[_rich(), _barren()])
    insights = audit_reproducibility(corpus)
    assert len(insights) == 2
    assert insights[0].evidence[0].paper_id == "RICH"
    assert insights[1].evidence[0].paper_id == "BARE"


def test_insight_shape_tags_evidence_reasoning() -> None:
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[_rich()]))
    ins = insights[0]
    assert ins.tags == ["reproducibility"]
    assert ins.is_synthesized is True
    assert len(ins.evidence) == 1
    assert ins.evidence[0].paper_id == "RICH"
    assert ins.reasoning is not None
    assert "heuristic" in ins.reasoning.lower()
    assert ins.score is not None
    assert ins.score.interpretation in ins.statement
    assert "A rigorous study" in ins.statement


def test_six_weighted_components_with_correct_weights() -> None:
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[_rich()]))
    score = insights[0].score
    assert score is not None
    names = [c.name for c in score.components]
    assert names == [
        "data_availability",
        "code_availability",
        "sample_size_adequacy",
        "statistical_rigor",
        "validation_strategy",
        "controls",
    ]
    weights = {c.name: c.weight for c in score.components}
    assert weights == {
        "data_availability": 0.25,
        "code_availability": 0.20,
        "sample_size_adequacy": 0.15,
        "statistical_rigor": 0.15,
        "validation_strategy": 0.15,
        "controls": 0.10,
    }
    # Not an equal mean: the six weights sum to 1.0 and are not all equal.
    assert sum(c.weight for c in score.components) == pytest.approx(1.0)
    assert len({c.weight for c in score.components}) > 1


def test_rich_paper_scores_high_and_all_dimensions_met() -> None:
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[_rich()]))
    score = insights[0].score
    assert score is not None
    # Every dimension is met -> all values 1.0 -> weighted mean 1.0 -> "very high".
    assert all(c.value == 1.0 for c in score.components)
    assert score.value == pytest.approx(1.0)
    assert score.value > 0.8


def test_barren_paper_scores_low_and_no_dimensions_met() -> None:
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[_barren()]))
    score = insights[0].score
    assert score is not None
    assert all(c.value == 0.0 for c in score.components)
    assert score.value == pytest.approx(0.0)
    assert score.value < 0.2


def test_rich_scores_strictly_higher_than_barren() -> None:
    corpus = Corpus(kb_id="kb", papers=[_rich(), _barren()])
    insights = audit_reproducibility(corpus)
    rich_score, barren_score = insights[0].score, insights[1].score
    assert rich_score is not None
    assert barren_score is not None
    assert rich_score.value > barren_score.value


def test_small_sample_n_below_threshold_does_not_count() -> None:
    # An explicit n is present (n=5) but < 30: sample_size_adequacy must be 0.
    paper = _paper(
        "SMALL",
        "A pilot study",
        "This was a small sample (n=5) pilot. We report p<0.05.",
    )
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    score = insights[0].score
    assert score is not None
    sample = next(c for c in score.components if c.name == "sample_size_adequacy")
    assert sample.value == 0.0
    # Statistical rigor *is* met here (p<0.05), so the regexes are independent.
    stat = next(c for c in score.components if c.name == "statistical_rigor")
    assert stat.value == 1.0


def test_adequate_sample_n_at_threshold_counts() -> None:
    paper = _paper("N30", "A study", "We recruited n = 30 subjects across two sites.")
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    score = insights[0].score
    assert score is not None
    sample = next(c for c in score.components if c.name == "sample_size_adequacy")
    assert sample.value == 1.0


def test_largest_n_clears_threshold_even_if_a_small_n_also_present() -> None:
    paper = _paper(
        "MIX",
        "A study with subgroups",
        "A subgroup of n=5 was analyzed within the full cohort of n=200.",
    )
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    sample = next(
        c
        for c in insights[0].score.components
        if c.name == "sample_size_adequacy"  # type: ignore[union-attr]
    )
    assert sample.value == 1.0


def test_word_boundary_avoids_false_code_match() -> None:
    # "githubbery" must NOT match the github word-boundary pattern; no other dim present.
    paper = _paper("WB", "Untitled", "We discussed githubbery as a cultural concept.")
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    code = next(
        c
        for c in insights[0].score.components
        if c.name == "code_availability"  # type: ignore[union-attr]
    )
    assert code.value == 0.0


def test_individual_dimensions_detected() -> None:
    cases = {
        "data_availability": "The dataset was deposited in zenodo for reuse.",
        "code_availability": "Our source code is released on gitlab.",
        "statistical_rigor": "Differences were significant after Bonferroni correction.",
        "validation_strategy": "We validated on an independent cohort.",
        "controls": "A placebo arm served as the comparator.",
    }
    for dim, abstract in cases.items():
        insights = audit_reproducibility(Corpus(kb_id="kb", papers=[_paper(dim, "t", abstract)]))
        comp = next(c for c in insights[0].score.components if c.name == dim)  # type: ignore[union-attr]
        assert comp.value == 1.0, f"expected {dim} to be detected in: {abstract!r}"


@pytest.mark.parametrize(
    "abstract",
    [
        "Findings were confirmed in three biological replicates.",
        "The experiment was performed in technical replicates.",
        "We confirmed the result across independent replications.",
        "The assay was replicated three times.",
        "Replicating the protocol confirmed the effect.",
    ],
)
def test_validation_strategy_replicate_phrasings_detected(abstract: str) -> None:
    paper = _paper("REP", "A study", abstract)
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    comp = next(
        c
        for c in insights[0].score.components  # type: ignore[union-attr]
        if c.name == "validation_strategy"
    )
    assert comp.value == 1.0, f"expected validation_strategy to be met in: {abstract!r}"


@pytest.mark.parametrize(
    "abstract",
    [
        "A control group was compared against the treatment arm.",
        "A negative control was included in every plate.",
        "A positive control confirmed assay sensitivity.",
        "Patients received a placebo in the comparator arm.",
        "Appropriate controls were run alongside each sample.",
    ],
)
def test_controls_phrasings_detected(abstract: str) -> None:
    paper = _paper("CTRL", "A study", abstract)
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    comp = next(
        c
        for c in insights[0].score.components  # type: ignore[union-attr]
        if c.name == "controls"
    )
    assert comp.value == 1.0, f"expected controls to be met in: {abstract!r}"


def test_partial_paper_scores_between_extremes() -> None:
    # Only data + code available (0.25 + 0.20 = 0.45 of weight), nothing else.
    paper = _paper(
        "PART",
        "Partial",
        "Data available at figshare; code available on github. No stats reported.",
    )
    insights = audit_reproducibility(Corpus(kb_id="kb", papers=[paper]))
    score = insights[0].score
    assert score is not None
    assert score.value == pytest.approx(0.45)
    met = {c.name for c in score.components if c.value == 1.0}
    assert met == {"data_availability", "code_availability"}


def test_full_texts_used_and_disclosed() -> None:
    from consilium.lifescience.reproducibility import audit_reproducibility
    from consilium.models import Corpus, Paper

    c = Corpus(kb_id="k", papers=[Paper(id="W1", title="t", abstract="a sparse abstract")])
    base = audit_reproducibility(c)[0]
    assert "abstract-only" in (base.reasoning or "")
    rich = (
        "Data available at GEO. Code on github. n=120. p<0.05. "
        "5-fold cross-validation. negative control."
    )
    full = audit_reproducibility(c, full_texts={"W1": rich})[0]
    assert "full text" in (full.reasoning or "")
    assert base.score is not None and full.score is not None
    assert full.score.value > base.score.value
