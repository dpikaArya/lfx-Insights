from __future__ import annotations

import pytest

from lfx_insights.models import Corpus, Paper, Theme
from lfx_insights.scoring.opportunity import rank_opportunities

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    # Corpus max year is 2024 (recent window = year >= 2022).
    return Corpus(
        kb_id="kb",
        papers=[
            # Theme HOT: small + all recent + rising counts.
            Paper(id="H1", title="hot a", year=2023),
            Paper(id="H2", title="hot b", year=2024),
            Paper(id="H3", title="hot c", year=2024),
            # Theme COLD: large + all old + flat/declining.
            Paper(id="C1", title="cold a", year=2015),
            Paper(id="C2", title="cold b", year=2015),
            Paper(id="C3", title="cold c", year=2016),
            Paper(id="C4", title="cold d", year=2016),
            Paper(id="C5", title="cold e", year=2017),
            Paper(id="C6", title="cold f", year=2017),
        ],
    )


def _themes() -> list[Theme]:
    return [
        Theme(id=0, label="Cold", paper_ids=["C1", "C2", "C3", "C4", "C5", "C6"]),
        Theme(id=1, label="Hot", paper_ids=["H1", "H2", "H3"]),
    ]


def test_empty_themes_returns_empty() -> None:
    assert rank_opportunities([], _corpus()) == []


def test_empty_corpus_returns_empty() -> None:
    assert rank_opportunities(_themes(), Corpus(kb_id="kb", papers=[])) == []


def test_tags_and_score_components() -> None:
    insights = rank_opportunities(_themes(), _corpus())
    assert len(insights) == 2
    for ins in insights:
        assert ins.tags == ["opportunity"]
        assert ins.is_synthesized is True
        assert ins.score is not None
        names = [c.name for c in ins.score.components]
        assert names == ["emergence", "scarcity", "momentum"]
        weights = {c.name: c.weight for c in ins.score.components}
        assert weights == {"emergence": 0.4, "scarcity": 0.3, "momentum": 0.3}
        # Score is honest: never a bare magic number.
        assert ins.score.interpretation in {
            "very low",
            "low",
            "moderate",
            "high",
            "very high",
        }
        assert ins.score.uncertainty is not None


def test_ranking_order_descending() -> None:
    insights = rank_opportunities(_themes(), _corpus())
    # Hot theme (small, recent, rising) must outrank Cold theme.
    labels = [ins.statement for ins in insights]
    assert "Hot" in labels[0]
    assert "Cold" in labels[1]
    assert insights[0].score is not None and insights[1].score is not None
    assert insights[0].score.value > insights[1].score.value
    # Sorted descending.
    values = [ins.score.value for ins in insights if ins.score is not None]
    assert values == sorted(values, reverse=True)


def test_emergence_component_values() -> None:
    insights = {ins.statement: ins for ins in rank_opportunities(_themes(), _corpus())}
    hot = next(i for s, i in insights.items() if "Hot" in s)
    cold = next(i for s, i in insights.items() if "Cold" in s)
    assert hot.score is not None and cold.score is not None
    hot_emergence = next(c for c in hot.score.components if c.name == "emergence")
    cold_emergence = next(c for c in cold.score.components if c.name == "emergence")
    # All Hot papers are within the recent window; no Cold papers are.
    assert hot_emergence.value == 1.0
    assert cold_emergence.value == 0.0


def test_scarcity_component_values() -> None:
    insights = {ins.statement: ins for ins in rank_opportunities(_themes(), _corpus())}
    hot = next(i for s, i in insights.items() if "Hot" in s)
    cold = next(i for s, i in insights.items() if "Cold" in s)
    assert hot.score is not None and cold.score is not None
    hot_scarcity = next(c for c in hot.score.components if c.name == "scarcity")
    cold_scarcity = next(c for c in cold.score.components if c.name == "scarcity")
    # Two themes: sizes [6, 3] -> minmax [1.0, 0.0]; scarcity = 1 - that.
    assert cold_scarcity.value == 0.0  # largest theme -> least scarce
    assert hot_scarcity.value == 1.0  # smallest theme -> most scarce


def test_evidence_capped_at_ten() -> None:
    big_papers = [Paper(id=f"P{i}", title=f"p {i}", year=2024) for i in range(15)]
    corpus = Corpus(kb_id="kb", papers=big_papers)
    theme = Theme(id=0, label="Big", paper_ids=[p.id for p in big_papers])
    insights = rank_opportunities([theme], corpus)
    assert len(insights) == 1
    assert len(insights[0].evidence) == 10
    assert all(e.paper_id.startswith("P") for e in insights[0].evidence)


def test_missing_years_are_neutral_not_crash() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="N1", title="no year"),
            Paper(id="N2", title="no year either"),
        ],
    )
    theme = Theme(id=0, label="Unknown", paper_ids=["N1", "N2"])
    insights = rank_opportunities([theme], corpus)
    assert len(insights) == 1
    assert insights[0].score is not None
    emergence = next(c for c in insights[0].score.components if c.name == "emergence")
    momentum = next(c for c in insights[0].score.components if c.name == "momentum")
    assert emergence.value == 0.5  # neutral when no known years
    # No slope signal -> flat slope 0.0 -> sign-aware momentum anchored at 0.5.
    assert momentum.value == 0.5


def _momentum_value(theme_label: str, themes: list[Theme], corpus: Corpus) -> float:
    insights = rank_opportunities(themes, corpus)
    ins = next(i for i in insights if theme_label in i.statement)
    assert ins.score is not None
    return next(c for c in ins.score.components if c.name == "momentum").value


def test_momentum_rising_theme_above_neutral() -> None:
    # Rising counts: 1 paper in 2021, 2 in 2022, 3 in 2023 -> positive slope.
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="R1", title="r", year=2021),
            Paper(id="R2", title="r", year=2022),
            Paper(id="R3", title="r", year=2022),
            Paper(id="R4", title="r", year=2023),
            Paper(id="R5", title="r", year=2023),
            Paper(id="R6", title="r", year=2023),
        ],
    )
    theme = Theme(id=0, label="Rising", paper_ids=[f"R{i}" for i in range(1, 7)])
    assert _momentum_value("Rising", [theme], corpus) > 0.5


def test_momentum_declining_theme_below_neutral() -> None:
    # Declining counts: 3 papers in 2021, 2 in 2022, 1 in 2023 -> negative slope.
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="D1", title="d", year=2021),
            Paper(id="D2", title="d", year=2021),
            Paper(id="D3", title="d", year=2021),
            Paper(id="D4", title="d", year=2022),
            Paper(id="D5", title="d", year=2022),
            Paper(id="D6", title="d", year=2023),
        ],
    )
    theme = Theme(id=0, label="Declining", paper_ids=[f"D{i}" for i in range(1, 7)])
    # A declining theme must NOT score high momentum, regardless of any other theme.
    assert _momentum_value("Declining", [theme], corpus) < 0.5


def test_momentum_is_independent_of_other_themes() -> None:
    # Both a declining and a rising theme present: the declining one must still
    # score < 0.5 and the rising one > 0.5 (no cross-theme rank inflation).
    corpus = Corpus(
        kb_id="kb",
        papers=[
            # Declining theme.
            Paper(id="D1", title="d", year=2021),
            Paper(id="D2", title="d", year=2021),
            Paper(id="D3", title="d", year=2021),
            Paper(id="D4", title="d", year=2022),
            Paper(id="D5", title="d", year=2022),
            Paper(id="D6", title="d", year=2023),
            # Rising theme.
            Paper(id="R1", title="r", year=2021),
            Paper(id="R2", title="r", year=2022),
            Paper(id="R3", title="r", year=2022),
            Paper(id="R4", title="r", year=2023),
            Paper(id="R5", title="r", year=2023),
            Paper(id="R6", title="r", year=2023),
        ],
    )
    themes = [
        Theme(id=0, label="Declining", paper_ids=[f"D{i}" for i in range(1, 7)]),
        Theme(id=1, label="Rising", paper_ids=[f"R{i}" for i in range(1, 7)]),
    ]
    assert _momentum_value("Declining", themes, corpus) < 0.5
    assert _momentum_value("Rising", themes, corpus) > 0.5


def test_momentum_flat_theme_is_neutral() -> None:
    # Equal counts each year -> zero slope -> neutral 0.5.
    corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="F1", title="f", year=2021),
            Paper(id="F2", title="f", year=2022),
            Paper(id="F3", title="f", year=2023),
        ],
    )
    theme = Theme(id=0, label="Flat", paper_ids=["F1", "F2", "F3"])
    assert _momentum_value("Flat", [theme], corpus) == 0.5


def test_momentum_is_deterministic() -> None:
    corpus = _corpus()
    themes = _themes()
    first = _momentum_value("Hot", themes, corpus)
    second = _momentum_value("Hot", themes, corpus)
    assert first == second


def test_missing_paper_id_does_not_crash() -> None:
    corpus = _corpus()
    theme = Theme(id=0, label="Partial", paper_ids=["H1", "GHOST", "H2"])
    insights = rank_opportunities([theme], corpus)
    assert len(insights) == 1
    # Only the two real papers become evidence.
    assert {e.paper_id for e in insights[0].evidence} == {"H1", "H2"}
