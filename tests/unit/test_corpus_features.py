from __future__ import annotations

import pytest

from consilium.corpus_features import (
    keyword_homogeneity,
    max_year,
    theme_papers,
    theme_years,
)
from consilium.models import Corpus, Paper, Theme

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="graph networks",
                abstract="we study graph neural networks",
                year=2020,
            ),
            Paper(
                id="W2", title="protein folding", abstract="alphafold predicts structure", year=2024
            ),
            Paper(id="W3", title="no year", abstract="dated unknown"),
        ],
    )


def test_max_year_and_resolution() -> None:
    c = _corpus()
    assert max_year(c) == 2024
    assert max_year(Corpus(kb_id="e", papers=[Paper(id="X", title="t")])) is None
    t = Theme(id=0, paper_ids=["W1", "W3", "missing"])
    assert [p.id for p in theme_papers(t, c)] == ["W1", "W3"]
    assert theme_years(t, c) == [2020]  # W3 undated, missing dropped


def test_keyword_homogeneity_word_boundary() -> None:
    c = _corpus()
    # "graph" appears in W1 only; "structure" in W2 only.
    t = Theme(id=0, paper_ids=["W1", "W2"], keywords=["graph", "structure"])
    h = keyword_homogeneity(t, c)
    assert 0.0 < h < 1.0
    # No keywords -> neutral 0.5
    assert keyword_homogeneity(Theme(id=1, paper_ids=["W1"]), c) == 0.5
    # Substring must NOT match across a word boundary: "net" should not match "networks".
    t2 = Theme(id=2, paper_ids=["W1"], keywords=["net"])
    assert keyword_homogeneity(t2, c) == 0.0
