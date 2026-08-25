from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from consilium.models import Corpus, Paper
from consilium.scoring.gap_validation import validate_gaps

pytestmark = pytest.mark.unit


class StubEmbedder:
    """Fixed vectors keyed by the first whitespace token of each text.

    Corpus papers sit on three orthonormal axes; a 4th axis is OFF-corpus, so a gap
    can be on-topic-but-open (small projection onto a corpus axis) or off-topic
    (lives entirely on the 4th axis).
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        out = [self.vectors[t.split()[0]] for t in texts]
        return np.asarray(out, dtype=np.float64)


def _papers() -> list[Paper]:
    return [
        Paper(id="W1", title="Alpha graph networks", abstract="alpha details", year=2021),
        Paper(id="W2", title="Beta generative models", abstract="beta details", year=2022),
        Paper(id="W3", title="Gamma protein folding", abstract="gamma details", year=2023),
    ]


def _corpus() -> Corpus:
    return Corpus(kb_id="kb", papers=_papers())


def _embedder() -> StubEmbedder:
    return StubEmbedder(
        {
            "Alpha": [1.0, 0.0, 0.0, 0.0],
            "Beta": [0.0, 1.0, 0.0, 0.0],
            "Gamma": [0.0, 0.0, 1.0, 0.0],
            "ALIGNED": [1.0, 0.0, 0.0, 0.0],  # cosine 1.0 with Alpha -> Not Supported
            "OPEN": [0.2, 0.0, 0.0, 1.0],  # max cosine ~0.20 -> on-topic, Confirmed
            "OFFTOPIC": [0.0, 0.0, 0.0, 1.0],  # cosine 0 with all -> Out of Scope
        }
    )


def test_empty_gaps_returns_empty() -> None:
    assert validate_gaps([], _corpus(), _embedder()) == []


def test_empty_corpus_returns_empty() -> None:
    empty = Corpus(kb_id="kb", papers=[])
    assert validate_gaps(["ALIGNED gap"], empty, _embedder()) == []


def test_aligned_gap_is_not_supported() -> None:
    [insight] = validate_gaps(["ALIGNED gap text"], _corpus(), _embedder())
    assert "Not Supported" in insight.statement
    assert insight.tags == ["gap", "not_supported"]
    assert insight.score is not None and insight.score.value < 0.4


def test_on_topic_open_gap_is_confirmed() -> None:
    [insight] = validate_gaps(["OPEN gap text"], _corpus(), _embedder())
    assert "Confirmed" in insight.statement
    assert insight.tags == ["gap", "confirmed"]
    assert "candidate open gap" in (insight.reasoning or "")
    assert insight.score is not None and insight.score.value > 0.6


def test_offtopic_gap_is_out_of_scope_with_high_uncertainty() -> None:
    [insight] = validate_gaps(["OFFTOPIC nonsense gap"], _corpus(), _embedder())
    assert "Out of Scope" in insight.statement
    assert insight.tags == ["gap", "out_of_scope"]
    # The verdict must warn that it may be off-topic, and be LESS certain, not more.
    assert "OFF-TOPIC" in (insight.reasoning or "")
    assert insight.score is not None and insight.score.uncertainty is not None
    assert insight.score.uncertainty >= 0.8


def test_score_carries_components_and_band() -> None:
    [insight] = validate_gaps(["OPEN gap"], _corpus(), _embedder())
    assert insight.score is not None
    assert {c.name for c in insight.score.components} == {"under_coverage", "sparsity"}
    assert insight.score.interpretation != "" and insight.score.uncertainty is not None


def test_evidence_is_capped_and_ranked() -> None:
    [insight] = validate_gaps(["ALIGNED gap"], _corpus(), _embedder())
    assert len(insight.evidence) <= 3
    assert insight.evidence[0].paper_id == "W1"


def test_missing_years_do_not_crash() -> None:
    corpus = Corpus(
        kb_id="kb",
        papers=[Paper(id="W1", title="Alpha x", abstract="a"), Paper(id="W2", title="Beta y")],
    )
    insights = validate_gaps(["OPEN gap"], corpus, _embedder())
    assert len(insights) == 1 and insights[0].score is not None


def test_one_insight_per_gap_in_order() -> None:
    insights = validate_gaps(["ALIGNED first", "OPEN second"], _corpus(), _embedder())
    assert len(insights) == 2
    assert "ALIGNED first" in insights[0].statement
    assert "OPEN second" in insights[1].statement


def test_corpus_coherence_helper() -> None:
    from consilium.scoring.gap_validation import _corpus_coherence

    assert _corpus_coherence([[1.0, 0.0]]) == 0.0  # < 2 papers
    # Orthogonal papers -> no internal coherence.
    assert _corpus_coherence([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) == 0.0
    # Identical papers -> coherence 1.0.
    assert _corpus_coherence([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]) == 1.0


def test_offtopic_floor_scales_with_corpus_coherence() -> None:
    # A gap with cosine ~0.4 to its nearest paper: off-topic in a COHERENT corpus
    # (high relative floor) but not in an INCOHERENT one — same coverage, different verdict.
    gap = [0.4, 0.9165, 0.0, 0.0]  # cosine 0.4 with the [1,0,0,0] axis

    coherent_emb = StubEmbedder({"MID": gap, "P": [1.0, 0.0, 0.0, 0.0]})
    coherent_corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="P a"),
            Paper(id="W2", title="P b"),
            Paper(id="W3", title="P c"),
        ],
    )
    [coherent] = validate_gaps(["MID gap"], coherent_corpus, coherent_emb)
    assert "Out of Scope" in coherent.statement  # 0.4 < 0.5 effective floor

    incoherent_emb = StubEmbedder(
        {"MID": gap, "Alpha": [1, 0, 0, 0], "Beta": [0, 1, 0, 0], "Gamma": [0, 0, 1, 0]}
    )
    [incoherent] = validate_gaps(["MID gap"], _corpus(), incoherent_emb)
    assert "Out of Scope" not in incoherent.statement  # floor falls back to 0.12


def test_related_threshold_scales_with_corpus_coherence() -> None:
    # Same coverage 0.55 to the nearest paper, two corpora. In a COHERENT corpus the
    # related threshold rises to 0.6*coherence, so 0.55 sits below it -> Confirmed
    # (on-topic, no close prior work). In an INCOHERENT corpus the threshold falls
    # back to the absolute 0.30 floor, so 0.55 is above it -> Uncertain (partial cover).
    # cosine 0.55 with the [1,0,0,0] axis; filler on the OFF-corpus 4th axis so the
    # gap does not accidentally align with another corpus axis in the orthogonal case.
    gap = [0.55, 0.0, 0.0, 0.8352]

    coherent_emb = StubEmbedder({"MID": gap, "P": [1.0, 0.0, 0.0, 0.0]})
    coherent_corpus = Corpus(
        kb_id="kb",
        papers=[
            Paper(id="W1", title="P a"),
            Paper(id="W2", title="P b"),
            Paper(id="W3", title="P c"),
        ],
    )
    [coherent] = validate_gaps(["MID gap"], coherent_corpus, coherent_emb)
    assert "Confirmed" in coherent.statement  # 0.55 <= 0.6 related threshold

    incoherent_emb = StubEmbedder(
        {"MID": gap, "Alpha": [1, 0, 0, 0], "Beta": [0, 1, 0, 0], "Gamma": [0, 0, 1, 0]}
    )
    [incoherent] = validate_gaps(["MID gap"], _corpus(), incoherent_emb)
    assert "Uncertain" in incoherent.statement  # 0.55 > 0.30 fallback related threshold
