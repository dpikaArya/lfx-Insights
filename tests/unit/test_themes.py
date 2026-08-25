from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from consilium.models import Corpus, Paper
from consilium.themes.discover import SimpleEmbedder, discover_themes
from consilium.themes.label import ThemeLabel, label_themes

pytestmark = pytest.mark.unit


class StubEmbedder:
    """Returns fixed vectors so clustering is fully deterministic."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        # texts come in paper order; map by first token of the title line
        out = []
        for t in texts:
            key = t.split()[0]
            out.append(self.vectors[key])
        return np.asarray(out, dtype=np.float64)


def _papers() -> list[Paper]:
    return [
        Paper(id="W1", title="Alpha methods one"),
        Paper(id="W2", title="Alpha methods two"),
        Paper(id="W3", title="Beta results three"),
        Paper(id="W4", title="Beta results four"),
    ]


def test_discover_themes_deterministic() -> None:
    stub = StubEmbedder({"Alpha": [1.0, 0.0], "Beta": [0.0, 1.0]})
    themes = discover_themes(_papers(), stub, k=2, random_state=0)
    groups = sorted(sorted(t.paper_ids) for t in themes)
    assert groups == [["W1", "W2"], ["W3", "W4"]]


def test_small_corpus_single_theme() -> None:
    stub = StubEmbedder({"Alpha": [1.0, 0.0]})
    themes = discover_themes(_papers()[:2], stub)
    assert len(themes) == 1
    assert sorted(themes[0].paper_ids) == ["W1", "W2"]


def test_label_themes_uses_llm() -> None:
    from consilium.llm.client import MockLLM

    stub = StubEmbedder({"Alpha": [1.0, 0.0], "Beta": [0.0, 1.0]})
    themes = discover_themes(_papers(), stub, k=2)
    corpus = Corpus(kb_id="kb", papers=_papers())
    llm = MockLLM(responder=lambda p, m: ThemeLabel(label="Topic", rationale="because"))
    labeled = label_themes(themes, corpus, llm)
    assert all(t.label == "Topic" for t in labeled)
    # No fabricated confidence score is attached to a theme.
    assert not hasattr(labeled[0], "confidence")


def test_choose_k_caps_small_corpora() -> None:
    # 16 papers -> k capped at 16//4 = 4 (no shattering into many tiny themes).
    papers = [
        Paper(
            id=f"W{i}", title=f"Topic{i % 4} study {i}", abstract=f"about topic {i % 4} number {i}"
        )
        for i in range(16)
    ]
    themes = discover_themes(papers, SimpleEmbedder(), random_state=0)
    assert 1 <= len(themes) <= 4


def test_default_embedder_routing() -> None:
    from consilium.config import Settings
    from consilium.themes.discover import (
        LiteLLMEmbedder,
        STEmbedder,
        default_embedder,
    )

    def emb(model: str):
        s = Settings()
        s.embedding.model = model
        return default_embedder(s)

    assert isinstance(emb("tfidf"), SimpleEmbedder)
    assert isinstance(emb("all-MiniLM-L6-v2"), STEmbedder)
    # An HF org/model name is a LOCAL sentence-transformers model, not an API route.
    assert isinstance(emb("allenai/specter2"), STEmbedder)
    assert isinstance(emb("text-embedding-3-large"), LiteLLMEmbedder)
    assert isinstance(emb("openai/text-embedding-3-large"), LiteLLMEmbedder)


def test_litellm_embedder_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from consilium.themes.discover import LiteLLMEmbedder

    def fake_embedding(model: str, input: list[str]) -> dict:
        # return out of order to prove we sort by index
        return {
            "data": [
                {"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(len(input)))
            ]
        }

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    out = LiteLLMEmbedder("text-embedding-3-large").encode(["a", "b", "c"])
    assert out.shape == (3, 2)
    assert out[2].tolist() == [2.0, 1.0]
