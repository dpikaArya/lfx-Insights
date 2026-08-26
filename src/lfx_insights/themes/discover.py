"""Theme discovery: embed papers, cluster (seeded), extract keywords.

Honest by construction: no fabricated "confidence" numbers are attached to themes;
k is chosen by silhouette over a bounded range, and tiny corpora collapse to a
single theme rather than inventing structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from tenacity import retry, stop_after_attempt, wait_exponential

from lfx_insights.models import Paper, Theme

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lfx_insights.config import Settings


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float64]: ...


class SimpleEmbedder:
    """Deterministic TF-IDF embedder (no network). Default and offline embedder."""

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        vec = TfidfVectorizer(stop_words="english", max_features=512)
        matrix = vec.fit_transform(texts)
        return np.asarray(matrix.todense(), dtype=np.float64)


class STEmbedder:
    """Local sentence-transformers embedder (downloads the model on first use).

    Any Hugging Face / sentence-transformers model name works; for scientific text a
    domain model (e.g. ``allenai/specter2``, ``malteos/scincl``,
    ``NeuML/pubmedbert-base-embeddings``) discriminates far better than the generic
    ``all-MiniLM-L6-v2`` default.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        return np.asarray(model.encode(texts), dtype=np.float64)


class LiteLLMEmbedder:
    """Hosted-API embedder via LiteLLM â€” the high-quality generalist tier.

    Use for cross-domain corpora where a hosted model generalises best, e.g.
    ``text-embedding-3-large`` (OpenAI, 3072-dim; needs the provider API key).
    Routes OpenAI/Cohere/Voyage/etc. through LiteLLM, like PerspicacitÃ©'s embedding layer.
    """

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=20))
    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        import litellm

        resp = litellm.embedding(model=self.model, input=texts)
        items = sorted(resp["data"], key=lambda d: d.get("index", 0))
        return np.asarray([d["embedding"] for d in items], dtype=np.float64)


# LiteLLM provider prefixes for hosted embeddings. NOT a bare "/" check: local HF
# model names are also "org/model" (e.g. allenai/specter2), so only an explicit
# provider prefix routes to the API tier.
_API_PREFIXES = (
    "openai/",
    "azure/",
    "cohere/",
    "voyage/",
    "mistral/",
    "gemini/",
    "vertex_ai/",
    "bedrock/",
    "jina_ai/",
    "deepinfra/",
)


def _is_api_embedding(model: str) -> bool:
    """True for hosted-API embedding models: bare OpenAI ``text-embedding-*`` names,
    or an explicit ``<provider>/<model>`` route LiteLLM recognises. Plain ``org/model``
    Hugging Face names (e.g. ``allenai/specter2``) are local sentence-transformers.
    """
    m = model.lower()
    return m.startswith("text-embedding-") or m.startswith(_API_PREFIXES)


def default_embedder(settings: Settings) -> Embedder:
    """Pick the embedder from ``settings.embedding.model``:

    - ``"tfidf"`` -> deterministic local TF-IDF (no network; offline/CI default);
    - an API model (``text-embedding-3-large`` or ``<provider>/<model>``) -> LiteLLM;
    - anything else -> local sentence-transformers.
    """
    model = settings.embedding.model
    if model.lower() == "tfidf":
        return SimpleEmbedder()
    if _is_api_embedding(model):
        return LiteLLMEmbedder(model)
    return STEmbedder(model)


def _choose_k(vectors: NDArray[np.float64], random_state: int) -> int:
    n = vectors.shape[0]
    # Cap k relative to corpus size (~>=4 papers/theme) so small corpora are not
    # shattered into many 1-2 paper themes (silhouette alone over-clusters here).
    upper = min(10, n - 1, max(2, n // 4))
    if upper < 2:
        return 1
    best_k, best_score = 2, -1.0
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(vectors, labels))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _keywords_for(texts: list[str], top_n: int = 6) -> list[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = vec.fit_transform(texts)
    means = np.asarray(matrix.mean(axis=0)).ravel()
    names = vec.get_feature_names_out()
    order = means.argsort()[::-1][:top_n]
    return [str(names[i]) for i in order if means[i] > 0]


def discover_themes(
    papers: list[Paper],
    embedder: Embedder,
    k: int | None = None,
    random_state: int = 0,
) -> list[Theme]:
    """Cluster ``papers`` into themes. Deterministic given a deterministic embedder."""
    if not papers:
        return []
    if len(papers) < 4:
        return [
            Theme(
                id=0,
                paper_ids=[p.id for p in papers],
                keywords=_keywords_for([p.text() for p in papers]),
            )
        ]

    vectors = embedder.encode([p.text() for p in papers])
    n_clusters = k if k is not None else _choose_k(vectors, random_state)
    if n_clusters <= 1:
        return [
            Theme(
                id=0,
                paper_ids=[p.id for p in papers],
                keywords=_keywords_for([p.text() for p in papers]),
            )
        ]

    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        vectors
    )
    themes: list[Theme] = []
    for cluster_id in range(n_clusters):
        members = [p for p, lab in zip(papers, labels, strict=True) if lab == cluster_id]
        if not members:
            continue
        themes.append(
            Theme(
                id=cluster_id,
                paper_ids=[p.id for p in members],
                keywords=_keywords_for([p.text() for p in members]),
            )
        )
    return themes
