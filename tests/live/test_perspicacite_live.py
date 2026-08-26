"""Live integration tests against a running PerspicacitÃ© MCP server (:8002).

Run with: uv run pytest -m live
These are skipped automatically when PerspicacitÃ© is not reachable.
"""

from __future__ import annotations

import httpx
import pytest

from lfx_insights.config import load_settings
from lfx_insights.errors import PerspicaciteUnavailable
from lfx_insights.scoring.evidence_strength import score_evidence_strength
from lfx_insights.scoring.novelty import score_novelty
from lfx_insights.sources.perspicacite import PerspicaciteBackend
from lfx_insights.themes.discover import SimpleEmbedder, discover_themes

pytestmark = pytest.mark.live

URL = load_settings(None).perspicacite.url


def _perspicacite_up() -> bool:
    try:
        httpx.post(
            URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            },
            timeout=5,
        )
        return True
    except httpx.HTTPError:
        return False


requires_perspicacite = pytest.mark.skipif(
    not _perspicacite_up(), reason="PerspicacitÃ© MCP not reachable on :8002"
)


@requires_perspicacite
def test_live_search_literature_returns_real_papers() -> None:
    backend = PerspicaciteBackend(URL, timeout=120)
    try:
        corpus = backend.build_or_select_kb(
            "graph neural networks for drug discovery", max_papers=5
        )
    finally:
        backend.close()
    assert len(corpus) >= 1
    assert corpus.papers[0].title  # real metadata came back
    assert corpus.kb_id == "graph-neural-networks-for-drug-discovery"


@requires_perspicacite
def test_live_scoring_on_real_corpus() -> None:
    backend = PerspicaciteBackend(URL, timeout=120)
    try:
        corpus = backend.build_or_select_kb("single cell RNA sequencing", max_papers=8)
    finally:
        backend.close()
    themes = discover_themes(corpus.papers, SimpleEmbedder())
    assert themes
    ev = score_evidence_strength(themes, corpus)
    nov = score_novelty(themes, corpus)
    assert ev and ev[0].score is not None and 0.0 <= ev[0].score.value <= 1.0
    assert nov and nov[0].score is not None


@requires_perspicacite
def test_live_unknown_kb_surfaces_app_error() -> None:
    # App-level PerspicacitÃ© failures must be surfaced, not silently swallowed:
    # querying a non-existent KB raises PerspicaciteUnavailable (not an empty list).
    backend = PerspicaciteBackend(URL, timeout=60)
    try:
        with pytest.raises(PerspicaciteUnavailable):
            backend.relevant_passages("microbiome", "lfx-insights-nonexistent-kb-xyz", k=2)
    finally:
        backend.close()
