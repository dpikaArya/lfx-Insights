"""Tests for the Word Add-in knowledge-base helpers and /api/insights endpoint.

The knowledge-base tests are fully offline (no LLM). The API tests stub the LLM
and corpus so the endpoint wiring can be exercised without Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfx_insights.models import Author, Corpus, Paper
from lfx_insights.office_kb import (
    ground_numeric_markers,
    load_corpus,
    render_citations,
    retrieve,
    verify_text_citations,
)


def _sample_kb(tmp_path: Path) -> Path:
    data = {
        "papers": [
            {
                "title": "Microbial inoculants in nitrogen cycling",
                "authors": "Jane Smith; Bob Jones",
                "year": 2021,
                "doi": "10.1234/nitrogen.2021",
                "abstract": "Inoculants improve nitrogen use efficiency in soils.",
                "venue": "Soil Science",
                "url": "https://doi.org/10.1234/nitrogen.2021",
            },
            {
                "title": "Phosphorus dynamics in agroecosystems",
                "authors": "Carol Lee",
                "year": 2019,
                "doi": "10.5678/phosphorus.2019",
                "abstract": "Phosphorus availability limits crop yield.",
                "venue": "Agronomy",
            },
        ]
    }
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(data))
    return p


def test_load_corpus_parses_papers(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    assert len(corpus.papers) == 2
    smith = next(p for p in corpus.papers if "nitrogen" in p.title)
    assert smith.doi == "10.1234/nitrogen.2021"
    assert [a.name for a in smith.authors] == ["Jane Smith", "Bob Jones"]
    assert smith.year == 2021


def test_retrieve_ranks_relevant_paper_first(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    hits = retrieve(corpus, "microbial inoculants nitrogen", k=5)
    assert hits
    top = hits[0][0]
    assert "nitrogen" in top.title.lower()
    assert all(score > 0 for _, score in hits)


def test_retrieve_empty_query_returns_nothing(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    assert retrieve(corpus, "   ", k=5) == []


def test_ground_numeric_markers_maps_to_doi(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    text = "Inoculants help [1] and phosphorus matters [2]."
    out = ground_numeric_markers(text, corpus.papers)
    assert "10.1234/nitrogen.2021" in out
    assert "10.5678/phosphorus.2019" in out


def test_render_citations_drops_unknown_and_formats_apa(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    # marker [3] is out of range -> left as prose and dropped from citations
    draft = "Nitrogen is central [1]. Unrelated [3]."
    rendered, references, cited = render_citations(draft, corpus.papers, corpus)
    assert "Smith" in rendered and "2021" in rendered
    assert "10.1234/nitrogen.2021" in references
    assert cited == ["10.1234/nitrogen.2021"]
    assert "[3]" not in references


def test_verify_text_citations_detects_known_and_unknown(tmp_path: Path) -> None:
    corpus = load_corpus(_sample_kb(tmp_path))
    text = "As shown by (Smith, 2021), inoculants matter."
    report = verify_text_citations(text, corpus)
    assert report["all_exist"] is True
    assert report["verified"] is True
    assert report["cited_ids"]

    text2 = "A fictional claim (Fake Author, 1999) is unsupported."
    report2 = verify_text_citations(text2, corpus)
    assert report2["all_exist"] is False
    assert report2["verified"] is False
    assert report2["unresolved"]


# ---------------------------------------------------------------------------
# API endpoint (stubbed LLM + corpus)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.unit

try:
    from fastapi.testclient import TestClient  # type: ignore

    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover - fastapi optional in some envs
    _HAVE_FASTAPI = False


@pytest.fixture()
def api_module():
    import lfx_insights.api as api

    return api


def _stub_corpus(tmp_path: Path) -> Corpus:
    return load_corpus(_sample_kb(tmp_path))


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_api_insights_citations_renders_apa(api_module, tmp_path, monkeypatch):
    corpus = _stub_corpus(tmp_path)

    def fake_chat(prompt: str) -> str:
        return "Microbial inoculants drive nitrogen cycling [1]."

    monkeypatch.setattr(api_module, "_get_llm", lambda: None)
    monkeypatch.setattr(api_module, "_get_corpus", lambda: corpus)
    monkeypatch.setattr(api_module, "_chat", fake_chat)

    client = TestClient(api_module.app)
    resp = client.post(
        "/api/insights",
        json={"action": "citations", "query": "microbial inoculants nitrogen"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["action"] == "citations"
    assert "Smith" in data["citations"]["references"]
    assert data["cited_ids"] == ["10.1234/nitrogen.2021"]
    assert data["evidence"]


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_api_insights_verify(api_module, tmp_path, monkeypatch):
    corpus = _stub_corpus(tmp_path)
    monkeypatch.setattr(api_module, "_get_corpus", lambda: corpus)

    client = TestClient(api_module.app)
    resp = client.post(
        "/api/insights",
        json={"action": "verify", "text": "As shown by (Smith, 2021), inoculants matter."},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["verified"] is True
    assert data["cited_ids"]


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_api_insights_missing_topic_400(api_module, monkeypatch):
    monkeypatch.setattr(api_module, "_get_corpus", lambda: Corpus(kb_id="x", papers=[]))
    client = TestClient(api_module.app)
    resp = client.post("/api/insights", json={"action": "ask", "query": ""})
    assert resp.status_code == 400
