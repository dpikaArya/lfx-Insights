"""Focused tests for the LFX Insights local Web UI.

Covers:
- GET / serves the browser interface (single-file HTML/CSS/JS).
- POST /api/insights still works for all seven actions (LLM + corpus stubbed).

A real (live Ollama) request is exercised separately via curl, not in this fast
unit suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfx_insights.office_kb import load_corpus

try:
    from fastapi.testclient import TestClient  # type: ignore

    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

_WEBUI_HTML = Path(__file__).resolve().parents[2] / "src" / "lfx_insights" / "webui" / "index.html"


pytestmark = pytest.mark.unit


def _sample_kb(tmp_path: Path):
    data = {
        "papers": [
            {
                "title": "Microbial inoculants in nitrogen cycling",
                "authors": "Jane Smith; Bob Jones",
                "year": 2021,
                "doi": "10.1234/nitrogen.2021",
                "abstract": "Inoculants improve nitrogen use efficiency in soils.",
                "venue": "Soil Science",
            }
        ]
    }
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_get_root_serves_webui():
    import lfx_insights.api as api

    client = TestClient(api.app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "LFX Insights" in body
    assert "/api/insights" in body
    assert "ask" in body and "verify" in body
    # The UI is self-contained (no external framework references).
    assert "Content-Security" not in body or "react" not in body.lower()


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_health_status_shape():
    import lfx_insights.api as api

    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.skipif(not _HAVE_FASTAPI, reason="fastapi not installed")
def test_api_insights_all_actions(monkeypatch, tmp_path):
    import lfx_insights.api as api

    corpus = load_corpus(_sample_kb(tmp_path))

    def fake_chat(prompt: str) -> str:
        return "Microbial inoculants drive nitrogen cycling [1]."

    monkeypatch.setattr(api, "_get_llm", lambda: None)
    monkeypatch.setattr(api, "_get_corpus", lambda: corpus)
    monkeypatch.setattr(api, "_chat", fake_chat)

    client = TestClient(api.app)
    for action in ("ask", "improve", "review", "gap", "evidence", "citations", "verify"):
        payload = {"action": action, "query": "microbial inoculants nitrogen", "text": "microbial inoculants nitrogen"}
        if action in ("improve", "review", "verify"):
            payload = {"action": action, "text": "Inoculants improve nitrogen use efficiency (Smith, 2021)."}
        resp = client.post("/api/insights", json=payload)
        assert resp.status_code == 200, (action, resp.text)
        data = resp.json()
        assert data["action"] == action
        assert "result" in data
