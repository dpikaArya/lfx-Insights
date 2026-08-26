"""Live eval-harness smoke against a running PerspicacitÃ© MCP server (:8002).

Exercises the real PerspicacitÃ© retrieval -> Consilium answer -> citation scoring
path through the ablation runner. Uses a deterministic MockLLM (so no LLM key is
needed); the assertion is that the perspicacite condition actually retrieves real
documents and the runner produces a citation Score. Auto-skips when :8002 is down.

Run with: uv run pytest -m live
"""

from __future__ import annotations

import httpx
import pytest

from lfx_insights.config import load_settings
from lfx_insights.eval.dataset import load_dataset
from lfx_insights.eval.runner import run_ablation
from lfx_insights.llm.client import MockLLM

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
def test_perspicacite_condition_retrieves_and_scores() -> None:
    settings = load_settings(None)
    settings.eval.retrieval_k = 5
    cases = load_dataset("bundled")[:1]

    report = run_ablation(
        cases,
        conditions=["perspicacite"],
        llm=MockLLM(),
        settings=settings,
        dataset="bundled-live",
        metrics_override=["citation"],
        judge="lexical",
    )

    [cond] = report.conditions
    assert cond.condition == "perspicacite"
    # Real retrieval happened for the (single) case.
    assert report.cases and report.cases[0].n_retrieved >= 1
    # A citation Score is produced (value may be low with a mock answerer).
    assert cond.citation is not None
    assert any("Perspicac" in c for c in cond.caveats)
