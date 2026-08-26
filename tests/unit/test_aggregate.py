from __future__ import annotations

from pathlib import Path

import pytest

from lfx_insights.aggregate import insight_counts, load_run
from lfx_insights.io.store import OutputStore

pytestmark = pytest.mark.unit


def test_load_run_reads_only_current_run(tmp_path: Path) -> None:
    # First run writes artifacts into the shared dir.
    s1 = OutputStore(tmp_path, run="r")
    s1.write_json("stale.astra.json", {"insights": [{"claim": "old"}]})
    s1.write_markdown("stale.md", "old")

    # A LATER run (new store on the SAME dir) must not see the stale files.
    s2 = OutputStore(tmp_path, run="r")
    s2.write_json("fresh.astra.json", {"insights": [{"claim": "a"}, {"claim": "b"}]})
    s2.write_markdown("fresh.md", "new")

    run = load_run(s2)
    assert set(run["astra"]) == {"fresh"}
    assert "stale" not in run["astra"]
    assert run["markdown"] == ["fresh.md"]
    assert insight_counts(run) == {"fresh": 2}


def test_load_run_separates_astra_indicium_other(tmp_path: Path) -> None:
    store = OutputStore(tmp_path, run="r")
    store.write_json("novelty.astra.json", {"insights": []})
    store.write_json("hypotheses.indicium.json", {"claims": {}})
    store.write_json("indicium_sources.json", [{"identifier": "W1"}])
    run = load_run(store)
    assert "novelty" in run["astra"]
    assert "hypotheses" in run["indicium"]
    assert "indicium_sources" in run["other"]
