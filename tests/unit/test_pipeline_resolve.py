"""Unit tests for pipeline stage resolution, kb-snapshot themes, and run records.

These lock the verified code-review fixes in ``lfx_insights.pipeline``:

1. ``_resolve`` raises :class:`~lfx_insights.errors.InsightsError` (listing the
   offending names) instead of silently dropping unknown stages.
2. ``run_kb_snapshot_stage`` ensures themes are discovered before serializing,
   rather than writing the possibly-empty ``ctx.themes``.
3. ``Stage`` no longer advertises an unenforced ``deps`` field.
4. ``run`` threads the full executed-stage list into the project/memory records,
   so they reflect every stage that ran (not just ASTRA-producing ones).
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from lfx_insights.context import RunContext
from lfx_insights.errors import InsightsError
from lfx_insights.io.store import OutputStore
from lfx_insights.llm.client import MockLLM
from lfx_insights.logging import configure
from lfx_insights.pipeline import PIPELINE, Stage, _resolve
from lfx_insights.pipeline import run as run_pipeline
from lfx_insights.sources.fake import FakeBackend
from lfx_insights.themes.discover import SimpleEmbedder
from lfx_insights.themes.label import ThemeLabel

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        settings=None,  # type: ignore[arg-type]  # stages here don't read settings
        backend=FakeBackend(),
        llm=MockLLM(responder=lambda p, m: ThemeLabel(label="Topic", rationale="r")),
        embedder=SimpleEmbedder(),
        store=OutputStore(tmp_path, run="t"),
        log=configure(),
    )


# --- finding 1: _resolve raises on unknown stage -----------------------------


def test_resolve_none_returns_full_pipeline() -> None:
    assert _resolve(None) == PIPELINE
    assert _resolve([]) == PIPELINE


def test_resolve_known_stages_preserves_order() -> None:
    resolved = _resolve(["novelty", "themes"])
    assert [s.name for s in resolved] == ["novelty", "themes"]


def test_resolve_raises_on_unknown_stage() -> None:
    with pytest.raises(InsightsError) as excinfo:
        _resolve(["themes", "does_not_exist"])
    # The unknown name is surfaced; the known stage is not silently dropped.
    assert "does_not_exist" in str(excinfo.value)


def test_resolve_lists_all_unknown_stages() -> None:
    with pytest.raises(InsightsError) as excinfo:
        _resolve(["nope_one", "nope_two"])
    msg = str(excinfo.value)
    assert "nope_one" in msg
    assert "nope_two" in msg


# --- finding 2: kb_snapshot ensures themes -----------------------------------


def test_kb_snapshot_ensures_themes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    summary = run_pipeline("drug discovery", ctx, stages=["kb_snapshot"])
    # Themes were discovered on demand even though no themes stage ran first.
    assert isinstance(summary["stages"], dict)
    assert summary["stages"]["kb_snapshot"]["themes"] >= 1
    assert ctx.themes  # ctx populated as a side effect of _ensure_themes
    snapshot = json.loads((tmp_path / "t" / "knowledge_base.json").read_text())
    assert len(snapshot["themes"]) >= 1
    assert len(snapshot["themes"]) == len(ctx.themes)


# --- finding 3: Stage no longer advertises deps ------------------------------


def test_stage_has_no_deps_field() -> None:
    field_names = {f.name for f in dataclasses.fields(Stage)}
    assert field_names == {"name", "fn"}
    assert not hasattr(Stage("x", lambda ctx, topic: {}), "deps")


# --- finding 4: project/memory records reflect ALL executed stages -----------


def test_project_record_lists_all_executed_stages(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # ``themes`` emits no .astra.json, so insight_counts alone would drop it.
    run_pipeline("drug discovery", ctx, stages=["themes", "project"])
    db = json.loads((tmp_path / "projects" / "project_database.json").read_text())
    assert len(db) == 1
    recorded = set(db[0]["stages"])
    assert "themes" in recorded
    assert "project" in recorded
    # Deterministic: created timestamp is left unset.
    assert db[0]["created"] is None


def test_memory_record_lists_all_executed_stages(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    run_pipeline("drug discovery", ctx, stages=["themes", "memory"])
    history = json.loads((tmp_path / "research_history.json").read_text())
    assert len(history) == 1
    recorded = set(history[0]["summary"]["stages"])
    assert "themes" in recorded
    assert "memory" in recorded
    assert history[0]["created"] is None
