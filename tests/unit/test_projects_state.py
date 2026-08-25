"""Unit tests for :mod:`consilium.projects` persistent state.

The persistent run history (:func:`record_run`) and project database
(:func:`record_project`) must append across runs and recover from a
pre-existing corrupt JSON file without raising. Each write is atomic, so a
corrupt or missing file is never made worse by a record call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consilium.projects.project_manager import record_project
from consilium.projects.research_memory import record_run

pytestmark = pytest.mark.unit


def _read_history(path: Path) -> list[dict[str, object]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return loaded


def test_record_run_appends_across_two_runs(tmp_path: Path) -> None:
    path = record_run(tmp_path, "drug discovery", {"score": 1}, created="2026-06-12")
    record_run(tmp_path, "protein folding", {"score": 2}, created="2026-06-13")

    assert path == tmp_path / "research_history.json"
    history = _read_history(path)
    assert len(history) == 2
    assert history[0] == {
        "topic": "drug discovery",
        "summary": {"score": 1},
        "created": "2026-06-12",
    }
    assert history[1] == {
        "topic": "protein folding",
        "summary": {"score": 2},
        "created": "2026-06-13",
    }


def test_record_run_recovers_from_corrupt_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "research_history.json"
    path.write_text("{ this is not valid json", encoding="utf-8")

    returned = record_run(tmp_path, "genome assembly", {"ok": True})

    assert returned == path
    history = _read_history(path)
    assert len(history) == 1
    assert history[0] == {
        "topic": "genome assembly",
        "summary": {"ok": True},
        "created": None,
    }


def test_record_run_ignores_non_list_json(tmp_path: Path) -> None:
    path = tmp_path / "research_history.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    record_run(tmp_path, "themes", {})

    history = _read_history(path)
    assert len(history) == 1
    assert history[0]["topic"] == "themes"


def test_record_run_leaves_no_temp_file(tmp_path: Path) -> None:
    record_run(tmp_path, "topic", {})

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_record_project_appends_across_two_runs(tmp_path: Path) -> None:
    path = record_project(
        tmp_path,
        "drug discovery",
        {"kb_id": "kb-1", "stages": {"themes": {}, "gaps": {}}},
        created="2026-06-12",
    )
    record_project(
        tmp_path,
        "protein folding",
        {"kb_id": "kb-2", "stages": {"hypotheses": {}}},
        created="2026-06-13",
    )

    assert path == tmp_path / "projects" / "project_database.json"
    db = _read_history(path)
    assert len(db) == 2
    assert db[0] == {
        "topic": "drug discovery",
        "kb_id": "kb-1",
        "stages": ["themes", "gaps"],
        "created": "2026-06-12",
    }
    assert db[1] == {
        "topic": "protein folding",
        "kb_id": "kb-2",
        "stages": ["hypotheses"],
        "created": "2026-06-13",
    }


def test_record_project_recovers_from_corrupt_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "projects" / "project_database.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json", encoding="utf-8")

    returned = record_project(tmp_path, "genome assembly", {"kb_id": "kb-9"})

    assert returned == path
    db = _read_history(path)
    assert len(db) == 1
    assert db[0] == {
        "topic": "genome assembly",
        "kb_id": "kb-9",
        "stages": [],
        "created": None,
    }


def test_record_project_ignores_non_list_json(tmp_path: Path) -> None:
    path = tmp_path / "projects" / "project_database.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    record_project(tmp_path, "themes", {})

    db = _read_history(path)
    assert len(db) == 1
    assert db[0]["topic"] == "themes"


def test_record_project_leaves_no_temp_file(tmp_path: Path) -> None:
    path = record_project(tmp_path, "topic", {})

    leftovers = list(path.parent.glob("*.tmp"))
    assert leftovers == []
