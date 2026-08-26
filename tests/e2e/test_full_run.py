from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from lfx_insights.cli import main
from lfx_insights.config import load_settings
from lfx_insights.context import build_context
from lfx_insights.pipeline import run as run_pipeline

pytestmark = pytest.mark.e2e


def test_full_quick_run_offline_produces_aggregation(tmp_path: Path) -> None:
    settings = load_settings(None)
    ctx = build_context(settings, offline=True, output_dir=str(tmp_path))
    summary = run_pipeline("drug discovery", ctx, stages=settings.pipeline.quick)
    stages = summary["stages"]
    assert isinstance(stages, dict)
    out = tmp_path / "default"
    # aggregation artifacts
    for artifact in (
        "dashboard.md",
        "brief.md",
        "explainability.md",
        "knowledge_base.json",
        "run.capsule.json",
    ):
        assert (out / artifact).exists(), artifact
    # cross-run state (written at the output-dir root, not the run dir)
    assert (tmp_path / "projects" / "project_database.json").exists()
    assert (tmp_path / "research_history.json").exists()
    # capsule conforms to the real asb-schema SciTaskCapsule slot names
    capsule = json.loads((out / "run.capsule.json").read_text())
    assert capsule["capsule_task_id"] == "drug-discovery"
    assert capsule["capsule_card"]["research_question"] == "drug discovery"
    # knowledge base bundles themes + artifacts
    kb = json.loads((out / "knowledge_base.json").read_text())
    assert kb["topic"] == "drug discovery"
    assert "themes" in kb and "artifacts" in kb


def test_memory_accumulates_across_runs(tmp_path: Path) -> None:
    settings = load_settings(None)
    for topic in ("topic one", "topic two"):
        ctx = build_context(settings, offline=True, output_dir=str(tmp_path))
        run_pipeline(topic, ctx, stages=["themes", "memory"])
    history = json.loads((tmp_path / "research_history.json").read_text())
    assert len(history) == 2
    assert {h["topic"] for h in history} == {"topic one", "topic two"}


def test_cli_dashboard_offline(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["dashboard", "--topic", "x", "--offline", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "dashboard.md").exists()
