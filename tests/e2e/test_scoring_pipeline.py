from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from lfx_insights.cli import main
from lfx_insights.config import load_settings
from lfx_insights.context import build_context
from lfx_insights.pipeline import run as run_pipeline

pytestmark = pytest.mark.e2e

SCORING = ["themes", "evidence_strength", "novelty", "opportunity", "funding", "meta_analysis"]


def test_scoring_pipeline_offline(tmp_path: Path) -> None:
    ctx = build_context(load_settings(None), offline=True, output_dir=str(tmp_path))
    summary = run_pipeline("drug discovery", ctx, stages=SCORING)
    stages = summary["stages"]
    assert isinstance(stages, dict)
    for name in SCORING:
        assert name in stages
    for artifact in ("evidence_strength", "novelty", "opportunities", "funding", "meta_analysis"):
        assert (tmp_path / "default" / f"{artifact}.md").exists()
        assert (tmp_path / "default" / f"{artifact}.astra.json").exists()


def test_gaps_stage_offline(tmp_path: Path) -> None:
    ctx = build_context(load_settings(None), offline=True, output_dir=str(tmp_path))
    ctx.gaps = ["no work exists on quantum molecular docking"]
    summary = run_pipeline("drug discovery", ctx, stages=["gaps"])
    stages = summary["stages"]
    assert isinstance(stages, dict)
    assert stages["gaps"]["insights"] == 1
    assert (tmp_path / "default" / "gaps.md").exists()


def test_gaps_stage_skips_without_input(tmp_path: Path) -> None:
    ctx = build_context(load_settings(None), offline=True, output_dir=str(tmp_path))
    summary = run_pipeline("drug discovery", ctx, stages=["gaps"])
    stages = summary["stages"]
    assert isinstance(stages, dict)
    assert "skipped" in stages["gaps"]


def test_cli_novelty_offline(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["novelty", "--topic", "x", "--offline", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "novelty.md").exists()


def test_cli_gaps_offline(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "gaps",
            "--topic",
            "x",
            "--gap",
            "no work on quantum docking",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "gaps.md").exists()
