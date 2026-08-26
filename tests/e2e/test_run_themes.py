from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from lfx_insights.cli import main
from lfx_insights.context import RunContext
from lfx_insights.io.store import OutputStore
from lfx_insights.llm.client import MockLLM
from lfx_insights.logging import configure
from lfx_insights.pipeline import run as run_pipeline
from lfx_insights.sources.fake import FakeBackend
from lfx_insights.themes.discover import SimpleEmbedder
from lfx_insights.themes.label import ThemeLabel

pytestmark = pytest.mark.e2e


def test_run_themes_produces_artifacts(tmp_path: Path) -> None:
    ctx = RunContext(
        settings=None,  # type: ignore[arg-type]  # stages here don't read settings
        backend=FakeBackend(),
        llm=MockLLM(responder=lambda p, m: ThemeLabel(label="Topic", rationale="r")),
        embedder=SimpleEmbedder(),
        store=OutputStore(tmp_path, run="t"),
        log=configure(),
    )
    summary = run_pipeline("drug discovery", ctx, stages=["themes"])
    assert isinstance(summary["stages"], dict)
    assert summary["stages"]["themes"]["themes"] >= 1
    md = (tmp_path / "t" / "themes.md").read_text()
    assert "# Theme Discovery" in md
    assert (tmp_path / "t" / "indicium_sources.json").exists()


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_cli_themes_offline(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["themes", "--topic", "drug discovery", "--offline", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "themes.md").exists()


def test_cli_run_offline_quick(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--topic", "drug discovery", "--quick", "--offline", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "themes.md").exists()


def test_cli_run_offline_only_and_until(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--topic",
            "drug discovery",
            "--only",
            "themes",
            "--until",
            "themes",
            "--skip",
            "nonexistent",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
