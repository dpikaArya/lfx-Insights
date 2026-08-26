from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from lfx_insights.cli import main
from lfx_insights.config import load_settings
from lfx_insights.context import build_context
from lfx_insights.pipeline import run as run_pipeline

pytestmark = pytest.mark.e2e

GEN = ["hypotheses", "questions", "manuscript", "grant", "review"]


def test_generation_pipeline_offline(tmp_path: Path) -> None:
    ctx = build_context(load_settings(None), offline=True, output_dir=str(tmp_path))
    summary = run_pipeline("drug discovery", ctx, stages=GEN)
    stages = summary["stages"]
    assert isinstance(stages, dict)
    for name in GEN:
        assert name in stages
    out = tmp_path / "default"
    for artifact in ("hypotheses.md", "questions.md", "manuscript.md", "grant.md", "review.md"):
        assert (out / artifact).exists()
    # hypotheses always export an indicium claim document (claims may be empty offline)
    assert (out / "hypotheses.indicium.json").exists()


@pytest.mark.parametrize("cmd", ["hypotheses", "questions", "manuscript", "grant", "review"])
def test_cli_generation_offline(tmp_path: Path, cmd: str) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [cmd, "--topic", "x", "--offline", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
