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

LS = ["study_design", "bioinformatics", "reproducibility", "datasets"]


def test_lifescience_pipeline_offline(tmp_path: Path) -> None:
    ctx = build_context(load_settings(None), offline=True, output_dir=str(tmp_path))
    summary = run_pipeline("drug discovery", ctx, stages=LS)
    stages = summary["stages"]
    assert isinstance(stages, dict)
    for name in LS:
        assert name in stages
        assert (tmp_path / "default" / f"{name}.md").exists()


def test_cli_stats_golden(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "stats",
            "--design",
            "two_sample_t",
            "--effect-size",
            "0.5",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_per_group"] == 64  # Cohen's classic d=0.5, alpha .05, power .80
    assert (tmp_path / "default" / "statistics.md").exists()


def test_cli_stats_bad_design_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["stats", "--design", "nope", "--effect-size", "0.5"])
    assert result.exit_code != 0


def test_cli_protocol(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["protocol", "--kind", "rna_seq", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "protocol_rna_seq.md").exists()


def test_cli_study_design_offline(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["study-design", "--topic", "x", "--offline", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "default" / "study_design.md").exists()
