"""Unit tests locking the ``run`` command's stage-selection logic.

These assert that ``--skip`` / ``--until`` are honored even when no stage-set
flag (``--only`` / ``--quick`` / ``--life-science``) is supplied: the command
defaults ``stages`` to the full pipeline before applying skip/until.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from consilium.cli import main
from consilium.pipeline import PIPELINE

pytestmark = pytest.mark.unit

_STAGE_NAMES = [s.name for s in PIPELINE]


def _run_summary(result_output: str) -> dict[str, object]:
    """Extract the trailing pretty-printed summary JSON from CLI stdout.

    The offline run also emits compact single-line structured log records to
    stdout; the command's summary is the only multi-line ``indent=2`` block and
    is echoed last, so we decode from the final top-level ``{`` to the end.
    """
    lines = result_output.splitlines()
    start = max(i for i, line in enumerate(lines) if line == "{")
    summary = json.loads("\n".join(lines[start:]))
    assert isinstance(summary, dict)
    return summary


def test_run_until_without_stage_flag_runs_only_until(tmp_path: Path) -> None:
    """``run --until themes`` (no --only/--quick/--life-science) runs only themes."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--topic",
            "drug discovery",
            "--until",
            "themes",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    stages = _run_summary(result.output)["stages"]
    assert isinstance(stages, dict)
    assert list(stages.keys()) == ["themes"]


def test_run_skip_without_stage_flag_excludes_skipped_stage(tmp_path: Path) -> None:
    """``run --skip themes`` (no stage flag) excludes themes from the full run."""
    runner = CliRunner()
    # Constrain with --until so the test stays fast while still proving --skip
    # is applied against the full pipeline (not silently ignored).
    result = runner.invoke(
        main,
        [
            "run",
            "--topic",
            "drug discovery",
            "--skip",
            "themes",
            "--until",
            "novelty",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    stages = _run_summary(result.output)["stages"]
    assert isinstance(stages, dict)
    assert "themes" not in stages
    # The stages before --until (minus the skipped one) still ran.
    assert "novelty" in stages
    assert "evidence_strength" in stages


def test_run_no_flags_defaults_to_full_pipeline(tmp_path: Path) -> None:
    """With no selection flags at all, every pipeline stage runs."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--topic",
            "drug discovery",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    stages = _run_summary(result.output)["stages"]
    assert isinstance(stages, dict)
    assert set(stages.keys()) == set(_STAGE_NAMES)
