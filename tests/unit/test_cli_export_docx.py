from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from lfx_insights.cli import main
from lfx_insights.models import GeneratedSection, Paper, SectionBundle

pytestmark = pytest.mark.unit

pytest.importorskip("docx")


def _write_artifact(run_dir: Path) -> None:
    bundle = SectionBundle(
        title="Manuscript Draft",
        sections=[GeneratedSection(name="introduction", text="Hello.", citations=["W1"])],
        references=[Paper(id="W1", title="Zeta", year=2020, source="Nature")],
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manuscript.sections.json").write_text(
        json.dumps(bundle.model_dump()), encoding="utf-8"
    )


def test_export_docx_happy_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_artifact(run_dir)
    result = CliRunner().invoke(
        main, ["export-docx", "--run", str(run_dir), "--artifact", "manuscript"]
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "manuscript.docx").exists()


def test_export_docx_missing_artifact_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["export-docx", "--run", str(tmp_path), "--artifact", "manuscript"]
    )
    assert result.exit_code != 0
    assert "manuscript.sections.json" in result.output
