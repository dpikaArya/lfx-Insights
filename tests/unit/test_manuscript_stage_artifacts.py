from __future__ import annotations

import pytest

from consilium.models import SectionBundle

pytestmark = pytest.mark.unit


def test_manuscript_stage_writes_sections_json(tmp_path) -> None:
    from consilium.config import load_settings
    from consilium.context import build_context
    from consilium.pipeline import run as run_pipeline

    settings = load_settings(None)
    ctx = build_context(settings, offline=True, output_dir=str(tmp_path))
    run_pipeline("graph neural networks", ctx, stages=["manuscript"])

    artifact = ctx.store.path("manuscript.sections.json")
    assert artifact.exists()
    bundle = SectionBundle.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert bundle.title == "Manuscript Draft"
    assert isinstance(bundle.references, list)
