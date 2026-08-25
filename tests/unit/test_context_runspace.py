"""Unit tests for run-directory namespacing in :mod:`consilium.context`.

Distinct topics must land in distinct run directories so artifacts never collide
under ``outputs/default/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from consilium.config import load_settings
from consilium.context import build_context, slugify

pytestmark = pytest.mark.unit


def test_slugify_basic() -> None:
    assert slugify("Drug Discovery") == "drug-discovery"


def test_slugify_collapses_and_strips_non_alnum() -> None:
    assert slugify("  CRISPR / Cas9: gene-editing!! ") == "crispr-cas9-gene-editing"


def test_slugify_truncates_to_64_chars() -> None:
    long = "a" * 200
    assert slugify(long) == "a" * 64


def test_slugify_empty_falls_back_to_default() -> None:
    assert slugify("") == "default"
    assert slugify("!!!") == "default"


def test_two_topics_get_different_run_dirs(tmp_path: Path) -> None:
    settings = load_settings(None)
    ctx_a = build_context(settings, offline=True, output_dir=str(tmp_path), topic="drug discovery")
    ctx_b = build_context(settings, offline=True, output_dir=str(tmp_path), topic="protein folding")

    assert ctx_a.store.run_dir != ctx_b.store.run_dir
    assert ctx_a.store.run_dir == tmp_path / "drug-discovery"
    assert ctx_b.store.run_dir == tmp_path / "protein-folding"


def test_topic_drives_run_when_run_not_given(tmp_path: Path) -> None:
    settings = load_settings(None)
    ctx = build_context(settings, offline=True, output_dir=str(tmp_path), topic="Genome Assembly")
    assert ctx.store.run_dir == tmp_path / "genome-assembly"


def test_explicit_run_wins_over_topic(tmp_path: Path) -> None:
    settings = load_settings(None)
    ctx = build_context(
        settings,
        offline=True,
        output_dir=str(tmp_path),
        run="custom-run",
        topic="drug discovery",
    )
    assert ctx.store.run_dir == tmp_path / "custom-run"


def test_no_topic_no_run_falls_back_to_default(tmp_path: Path) -> None:
    settings = load_settings(None)
    ctx = build_context(settings, offline=True, output_dir=str(tmp_path))
    assert ctx.store.run_dir == tmp_path / "default"
