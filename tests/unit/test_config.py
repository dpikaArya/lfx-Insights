from __future__ import annotations

from pathlib import Path

import pytest

from consilium.config import load_settings

pytestmark = pytest.mark.unit


def test_defaults() -> None:
    s = load_settings(None)
    assert s.llm.model == "ollama/qwen2.5-coder:7b"
    assert s.llm.fallback == []
    assert s.llm.ollama_base_url == "http://localhost:11434"
    assert s.perspicacite.url.endswith("/mcp")


def test_yaml_loaded(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text("llm:\n  model: my-model\noutput_dir: out2\n")
    s = load_settings(cfg)
    assert s.llm.model == "my-model"
    assert s.output_dir == "out2"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yml"
    cfg.write_text("llm:\n  model: from-yaml\n")
    monkeypatch.setenv("CONSILIUM_LLM__MODEL", "from-env")
    s = load_settings(cfg)
    assert s.llm.model == "from-env"


def test_local_model_configurable() -> None:
    """The local Ollama model can be changed via environment variable."""
    import os

    os.environ["CONSILIUM_LLM__MODEL"] = "ollama/llama3.2"
    try:
        s = load_settings(None)
        assert s.llm.model == "ollama/llama3.2"
    finally:
        del os.environ["CONSILIUM_LLM__MODEL"]


def test_no_fallback_by_default() -> None:
    """Default configuration has no external fallback chain."""
    s = load_settings(None)
    assert s.llm.fallback == []


def test_ollama_base_url_configurable() -> None:
    """Ollama base URL is configurable via settings."""
    s = load_settings(None)
    s.llm.ollama_base_url = "http://custom-host:11434"
    assert s.llm.ollama_base_url == "http://custom-host:11434"
