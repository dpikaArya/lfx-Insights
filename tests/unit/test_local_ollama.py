"""Tests for local Ollama configuration and validation."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from lfx_insights.config import Settings
from lfx_insights.errors import OllamaUnavailable
from lfx_insights.llm.client import LiteLLMClient, MockLLM, build_client, validate_ollama

pytestmark = pytest.mark.unit


class _FakeResponse:
    """Minimal httpx response mock."""

    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._data


class Foo(BaseModel):
    label: str
    n: int = 0


def test_validate_ollama_skips_non_ollama_model() -> None:
    """validate_ollama is a no-op for non-Ollama models."""
    settings = Settings()
    settings.llm.model = "gpt-4o"
    # Should not raise
    validate_ollama(settings)


def test_validate_ollama_raises_when_server_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_ollama raises OllamaUnavailable when Ollama is not running."""
    import httpx

    settings = Settings()
    settings.llm.model = "ollama/qwen2.5-coder:7b"

    def fake_get(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(OllamaUnavailable, match="Ollama is not reachable"):
        validate_ollama(settings)


def test_validate_ollama_raises_when_model_not_pulled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_ollama raises OllamaUnavailable when the model is not available."""
    import httpx

    settings = Settings()
    settings.llm.model = "ollama/nonexistent-model:latest"

    def fake_get(*args: object, **kwargs: object) -> object:
        return _FakeResponse({"models": [{"name": "qwen2.5-coder:7b"}]})

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(OllamaUnavailable, match="not available"):
        validate_ollama(settings)


def test_validate_ollama_passes_when_model_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_ollama succeeds when the model is available."""
    settings = Settings()
    settings.llm.model = "ollama/qwen2.5-coder:7b"

    def fake_get(*args: object, **kwargs: object) -> object:
        return _FakeResponse({"models": [{"name": "qwen2.5-coder:7b"}]})

    monkeypatch.setattr("httpx.get", fake_get)
    # Should not raise
    validate_ollama(settings)


def test_validate_ollama_handles_model_with_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_ollama matches models with exact tag."""
    settings = Settings()
    settings.llm.model = "ollama/qwen2.5-coder:7b"

    def fake_get(*args: object, **kwargs: object) -> object:
        # Server lists it with the same tag
        return _FakeResponse({"models": [{"name": "qwen2.5-coder:7b"}]})

    monkeypatch.setattr("httpx.get", fake_get)
    # Should not raise
    validate_ollama(settings)


def test_validate_ollama_rejects_tag_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_ollama rejects when the server has a different tag than requested."""
    settings = Settings()
    settings.llm.model = "ollama/qwen2.5-coder:7b"

    def fake_get(*args: object, **kwargs: object) -> object:
        # Server only has :latest, user wants :7b
        return _FakeResponse({"models": [{"name": "qwen2.5-coder:latest"}]})

    monkeypatch.setattr("httpx.get", fake_get)

    with pytest.raises(OllamaUnavailable, match="not available"):
        validate_ollama(settings)


def test_build_client_returns_litellm_for_ollama() -> None:
    """build_client returns LiteLLMClient when mock=False (for any model)."""
    settings = Settings()
    settings.llm.model = "ollama/qwen2.5-coder:7b"
    settings.llm.mock = False
    client = build_client(settings)
    assert isinstance(client, LiteLLMClient)


def test_build_client_returns_mock_when_configured() -> None:
    """build_client returns MockLLM when mock=True."""
    settings = Settings()
    settings.llm.mock = True
    client = build_client(settings)
    assert isinstance(client, MockLLM)


def test_no_anthropic_key_required_for_local_config() -> None:
    """Default local config must not reference any external provider model."""
    settings = Settings()
    model = settings.llm.model
    # Must start with ollama/ â€” not with anthropic/, openai/, gpt-, claude-, etc.
    assert model.startswith("ollama/"), f"Expected local model, got: {model}"
    # Fallback must be empty â€” no external fallback
    assert settings.llm.fallback == [], f"Expected empty fallback, got: {settings.llm.fallback}"


def test_local_embedding_config() -> None:
    """Default embedding is local sentence-transformers, not hosted."""
    settings = Settings()
    embedding_model = settings.embedding.model
    # Must be a local model, not an API model
    from lfx_insights.themes.discover import _is_api_embedding

    assert not _is_api_embedding(
        embedding_model
    ), f"Expected local embedding, got API model: {embedding_model}"
