from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from lfx_insights.config import Settings
from lfx_insights.llm.client import LiteLLMClient, MockLLM, build_client

pytestmark = pytest.mark.unit


class Foo(BaseModel):
    label: str
    n: int = 0


def test_litellm_parses_structured_output_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    calls = {"n": 0}

    def fake_completion(**kwargs: object) -> dict:
        calls["n"] += 1
        return {"choices": [{"message": {"content": json.dumps({"label": "hi", "n": 3})}}]}

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient(model="m", cache=True, cache_dir=tmp_path / "cache")

    out = client.complete_structured("a prompt", Foo)
    assert out.label == "hi" and out.n == 3
    assert calls["n"] == 1

    # Identical call hits the on-disk cache â€” no second provider call.
    again = client.complete_structured("a prompt", Foo)
    assert again.label == "hi"
    assert calls["n"] == 1


def test_litellm_falls_back_to_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)  # skip tenacity backoff
    seen: list[str] = []

    def fake_completion(model: str, **kwargs: object) -> dict:
        seen.append(model)
        if model == "bad":
            raise RuntimeError("provider down")
        return {"choices": [{"message": {"content": json.dumps({"label": "ok"})}}]}

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient(model="bad", fallback=["good"], cache=False)
    out = client.complete_structured("p", Foo)
    assert out.label == "ok"
    assert seen[0] == "bad" and "good" in seen


def test_litellm_all_models_fail_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    def fake_completion(**kwargs: object) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    client = LiteLLMClient(model="a", fallback=["b"], cache=False)
    with pytest.raises(RuntimeError, match="All LLM models failed"):
        client.complete_structured("p", Foo)


def test_build_client_selects_mock_or_real() -> None:
    mock_settings = Settings()
    mock_settings.llm.mock = True
    assert isinstance(build_client(mock_settings), MockLLM)

    real_settings = Settings()
    real_settings.llm.mock = False
    assert isinstance(build_client(real_settings), LiteLLMClient)
