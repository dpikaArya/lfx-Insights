"""LLM client: a structured-output interface with a real litellm impl and a
deterministic mock used in tests and ``--offline`` runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from collections.abc import Callable

    from lfx_insights.config import Settings

T = TypeVar("T", bound=BaseModel)

CACHE_DIR = Path(".consilium_cache")


class _PlainText(BaseModel):
    text: str = ""


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated pydantic object."""

    def complete_structured(
        self, prompt: str, response_model: type[T], *, temperature: float | None = None
    ) -> T: ...

    def complete(self, prompt: str, *, temperature: float | None = None) -> str: ...


def _minimal_instance[ModelT: BaseModel](response_model: type[ModelT]) -> ModelT:
    """Best-effort construction of a model with only placeholder values.

    Used by MockLLM when no canned response matches â€” lets ``--offline`` runs
    complete end-to-end without a provider.
    """
    kwargs: dict[str, object] = {}
    for name, field in response_model.model_fields.items():
        if not field.is_required():
            continue
        ann = field.annotation
        if ann is str:
            kwargs[name] = f"mock-{name}"
        elif ann is int:
            kwargs[name] = 0
        elif ann is float:
            kwargs[name] = 0.0
        elif ann is bool:
            kwargs[name] = False
        elif ann is list or getattr(ann, "__origin__", None) is list:
            kwargs[name] = []
        else:
            kwargs[name] = None
    return response_model(**kwargs)


class MockLLM:
    """Deterministic LLM for tests/offline.

    Resolution order: ``responder`` callable -> substring match in ``responses``
    -> minimal auto-constructed instance.
    """

    def __init__(
        self,
        responses: dict[str, BaseModel] | None = None,
        responder: Callable[[str, type[BaseModel]], BaseModel] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.responder = responder
        self.calls: list[str] = []

    def complete_structured(
        self, prompt: str, response_model: type[T], *, temperature: float | None = None
    ) -> T:
        self.calls.append(prompt)
        if self.responder is not None:
            result = self.responder(prompt, response_model)
            return response_model.model_validate(result.model_dump())
        for key, value in self.responses.items():
            if key in prompt and isinstance(value, response_model):
                return value
        return _minimal_instance(response_model)

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        result = self.complete_structured(prompt, _PlainText, temperature=temperature)
        return result.text if result.text else "mock response"


class LiteLLMClient:
    """litellm-backed structured completion with retries, fallback chain, and disk cache."""

    def __init__(
        self,
        model: str,
        fallback: list[str] | None = None,
        temperature: float = 0.2,
        cache: bool = True,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.model = model
        self.fallback = fallback or []
        self.temperature = temperature
        self.cache = cache
        self.cache_dir = cache_dir

    def _cache_key(self, model: str, prompt: str, schema: dict[str, object]) -> str:
        blob = json.dumps({"m": model, "p": prompt, "s": schema}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete_structured(
        self, prompt: str, response_model: type[T], *, temperature: float | None = None
    ) -> T:
        schema = response_model.model_json_schema()
        key = self._cache_key(self.model, prompt, schema)
        cache_path = self.cache_dir / f"{key}.json"
        if self.cache and cache_path.exists():
            return response_model.model_validate_json(cache_path.read_text())

        content = self._call_with_fallbacks(prompt, schema, temperature)
        obj = response_model.model_validate_json(content)
        if self.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(obj.model_dump_json())
        return obj

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        result = self.complete_structured(prompt, _PlainText, temperature=temperature)
        return result.text

    def _call_with_fallbacks(
        self, prompt: str, schema: dict[str, object], temperature: float | None
    ) -> str:
        models = [self.model, *self.fallback]
        last_err: Exception | None = None
        for model in models:
            try:
                return self._one_call(model, prompt, schema, temperature)
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"All LLM models failed: {models}") from last_err

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=20))
    def _one_call(
        self, model: str, prompt: str, schema: dict[str, object], temperature: float | None
    ) -> str:
        import litellm

        system = (
            f"Return ONLY a JSON object that conforms to this JSON schema:\n{json.dumps(schema)}"
        )
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
            response_format={"type": "json_object"},
        )
        return str(resp["choices"][0]["message"]["content"])


def build_client(settings: Settings) -> LLMClient:
    if settings.llm.mock:
        return MockLLM()
    return LiteLLMClient(
        model=settings.llm.model,
        fallback=settings.llm.fallback,
        temperature=settings.llm.temperature,
        cache=settings.llm.cache,
    )


def validate_ollama(settings: Settings) -> None:
    """Verify that the Ollama server is reachable when an Ollama model is configured.

    Raises :class:`~consilium.errors.OllamaUnavailable` with an actionable message
    if the endpoint is down or the model is not pulled.
    """
    from lfx_insights.errors import OllamaUnavailable

    model = settings.llm.model
    if not model.lower().startswith("ollama/"):
        return

    base_url = settings.llm.ollama_base_url.rstrip("/")
    model_name = model.split("/", 1)[1] if "/" in model else model

    # 1. Check Ollama server is reachable
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise OllamaUnavailable(
            f"Ollama is not reachable at {base_url}. "
            "Start Ollama and retry. "
            "Install: https://ollama.com/download  |  "
            "Start: ollama serve"
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(
            f"Ollama returned an error at {base_url}: {exc}"
        ) from exc

    # 2. Check the configured model is available
    try:
        models_data = resp.json()
        available = {m.get("name", "") for m in models_data.get("models", [])}
        # Normalize for comparison: if the server lists a model without a tag
        # (e.g. "qwen2.5-coder"), treat it as ":latest" for matching.
        normalized_available: set[str] = set()
        for name in available:
            normalized_available.add(name)
            if ":" not in name:
                normalized_available.add(f"{name}:latest")
        if model_name in normalized_available:
            return
        raise OllamaUnavailable(
            f"Ollama model '{model_name}' is not available. "
            f"Available models: {', '.join(sorted(available)) or '(none)'}. "
            f"Pull it with: ollama pull {model_name}"
        )
    except (ValueError, KeyError):
        # If we can't parse the model list, the server is up but we can't
        # verify the model â€” let LiteLLM handle the error at call time.
        pass
