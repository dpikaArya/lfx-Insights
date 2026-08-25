"""RunContext: the shared dependency bundle passed through pipeline stages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consilium.config import Settings
    from consilium.io.store import OutputStore
    from consilium.llm.client import LLMClient
    from consilium.models import Corpus, GeneratedSection, Theme
    from consilium.sources.base import RetrievalBackend
    from consilium.themes.discover import Embedder


@dataclass
class RunContext:
    settings: Settings
    backend: RetrievalBackend
    llm: LLMClient
    embedder: Embedder
    store: OutputStore
    log: Any
    corpus: Corpus | None = None
    themes: list[Theme] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    manuscript: list[GeneratedSection] = field(default_factory=list)


def slugify(text: str) -> str:
    """Filesystem-safe slug for run directory names (mirrors other modules' ``_slug``)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64] or "default"


def build_context(
    settings: Settings,
    *,
    offline: bool = False,
    output_dir: str | None = None,
    run: str | None = None,
    topic: str | None = None,
) -> RunContext:
    """Construct a RunContext from settings (real backends, or offline fakes).

    The run directory namespaces all artifacts so distinct topics do not collide
    under ``outputs/default/``. ``run`` wins when explicitly given; otherwise it is
    derived from ``topic`` via :func:`slugify`; otherwise it falls back to
    ``"default"``.
    """
    from consilium.io.store import OutputStore
    from consilium.llm.client import MockLLM, build_client, validate_ollama
    from consilium.logging import configure
    from consilium.sources.base import build_backend
    from consilium.themes.discover import SimpleEmbedder, default_embedder

    if run is None:
        run = slugify(topic) if topic is not None else "default"

    log = configure()
    backend = build_backend(settings, offline=offline)
    llm: LLMClient
    embedder: Embedder
    if offline:
        llm = MockLLM()
        embedder = SimpleEmbedder()
    else:
        validate_ollama(settings)
        llm = build_client(settings)
        embedder = default_embedder(settings)
    store = OutputStore(output_dir or settings.output_dir, run=run)
    return RunContext(
        settings=settings,
        backend=backend,
        llm=llm,
        embedder=embedder,
        store=store,
        log=log,
    )
