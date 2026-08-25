"""The retrieval backend contract.

Consilium consumes a Perspicacité knowledge base through this Protocol. The
Phase-1 surface is intentionally small (build/select a KB, fetch passages, fetch
paper content); claims/citation-graph methods are added in later phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from consilium.config import Settings
    from consilium.models import Corpus, Passage


class RetrievalBackend(Protocol):
    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        """Build (or select an existing) knowledge base for ``topic``."""
        ...

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        """Return the top-``k`` passages relevant to ``query`` from the KB."""
        ...

    def paper_content(self, paper_id: str) -> str:
        """Return the full text (or best-available content) for a paper."""
        ...


def build_backend(settings: Settings, *, offline: bool = False) -> RetrievalBackend:
    """Return the configured backend. ``offline`` selects the in-memory fake."""
    from consilium.sources.fake import FakeBackend
    from consilium.sources.perspicacite import PerspicaciteBackend

    if offline:
        return FakeBackend()
    return PerspicaciteBackend(url=settings.perspicacite.url, timeout=settings.perspicacite.timeout)
