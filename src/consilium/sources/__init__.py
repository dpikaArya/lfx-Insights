"""Retrieval backends. Consilium delegates all literature work to Perspicacité."""

from consilium.sources.base import RetrievalBackend, build_backend
from consilium.sources.fake import FakeBackend
from consilium.sources.perspicacite import PerspicaciteBackend

__all__ = ["FakeBackend", "PerspicaciteBackend", "RetrievalBackend", "build_backend"]
