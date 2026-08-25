"""Theme discovery and labeling."""

from consilium.themes.discover import (
    Embedder,
    SimpleEmbedder,
    default_embedder,
    discover_themes,
)
from consilium.themes.label import ThemeLabel, label_themes

__all__ = [
    "Embedder",
    "SimpleEmbedder",
    "ThemeLabel",
    "default_embedder",
    "discover_themes",
    "label_themes",
]
