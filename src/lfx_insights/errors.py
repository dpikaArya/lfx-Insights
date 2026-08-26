"""lfx Insights exception hierarchy."""

from __future__ import annotations


class InsightsError(Exception):
    """Base class for all lfx Insights errors."""


class ConfigError(InsightsError):
    """Raised when configuration is invalid or cannot be loaded."""


class PerspicaciteUnavailable(InsightsError):  # noqa: N818 - reads better without "Error"
    """Raised when the Perspicacité backend cannot be reached.

    lfx Insights never silently falls back to home-grown search or memory; it fails
    loudly so the user can start Perspicacité.
    """


class OllamaUnavailable(InsightsError):  # noqa: N818 - reads better without "Error"
    """Raised when the local Ollama server cannot be reached.

    lfx Insights never silently falls back to an external provider when Ollama is
    configured; it fails loudly so the user can start Ollama.
    """


class GroundingError(InsightsError):
    """Raised when a claim/citation cannot be grounded in the corpus."""
