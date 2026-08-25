"""Consilium exception hierarchy."""

from __future__ import annotations


class ConsiliumError(Exception):
    """Base class for all Consilium errors."""


class ConfigError(ConsiliumError):
    """Raised when configuration is invalid or cannot be loaded."""


class PerspicaciteUnavailable(ConsiliumError):  # noqa: N818 - reads better without "Error"
    """Raised when the Perspicacité backend cannot be reached.

    Consilium never silently falls back to home-grown search or memory; it fails
    loudly so the user can start Perspicacité.
    """


class OllamaUnavailable(ConsiliumError):  # noqa: N818 - reads better without "Error"
    """Raised when the local Ollama server cannot be reached.

    Consilium never silently falls back to an external provider when Ollama is
    configured; it fails loudly so the user can start Ollama.
    """


class GroundingError(ConsiliumError):
    """Raised when a claim/citation cannot be grounded in the corpus."""
