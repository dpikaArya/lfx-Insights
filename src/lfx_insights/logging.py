"""structlog configuration."""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO") -> structlog.stdlib.BoundLogger:
    """Configure structlog and return a bound logger named ``consilium``."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # Logs go to stderr so stdout stays clean for CLI JSON output.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logger: structlog.stdlib.BoundLogger = structlog.get_logger("consilium")
    return logger
