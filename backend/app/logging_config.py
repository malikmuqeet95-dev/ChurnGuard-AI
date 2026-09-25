from __future__ import annotations

import logging
import sys
from typing import Optional


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide structured logging.

    This function is intentionally idempotent so it can safely
    be called multiple times during application startup or tests.
    """

    normalized_level = str(log_level).upper()

    numeric_level = getattr(logging, normalized_level, None)

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level '{log_level}'. "
            "Expected DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    root_logger = logging.getLogger()

    formatter = logging.Formatter(LOG_FORMAT)

    handler: Optional[logging.Handler] = None

    for existing_handler in root_logger.handlers:
        if getattr(existing_handler, "_churnguard_handler", False):
            handler = existing_handler
            break

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler._churnguard_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)

    handler.setFormatter(formatter)

    root_logger.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named application logger.

    Example:
        logger = get_logger(__name__)
    """

    return logging.getLogger(name)