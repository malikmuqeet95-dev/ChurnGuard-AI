import logging

import pytest

from backend.app.logging_config import configure_logging, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_configure_logging_accepts_valid_level():
    configure_logging("INFO")

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO


def test_configure_logging_accepts_debug():
    configure_logging("DEBUG")

    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError):
        configure_logging("INVALID_LEVEL")


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")

    root_logger = logging.getLogger()

    churnguard_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_churnguard_handler", False)
    ]

    assert len(churnguard_handlers) == 1