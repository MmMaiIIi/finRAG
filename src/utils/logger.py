"""Logging configuration helpers."""

from __future__ import annotations

import logging


def setup_logger(name: str, level: int | str = "INFO") -> logging.Logger:
    """Create or get a logger with a standard console handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger

