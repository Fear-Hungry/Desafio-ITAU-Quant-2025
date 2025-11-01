"""Minimal structured logging helpers for the project."""

from __future__ import annotations

import logging
from typing import Mapping

DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def configure_logging(level: int = logging.INFO, *, fmt: str = DEFAULT_FORMAT) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        logging.getLogger().setLevel(level)


def get_logger(name: str, *, level: int | None = None) -> logging.Logger:
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def log_dict(
    logger: logging.Logger,
    message: str,
    payload: Mapping[str, object],
    level: int = logging.INFO,
) -> None:
    serialised = ", ".join(f"{key}={value}" for key, value in payload.items())
    logger.log(level, "%s | %s", message, serialised)
