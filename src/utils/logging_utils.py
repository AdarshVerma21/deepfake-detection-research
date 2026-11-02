"""Logging helpers for reproducible experiment tracing."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(path / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
