"""
Logging utilities for semantic media search.
"""

import logging
import sys
from pathlib import Path

from src.config import LOG_LEVEL


def setup_logging(
    log_file: Path | None = None, level: str = LOG_LEVEL
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file. If None, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("semantic_media_search")
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"semantic_media_search.{name}")
