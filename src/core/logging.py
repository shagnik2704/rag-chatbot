"""Enterprise structured logging configuration."""

import logging
import sys


def setup_logger(name: str = "rag_chatbot", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured, formatted logger.

    Args:
        name: Logger name, defaults to 'rag_chatbot'.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger instance configured with consistent format.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
