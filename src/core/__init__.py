"""Core package containing configuration, exceptions, and logging."""

from src.core.config import Settings, get_settings
from src.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    DocumentIngestionError,
    EmbeddingError,
    LLMClientError,
    RAGException,
    RetrievalError,
    VectorStoreError,
)
from src.core.logging import setup_logger

__all__ = [
    "Settings",
    "get_settings",
    "RAGException",
    "DocumentIngestionError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "LLMClientError",
    "ConfigurationError",
    "setup_logger",
]
