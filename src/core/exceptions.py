"""Domain exceptions for the RAG chatbot system."""


class RAGException(Exception):
    """Base exception for all domain-specific errors in the RAG pipeline."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentIngestionError(RAGException):
    """Raised when loading or parsing a source document fails."""


class ChunkingError(RAGException):
    """Raised when semantic or structural chunking fails."""


class EmbeddingError(RAGException):
    """Raised when vector embedding generation fails."""


class VectorStoreError(RAGException):
    """Raised when vector database operations fail."""


class RetrievalError(RAGException):
    """Raised during retrieval or rank fusion failures."""


class LLMClientError(RAGException):
    """Raised when Sarvam AI or LLM API calls fail or return invalid responses."""


class ConfigurationError(RAGException):
    """Raised when configuration validation or environment variables are invalid."""
