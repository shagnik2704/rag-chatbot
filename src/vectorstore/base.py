"""Abstract base class for vector store implementations."""

from abc import ABC, abstractmethod
from typing import Any
from src.models.chunk import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and similarity search."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Stores chunks alongside their dense vector embeddings."""
        pass

    @abstractmethod
    def similarity_search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 4,
        where_filter: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieves top_k most similar chunks for a given query vector.

        Returns:
            List of tuples: (DocumentChunk, similarity_score)
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the number of indexed chunks in the vector store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all records from the vector store."""
        pass
