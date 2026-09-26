"""Base abstraction for dense embedding providers."""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract base class for dense vector embedding generators."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a batch of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generates embedding for a single search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimensionality of the embeddings."""
        pass
