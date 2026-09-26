"""Embeddings package exports."""

from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings

__all__ = [
    "BaseEmbeddingProvider",
    "LocalSentenceTransformerEmbeddings",
]
