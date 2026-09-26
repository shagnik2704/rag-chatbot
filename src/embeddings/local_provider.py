"""Local sentence-transformers embedding provider."""

from sentence_transformers import SentenceTransformer
from src.core.exceptions import EmbeddingError
from src.core.logging import setup_logger
from src.embeddings.base import BaseEmbeddingProvider

logger = setup_logger(__name__)


class LocalSentenceTransformerEmbeddings(BaseEmbeddingProvider):
    """Local embedding provider using SentenceTransformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        logger.info(f"Loading local embedding model: '{model_name}'...")
        try:
            self._model = SentenceTransformer(model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = self._model.get_embedding_dimension()
            else:
                self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self._dim}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate document embeddings: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        if not text:
            raise EmbeddingError("Query text cannot be empty.")
        try:
            embedding = self._model.encode(
                text,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}") from e

    @property
    def dimension(self) -> int:
        return self._dim
