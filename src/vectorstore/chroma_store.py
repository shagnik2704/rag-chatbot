"""ChromaDB implementation of BaseVectorStore."""

from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.exceptions import VectorStoreError
from src.core.logging import setup_logger
from src.models.chunk import ChunkMetadata, DocumentChunk
from src.vectorstore.base import BaseVectorStore

logger = setup_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store with metadata filtering."""

    def __init__(
        self,
        persist_dir: Path | str = "storage/chroma_db",
        collection_name: str = "future_ready_faq",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Initialized ChromaDB at '{self.persist_dir}' | Collection: '{self.collection_name}' | Chunks: {self._collection.count()}"
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}") from e

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        try:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata.to_flat_dict() for chunk in chunks]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks into ChromaDB: {e}") from e

    def similarity_search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 4,
        where_filter: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        try:
            count = self._collection.count()
            if count == 0:
                return []

            actual_k = min(top_k, count)
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=actual_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            hits: list[tuple[DocumentChunk, float]] = []
            if not results or not results["ids"] or not results["ids"][0]:
                return hits

            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                chunk_id = ids[i]
                doc_text = docs[i]
                meta_dict = metas[i]
                dist = distances[i] if i < len(distances) else 1.0

                # Convert cosine distance to similarity score
                similarity = max(0.0, 1.0 - dist)
                metadata = ChunkMetadata.from_flat_dict(meta_dict)

                chunk = DocumentChunk(
                    id=chunk_id,
                    text=doc_text,
                    metadata=metadata,
                )
                hits.append((chunk, similarity))

            return hits
        except Exception as e:
            raise VectorStoreError(f"Failed during vector search: {e}") from e

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection '{self.collection_name}' in ChromaDB.")
        except Exception as e:
            raise VectorStoreError(f"Failed to clear collection: {e}") from e
