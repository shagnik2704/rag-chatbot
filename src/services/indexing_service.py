"""Service for end-to-end ingestion, structural chunking, and dual indexing."""

from pathlib import Path
from src.core.exceptions import DocumentIngestionError
from src.core.logging import setup_logger
from src.embeddings.base import BaseEmbeddingProvider
from src.ingestion.loaders import DocumentLoaderFactory
from src.ingestion.qa_chunker import BaseChunker, QAStructuralChunker
from src.models.chunk import DocumentChunk
from src.retrieval.bm25_index import BM25Index
from src.vectorstore.base import BaseVectorStore

logger = setup_logger(__name__)


class IndexingService:
    """Manages document ingestion, chunking, and population of vector and lexical indexes."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        bm25_index: BM25Index,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_index = bm25_index
        self.chunker = chunker or QAStructuralChunker()

    def index_document(self, file_path: Path | str, clear_existing: bool = False) -> list[DocumentChunk]:
        """Ingests, chunks, embeds, and indexes a single document file."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentIngestionError(f"Document file does not exist: {path}")

        logger.info(f"Starting ingestion for file: '{path.name}'...")
        loader = DocumentLoaderFactory.get_loader(path)
        raw_text = loader.load(path)

        logger.info(f"Extracted {len(raw_text)} characters. Chunking document...")
        chunks = self.chunker.chunk(raw_text, source_doc=path.name)

        if clear_existing:
            logger.info("Clearing existing vector store collection...")
            self.vector_store.clear()

        # Generate dense embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts_to_embed)

        # Store in Vector DB
        logger.info("Upserting into ChromaDB vector store...")
        self.vector_store.add_chunks(chunks, embeddings)

        # Build BM25 Lexical Index
        logger.info("Building BM25 sparse index...")
        self.bm25_index.index_chunks(chunks)

        logger.info(f"Indexing completed successfully! {len(chunks)} chunks indexed.")
        return chunks
