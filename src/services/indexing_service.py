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
        bm25_index: BM25Index | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_index = bm25_index
        self.chunker = chunker or QAStructuralChunker()

    async def aindex_document(self, file_path: Path | str, clear_existing: bool = False) -> list[DocumentChunk]:
        """Asynchronously ingests, chunks, embeds, and indexes a single document file."""
        import asyncio
        path = Path(file_path)
        if not path.exists():
            raise DocumentIngestionError(f"Document file does not exist: {path}")

        logger.info(f"Starting async ingestion for file: '{path.name}'...")
        loader = DocumentLoaderFactory.get_loader(path)
        raw_text = await asyncio.to_thread(loader.load, path)

        logger.info(f"Extracted {len(raw_text)} characters. Chunking document...")
        chunks = self.chunker.chunk(raw_text, source_doc=path.name)

        if clear_existing:
            logger.info("Clearing existing vector store...")
            if hasattr(self.vector_store, "aclear"):
                await self.vector_store.aclear()
            else:
                self.vector_store.clear()

        # Generate dense embeddings without blocking event loop
        logger.info(f"Generating embeddings for {len(chunks)} chunks in threadpool...")
        texts_to_embed = [c.text for c in chunks]
        embeddings = await asyncio.to_thread(self.embedding_provider.embed_documents, texts_to_embed)

        # Store in Vector DB
        logger.info("Upserting into vector store...")
        if hasattr(self.vector_store, "aadd_chunks"):
            await self.vector_store.aadd_chunks(chunks, embeddings)
        else:
            self.vector_store.add_chunks(chunks, embeddings)

        if self.bm25_index is not None:
            logger.info("Building BM25 sparse index...")
            self.bm25_index.index_chunks(chunks)

        logger.info(f"Async indexing completed successfully! {len(chunks)} chunks indexed.")
        return chunks

    def index_document(self, file_path: Path | str, clear_existing: bool = False) -> list[DocumentChunk]:
        """Synchronous wrapper for index_document."""
        import asyncio
        return asyncio.run(self.aindex_document(file_path, clear_existing=clear_existing))



