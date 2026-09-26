"""Hybrid retriever combining Dense Vector Search and BM25 Sparse Search via RRF."""

import asyncio
from src.core.exceptions import RetrievalError
from src.core.logging import setup_logger
from src.embeddings.base import BaseEmbeddingProvider
from src.models.chunk import DocumentChunk
from src.models.query import RetrievedChunk
from src.retrieval.bm25_index import BM25Index
from src.vectorstore.base import BaseVectorStore

logger = setup_logger(__name__)


class HybridRetriever:
    """Orchestrates hybrid dense + sparse retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        bm25_index: BM25Index,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    async def aretrieve(
        self,
        query: str,
        top_k: int = 4,
        section_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Asynchronously executes dual retrieval concurrently using asyncio.gather."""
        if not query.strip():
            return []

        try:
            candidate_k = max(top_k * 3, 10)
            where_filter = {"section": section_filter} if section_filter else None

            # Generate query embedding in threadpool to not block event loop
            query_vector = await asyncio.to_thread(self.embedding_provider.embed_query, query)

            # Concurrent execution of dense and sparse search
            dense_task = asyncio.to_thread(
                self.vector_store.similarity_search_by_vector,
                query_vector=query_vector,
                top_k=candidate_k,
                where_filter=where_filter,
            )
            sparse_task = asyncio.to_thread(
                self.bm25_index.search,
                query=query,
                top_k=candidate_k,
            )

            dense_results, sparse_raw_results = await asyncio.gather(dense_task, sparse_task)

            if section_filter:
                sparse_results = [
                    (c, s) for c, s in sparse_raw_results if c.metadata.section == section_filter
                ]
            else:
                sparse_results = sparse_raw_results

            return self._fuse_results(dense_results, sparse_results, top_k, query)
        except Exception as e:
            raise RetrievalError(f"Async hybrid retrieval failed: {e}") from e

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        section_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Executes synchronous dual retrieval and fuses candidate ranks."""
        if not query.strip():
            return []

        try:
            candidate_k = max(top_k * 3, 10)
            query_vector = self.embedding_provider.embed_query(query)
            where_filter = {"section": section_filter} if section_filter else None
            dense_results = self.vector_store.similarity_search_by_vector(
                query_vector=query_vector,
                top_k=candidate_k,
                where_filter=where_filter,
            )
            sparse_raw_results = self.bm25_index.search(query=query, top_k=candidate_k)
            if section_filter:
                sparse_results = [
                    (c, s) for c, s in sparse_raw_results if c.metadata.section == section_filter
                ]
            else:
                sparse_results = sparse_raw_results

            return self._fuse_results(dense_results, sparse_results, top_k, query)
        except Exception as e:
            raise RetrievalError(f"Hybrid retrieval failed: {e}") from e

    def _fuse_results(
        self,
        dense_results: list[tuple[DocumentChunk, float]],
        sparse_results: list[tuple[DocumentChunk, float]],
        top_k: int,
        query: str,
    ) -> list[RetrievedChunk]:
        # Reciprocal Rank Fusion (RRF)
        chunk_map: dict[str, DocumentChunk] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_map[chunk.id] = chunk
            dense_scores[chunk.id] = score
            dense_ranks[chunk.id] = rank

        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.id] = chunk
            sparse_scores[chunk.id] = score
            sparse_ranks[chunk.id] = rank

        fused_candidates: list[tuple[str, float]] = []
        for chunk_id in chunk_map:
            d_rank = dense_ranks.get(chunk_id, 1000)
            s_rank = sparse_ranks.get(chunk_id, 1000)

            rrf_score = (self.dense_weight * (1.0 / (self.rrf_k + d_rank))) + (
                self.sparse_weight * (1.0 / (self.rrf_k + s_rank))
            )
            fused_candidates.append((chunk_id, rrf_score))

        # Sort by RRF score descending
        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        retrieved: list[RetrievedChunk] = []
        for final_rank, (chunk_id, rrf_score) in enumerate(fused_candidates[:top_k], start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk_map[chunk_id],
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    sparse_score=sparse_scores.get(chunk_id, 0.0),
                    rrf_score=rrf_score,
                    rank=final_rank,
                )
            )

        logger.info(
            f"Retrieved {len(retrieved)} chunks for query '{query[:40]}...' (Dense hits: {len(dense_results)}, BM25 hits: {len(sparse_results)})"
        )
        return retrieved
