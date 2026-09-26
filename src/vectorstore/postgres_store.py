"""PostgreSQL + pgvector implementation of vector storage and native SQL hybrid retrieval."""

import asyncio
from typing import Any
import asyncpg

from src.core.exceptions import VectorStoreError
from src.core.logging import setup_logger
from src.db.connection import get_db_pool
from src.models.chunk import ChunkMetadata, DocumentChunk
from src.models.query import RetrievedChunk
from src.vectorstore.base import BaseVectorStore

logger = setup_logger(__name__)


class PostgresVectorStore(BaseVectorStore):
    """Production vector store using PostgreSQL, pgvector (HNSW), and tsvector (GIN)."""

    def __init__(self, pool: asyncpg.Pool | None = None) -> None:
        self._pool = pool

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await get_db_pool()
        return self._pool

    async def aadd_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Upserts chunks and dense vectors into PostgreSQL."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})"
            )

        pool = await self._get_pool()
        query = """
        INSERT INTO document_chunks (
            id, section, question_number, question_text, text, source_doc, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            section = EXCLUDED.section,
            question_number = EXCLUDED.question_number,
            question_text = EXCLUDED.question_text,
            text = EXCLUDED.text,
            source_doc = EXCLUDED.source_doc,
            embedding = EXCLUDED.embedding,
            created_at = NOW();
        """

        records = [
            (
                chunk.id,
                chunk.metadata.section,
                str(chunk.metadata.question_number) if chunk.metadata.question_number is not None else None,
                chunk.metadata.question_text,
                chunk.text,
                chunk.metadata.source_document,
                emb,
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        try:
            async with pool.acquire() as conn:
                await conn.executemany(query, records)
            logger.info(f"Successfully upserted {len(chunks)} chunks into PostgreSQL.")
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks into PostgreSQL: {e}") from e

    async def asimilarity_search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 4,
        where_filter: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Dense vector search using pgvector HNSW cosine distance."""
        pool = await self._get_pool()
        section_filter = where_filter.get("section") if where_filter else None

        query = """
        SELECT 
            id, section, question_number, question_text, text, source_doc,
            (1 - (embedding <=> $1)) AS similarity
        FROM document_chunks
        WHERE ($2::text IS NULL OR section = $2)
        ORDER BY embedding <=> $1 ASC
        LIMIT $3;
        """

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, query_vector, section_filter, top_k)

            results: list[tuple[DocumentChunk, float]] = []
            for r in rows:
                q_num = None
                if r["question_number"] is not None and str(r["question_number"]).isdigit():
                    q_num = int(r["question_number"])

                meta = ChunkMetadata(
                    section=r["section"],
                    question_number=q_num,
                    question_text=r["question_text"],
                    source_document=r["source_doc"],
                    char_length=len(r["text"]),
                    token_count=len(r["text"].split()),
                )
                chunk = DocumentChunk(
                    id=r["id"],
                    text=r["text"],
                    metadata=meta,
                )
                results.append((chunk, float(r["similarity"])))
            return results

        except Exception as e:
            raise VectorStoreError(f"PostgreSQL dense search failed: {e}") from e

    async def ahybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 4,
        section_filter: str | None = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Unified Hybrid Search (Dense HNSW + Sparse GIN tsvector) with Reciprocal Rank Fusion."""
        pool = await self._get_pool()
        candidate_k = max(top_k * 3, 10)

        sql = """
        WITH dense_search AS (
            SELECT 
                id,
                1 - (embedding <=> $1) AS dense_score,
                ROW_NUMBER() OVER (ORDER BY embedding <=> $1 ASC) AS dense_rank
            FROM document_chunks
            WHERE ($2::text IS NULL OR section = $2)
            ORDER BY embedding <=> $1 ASC
            LIMIT $3
        ),
        sparse_search AS (
            SELECT 
                id,
                ts_rank_cd(tsv, plainto_tsquery('english', $4)) AS sparse_score,
                ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC) AS sparse_rank
            FROM document_chunks
            WHERE ($2::text IS NULL OR section = $2)
              AND tsv @@ plainto_tsquery('english', $4)
            ORDER BY sparse_score DESC
            LIMIT $3
        ),
        fused AS (
            SELECT
                coalesce(d.id, s.id) AS id,
                coalesce(d.dense_score, 0.0) AS dense_score,
                coalesce(s.sparse_score, 0.0) AS sparse_score,
                (
                    $5 * (1.0 / ($6 + coalesce(d.dense_rank, 1000))) +
                    $7 * (1.0 / ($6 + coalesce(s.sparse_rank, 1000)))
                ) AS rrf_score
            FROM dense_search d
            FULL OUTER JOIN sparse_search s ON d.id = s.id
        )
        SELECT 
            c.id,
            c.section,
            c.question_number,
            c.question_text,
            c.text,
            c.source_doc,
            f.dense_score,
            f.sparse_score,
            f.rrf_score,
            ROW_NUMBER() OVER (ORDER BY f.rrf_score DESC) AS final_rank
        FROM fused f
        JOIN document_chunks c ON f.id = c.id
        ORDER BY f.rrf_score DESC
        LIMIT $8;
        """

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    sql,
                    query_vector,
                    section_filter,
                    candidate_k,
                    query,
                    dense_weight,
                    rrf_k,
                    sparse_weight,
                    top_k,
                )

            retrieved: list[RetrievedChunk] = []
            for r in rows:
                q_num = None
                if r["question_number"] is not None and str(r["question_number"]).isdigit():
                    q_num = int(r["question_number"])

                meta = ChunkMetadata(
                    section=r["section"],
                    question_number=q_num,
                    question_text=r["question_text"],
                    source_document=r["source_doc"],
                    char_length=len(r["text"]),
                    token_count=len(r["text"].split()),
                )
                chunk = DocumentChunk(
                    id=r["id"],
                    text=r["text"],
                    metadata=meta,
                )
                retrieved.append(
                    RetrievedChunk(
                        chunk=chunk,
                        dense_score=float(r["dense_score"]),
                        sparse_score=float(r["sparse_score"]),
                        rrf_score=float(r["rrf_score"]),
                        rank=int(r["final_rank"]),
                    )
                )
            return retrieved
        except Exception as e:
            raise VectorStoreError(f"PostgreSQL hybrid search failed: {e}") from e

    async def aget_sections(self) -> list[str]:
        """Returns sorted distinct section titles from PostgreSQL."""
        pool = await self._get_pool()
        query = "SELECT DISTINCT section FROM document_chunks WHERE section IS NOT NULL ORDER BY section;"
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)
            return [r["section"] for r in rows if r["section"]]
        except Exception as e:
            raise VectorStoreError(f"Failed to fetch sections from PostgreSQL: {e}") from e

    async def acount(self) -> int:
        """Returns total chunks count in PostgreSQL."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval("SELECT count(*) FROM document_chunks;")
            return int(count)
        except Exception as e:
            raise VectorStoreError(f"Failed to count chunks in PostgreSQL: {e}") from e

    async def aclear(self) -> None:
        """Truncates document chunks table."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE document_chunks;")
            logger.info("Cleared all records from PostgreSQL document_chunks table.")
        except Exception as e:
            raise VectorStoreError(f"Failed to truncate document_chunks: {e}") from e

    # --- Synchronous compatibility wrappers for BaseVectorStore interface ---

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self._run_sync(self.aadd_chunks(chunks, embeddings))

    def similarity_search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 4,
        where_filter: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        return self._run_sync(
            self.asimilarity_search_by_vector(query_vector, top_k, where_filter)
        )

    def count(self) -> int:
        return self._run_sync(self.acount())

    def clear(self) -> None:
        self._run_sync(self.aclear())

    def _run_sync(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            # If already inside an event loop, run in a separate worker thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()
