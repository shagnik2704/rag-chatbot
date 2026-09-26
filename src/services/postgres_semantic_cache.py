"""PostgreSQL + pgvector semantic cache with LRU eviction and concurrency safety."""

import asyncio
from typing import Optional
import asyncpg

from src.core.config import get_settings
from src.core.logging import setup_logger
from src.db.connection import get_db_pool

logger = setup_logger(__name__)


class PostgresSemanticCache:
    """Thread-safe and process-safe semantic vector cache backed by PostgreSQL + pgvector."""

    def __init__(
        self,
        pool: Optional[asyncpg.Pool] = None,
        similarity_threshold: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._pool = pool
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.semantic_cache_threshold
        )
        self.max_size = max_size if max_size is not None else settings.semantic_cache_max_size

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await get_db_pool()
        return self._pool

    async def aget(self, query_vector: list[float]) -> str | None:
        """Looks up a semantically similar cached response in PostgreSQL."""
        pool = await self._get_pool()

        query = """
        SELECT id, query, answer, (1 - (query_vector <=> $1)) AS similarity
        FROM semantic_cache
        WHERE (1 - (query_vector <=> $1)) >= $2
        ORDER BY similarity DESC
        LIMIT 1;
        """

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, query_vector, self.similarity_threshold)
                if row:
                    cache_id = row["id"]
                    # Update access statistics for LRU tracking
                    await conn.execute(
                        """
                        UPDATE semantic_cache
                        SET hit_count = hit_count + 1,
                            last_accessed_at = NOW()
                        WHERE id = $1;
                        """,
                        cache_id,
                    )
                    logger.info(
                        f"PostgreSQL semantic cache HIT (similarity: {row['similarity']:.3f}) for query: '{row['query']}'"
                    )
                    return row["answer"]
                return None
        except Exception as e:
            logger.warning(f"Error querying PostgreSQL semantic cache: {e}")
            return None

    async def aset(self, query: str, query_vector: list[float], answer: str) -> None:
        """Stores a new query and answer pair in PostgreSQL with LRU eviction."""
        if not answer.strip():
            return

        pool = await self._get_pool()

        insert_sql = """
        INSERT INTO semantic_cache (query, query_vector, answer)
        VALUES ($1, $2, $3);
        """

        evict_sql = """
        DELETE FROM semantic_cache
        WHERE id IN (
            SELECT id FROM semantic_cache
            ORDER BY last_accessed_at ASC
            LIMIT (SELECT GREATEST(0, count(*) - $1) FROM semantic_cache)
        );
        """

        try:
            async with pool.acquire() as conn:
                await conn.execute(insert_sql, query.strip(), query_vector, answer.strip())
                # Enforce LRU cap
                await conn.execute(evict_sql, self.max_size)
            logger.debug(f"Cached answer in PostgreSQL for query: '{query[:40]}...'")
        except Exception as e:
            logger.warning(f"Failed to insert into PostgreSQL semantic cache: {e}")

    async def aclear(self) -> None:
        """Truncates semantic cache table."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE semantic_cache;")
            logger.info("Cleared all records from PostgreSQL semantic cache.")
        except Exception as e:
            logger.warning(f"Failed to clear semantic cache: {e}")

    async def asize(self) -> int:
        """Returns total entries in semantic cache."""
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                count = await conn.fetchval("SELECT count(*) FROM semantic_cache;")
            return int(count)
        except Exception as e:
            logger.warning(f"Failed to get cache size: {e}")
            return 0

    # --- Synchronous compatibility wrappers ---

    def get(self, query_vector: list[float]) -> str | None:
        return self._run_sync(self.aget(query_vector))

    def set(self, query: str, query_vector: list[float], answer: str) -> None:
        self._run_sync(self.aset(query, query_vector, answer))

    def clear(self) -> None:
        self._run_sync(self.aclear())

    def size(self) -> int:
        return self._run_sync(self.asize())

    def _run_sync(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()
