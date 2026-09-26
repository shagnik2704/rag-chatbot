"""PostgreSQL connection pool management with pgvector type registration."""

import asyncio
from pathlib import Path
from typing import Optional
import asyncpg
from pgvector.asyncpg import register_vector

from src.core.config import get_settings
from src.core.logging import setup_logger

logger = setup_logger(__name__)

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Configures each connection in the pool with pgvector codecs."""
    await register_vector(conn)


async def get_db_pool() -> asyncpg.Pool:
    """Returns or creates the singleton asyncpg connection pool."""
    global _pool
    current_loop = asyncio.get_running_loop()

    if _pool is not None:
        if getattr(_pool, "_loop", None) is not current_loop or _pool._loop.is_closed():
            _pool = None
        else:
            return _pool

    settings = get_settings()
    dsn = settings.get_database_dsn()
    logger.info(f"Connecting to PostgreSQL at {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}...")

    try:
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=20,
            init=_init_connection,
            timeout=30.0,
            command_timeout=60.0,
        )
        logger.info("PostgreSQL connection pool established successfully.")
        return _pool
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection pool: {e}", exc_info=True)
        raise



async def init_db(pool: asyncpg.Pool) -> None:
    """Applies schema DDL if tables and indexes do not exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found at {schema_path}")
        return

    schema_sql = schema_path.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        logger.info("Applying PostgreSQL schema DDL...")
        await conn.execute(schema_sql)
        logger.info("PostgreSQL schema DDL applied successfully.")


async def close_db_pool() -> None:
    """Closes all connections in the pool."""
    global _pool
    if _pool is not None:
        logger.info("Closing PostgreSQL connection pool...")
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed.")
