#!/usr/bin/env bash
set -e

echo "=== Future-Ready Children RAG Chatbot Container Starting ==="

# 1. Wait for PostgreSQL to become reachable
echo "Checking PostgreSQL connection at ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-rag_db}..."

python - << 'EOF'
import asyncio
import os
import sys
import asyncpg
from pgvector.asyncpg import register_vector

host = os.getenv("POSTGRES_HOST", "localhost")
port = int(os.getenv("POSTGRES_PORT", 5432))
user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
database = os.getenv("POSTGRES_DB", "rag_db")

dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}" if password else f"postgresql://{user}@{host}:{port}/{database}"

async def wait_for_postgres():
    max_retries = 30
    for i in range(1, max_retries + 1):
        try:
            conn = await asyncpg.connect(dsn=dsn, timeout=5)
            await register_vector(conn)
            await conn.close()
            print(f"Successfully connected to PostgreSQL (attempt {i}/{max_retries}).")
            return True
        except Exception as e:
            print(f"Waiting for PostgreSQL ({i}/{max_retries}): {e}")
            await asyncio.sleep(1)
    print("Error: PostgreSQL unreachable after maximum retries.")
    return False

if not asyncio.run(wait_for_postgres()):
    sys.exit(1)
EOF

# 2. Check if database has indexed chunks, otherwise auto-seed
echo "Checking database status..."

python - << 'EOF'
import asyncio
from pathlib import Path
from src.db.connection import get_db_pool, init_db
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.services.indexing_service import IndexingService
from src.vectorstore.postgres_store import PostgresVectorStore

async def check_and_seed():
    pool = await get_db_pool()
    await init_db(pool)
    store = PostgresVectorStore(pool=pool)
    count = await store.acount()

    if count == 0:
        print("Database is empty. Performing zero-touch initial FAQ indexing...")
        doc_path = Path("data/raw/Future-Ready Children_ FAQ.docx")
        if not doc_path.exists():
            doc_path = Path("data/raw/Future-Ready Children_ FAQ.pdf")

        if doc_path.exists():
            emb = LocalSentenceTransformerEmbeddings()
            idx = IndexingService(vector_store=store, embedding_provider=emb)
            chunks = await idx.aindex_document(doc_path, clear_existing=True)
            print(f"Initial indexing complete: {len(chunks)} chunks loaded into PostgreSQL.")
        else:
            print("Warning: No raw FAQ document found in data/raw/. Skipping indexing.")
    else:
        print(f"Database already populated with {count} chunks. Ready.")

asyncio.run(check_and_seed())
EOF

echo "=== Launching Production Application Server ==="
exec "$@"
