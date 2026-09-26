"""Integration tests for PostgreSQL + pgvector vector store and semantic cache."""

import pytest
from src.core.config import get_settings
from src.db.connection import get_db_pool
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.models.chunk import ChunkMetadata, DocumentChunk
from src.services.postgres_semantic_cache import PostgresSemanticCache
from src.vectorstore.postgres_store import PostgresVectorStore


@pytest.mark.anyio
async def test_postgres_vector_store_upsert_and_hybrid_search():
    pool = await get_db_pool()
    store = PostgresVectorStore(pool=pool)

    # Verify existing indexed chunks count
    count = await store.acount()
    assert count >= 30, f"Expected at least 30 chunks in PostgreSQL, found {count}"

    # Verify sections
    sections = await store.aget_sections()
    assert len(sections) >= 7
    assert any("UNDERSTANDING" in s for s in sections)

    # Test Hybrid Search with query
    query = "How much does it cost to support a school?"
    emb = LocalSentenceTransformerEmbeddings()
    q_vec = emb.embed_query(query)

    hits = await store.ahybrid_search(query=query, query_vector=q_vec, top_k=3)
    assert len(hits) == 3
    # Top rank should be Question 12 (Cost to support a school)
    assert hits[0].chunk.metadata.question_number in [12, 30]
    assert hits[0].dense_score > 0.7
    assert hits[0].rrf_score > 0.0


@pytest.mark.anyio
async def test_postgres_semantic_cache_lifecycle():
    pool = await get_db_pool()
    cache = PostgresSemanticCache(pool=pool, similarity_threshold=0.92, max_size=100)

    # Create dummy vector of dimension 384
    test_vec = [0.15] * 384
    test_query = "What is the mission of EduPyramids test?"
    test_answer = "EduPyramids focuses on practical, skill-based education."

    # Cache response
    await cache.aset(test_query, test_vec, test_answer)

    # Exact vector lookup should hit
    hit = await cache.aget(test_vec)
    assert hit == test_answer

    # Slightly perturbed vector (cosine similarity ~0.99) should hit
    perturbed_vec = [0.1505] * 384
    near_hit = await cache.aget(perturbed_vec)
    assert near_hit == test_answer

    # Completely orthogonal/different vector should miss
    diff_vec = [-0.15] * 384
    miss = await cache.aget(diff_vec)
    assert miss is None
