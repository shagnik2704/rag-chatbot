-- Future-Ready Children RAG PostgreSQL + pgvector Schema

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Document Chunks Table with vector and full-text search columns
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(64) PRIMARY KEY,
    section VARCHAR(255) NOT NULL,
    question_number VARCHAR(32),
    question_text TEXT,
    text TEXT NOT NULL,
    source_doc VARCHAR(255) NOT NULL,
    embedding vector(384),
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(question_text, '') || ' ' || text)
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for ultra-fast dense cosine similarity
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text lexical search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv 
ON document_chunks USING gin (tsv);

-- B-Tree index for section metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_section 
ON document_chunks (section);

-- Semantic Cache Table with LRU eviction tracking
CREATE TABLE IF NOT EXISTS semantic_cache (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_vector vector(384) NOT NULL,
    answer TEXT NOT NULL,
    hit_count INTEGER DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HNSW index for cache cosine similarity lookups
CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw 
ON semantic_cache USING hnsw (query_vector vector_cosine_ops);

-- B-Tree index on last_accessed_at for LRU eviction queries
CREATE INDEX IF NOT EXISTS idx_cache_last_accessed 
ON semantic_cache (last_accessed_at);
