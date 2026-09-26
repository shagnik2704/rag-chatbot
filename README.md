# Enterprise RAG Chatbot: Future-Ready Children Q&A

An enterprise-grade Retrieval-Augmented Generation (RAG) system for querying the **Future-Ready Children Champion FAQ & Talking Points** document. Powered by **Sarvam AI (`glm5.3`)**, local dense vector embeddings (`bge-small-en-v1.5`), unified **PostgreSQL + `pgvector`** hybrid retrieval with Reciprocal Rank Fusion (RRF), a FastAPI backend, and a clean responsive React + TypeScript frontend.

---

## Key Features

- **PostgreSQL + `pgvector` Unified Storage:** Eliminates embedded SQLite file lock contentions and in-memory heap duplication with a dedicated, ACID-compliant database.
- **Pure SQL Hybrid Retrieval (CTE):** Dense cosine similarity (HNSW index) and sparse keyword search (`tsvector` + GIN index) fused via Reciprocal Rank Fusion (RRF) directly inside the database engine.
- **PostgreSQL Semantic Cache with LRU:** Sub-30ms response time for semantically equivalent queries with automatic LRU eviction and multi-worker concurrency safety.
- **Sarvam AI GLM-5.3 Integration:** Resilient LLM inference client with exponential backoff (`tenacity`), streaming SSE tokens, and shared connection pooling (`httpx.AsyncClient`).
- **Grounded Attribution:** Enforces factual answers, citations to specific sections/questions, and escalation routing to campaign leadership.
- **Client-Facing UI:** React + TypeScript (Vite) interface with real-time token streaming and stream interruption (`AbortController`).
- **Docker & CI/CD Ready:** Multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow with automated pgvector test services.

---

## Architecture Overview

```
rag-chatbot/
├── backend: FastAPI (port 8000)
│   ├── src/core/           # Settings, logging, and custom typed exceptions
│   ├── src/db/             # PostgreSQL connection pool & schema DDL (pgvector + tsvector)
│   ├── src/models/         # Chunk, query, and response domain models
│   ├── src/ingestion/      # Loaders (PDF/DOCX) & QAStructuralChunker
│   ├── src/embeddings/     # LocalSentenceTransformerEmbeddings
│   ├── src/vectorstore/    # PostgresVectorStore (pgvector HNSW + tsvector GIN)
│   ├── src/llm/            # SarvamGLMClient (model: glm5.3) & Grounded Prompts
│   ├── src/services/       # IndexingService, PostgresSemanticCache, & RAGService
│   └── src/app/api.py      # FastAPI application (serves API & React SPA)
│
└── frontend: React + TypeScript (Vite, port 5173 / port 8000 in prod)
    ├── src/types/          # Type-safe chat, citation, and status schemas
    ├── src/api/client.ts   # REST API client with AbortSignal stream cancellation
    ├── src/components/     # Modular enterprise UI components (no emojis)
    ├── src/hooks/useChat.ts# Chat state, streaming, and stop button logic
    └── src/index.css       # Clean, accessible design system styling
```

---

## Quickstart with Docker Compose

The easiest way to run the entire full-stack application (PostgreSQL + pgvector + FastAPI + React UI) is via Docker Compose:

```bash
# 1. Clone repository and configure environment
cp .env.example .env
# Set your SARVAM_API_KEY in .env

# 2. Launch PostgreSQL with pgvector and full-stack app
docker compose up -d

# 3. Access application
# Open http://localhost:8000 in your browser
```

---

## Local Development Setup

### 1. PostgreSQL with pgvector
```bash
# Using Homebrew (macOS)
brew install pgvector
brew services restart postgresql@17

# Create database and apply schema
psql -d postgres -c "CREATE DATABASE rag_db;"
psql -d rag_db -f src/db/schema.sql
```

### 2. Python Environment Setup
```bash
# Using uv (fastest)
uv sync
# Or source .venv/bin/activate
```

### 3. Ingest and Index the Document
```bash
python -c "
import asyncio
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.vectorstore.postgres_store import PostgresVectorStore
from src.services.indexing_service import IndexingService

async def main():
    emb = LocalSentenceTransformerEmbeddings()
    vs = PostgresVectorStore()
    idx = IndexingService(vector_store=vs, embedding_provider=emb)
    await idx.aindex_document('data/raw/Future-Ready Children_ FAQ.docx', clear_existing=True)

asyncio.run(main())
"
```

### 4. Run Backend Server
```bash
uvicorn src.app.api:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Run Frontend Dev Server
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## Testing & CI/CD

Run the test suite across unit and database integration tests:
```bash
.venv/bin/pytest -v
```

GitHub Actions automatically runs:
- Automated PostgreSQL 16 + `pgvector` service container
- Full ingestion and hybrid retrieval test suite
- Frontend build and TypeScript check (`npm run build`)
- Multi-stage Docker build verification
