# Enterprise RAG Chatbot: Future-Ready Children Q&A

An enterprise-grade Retrieval-Augmented Generation (RAG) system for querying the **Future-Ready Children Champion FAQ & Talking Points** document. Powered by **Sarvam AI (`glm5.3`)**, local dense vector embeddings (`bge-small-en-v1.5`), BM25 sparse lexical search with Reciprocal Rank Fusion (RRF), a FastAPI backend, and a clean responsive React + TypeScript frontend.

---

## Key Features

- **Structural Q&A Ingestion:** Purpose-built parser that respects section hierarchies and question-answer pairs rather than blind token chunking.
- **Dual-Index Hybrid Retrieval:** Dense vector similarity (ChromaDB) combined with BM25 sparse keyword ranking via Reciprocal Rank Fusion (RRF).
- **Sarvam AI GLM-5.3 Integration:** Resilient LLM inference client with exponential backoff (`tenacity`) and timeout handling.
- **Grounded Attribution:** Enforces factual answers, citations to specific sections/questions, champion talking points, and escalation routing.
- **Clean React + TypeScript Frontend:** Minimalist, enterprise-grade design without emojis or extravagant styling.
- **FastAPI REST Service:** Fully typed backend with CORS and OpenAPI specs.
- **CLI & Test Suite:** Rich terminal interface and 100% passing Pytest unit test coverage.

---

## Architecture Overview

```
rag-chatbot/
├── backend: FastAPI (port 8000)
│   ├── src/core/           # Settings, logging, and custom typed exceptions
│   ├── src/models/         # Chunk, query, and response domain models
│   ├── src/ingestion/      # Loaders (PDF/DOCX) & QAStructuralChunker
│   ├── src/embeddings/     # LocalSentenceTransformerEmbeddings
│   ├── src/vectorstore/    # ChromaVectorStore (persistent)
│   ├── src/retrieval/      # BM25Index & HybridRetriever (Dense + BM25 with RRF)
│   ├── src/llm/            # SarvamGLMClient (model: glm5.3) & Grounded Prompts
│   ├── src/services/       # IndexingService & RAGService
│   └── src/app/api.py      # FastAPI application
│
└── frontend: React + TypeScript (Vite, port 5173)
    ├── src/types/          # Type-safe chat, citation, and status schemas
    ├── src/api/client.ts   # REST API client
    ├── src/components/     # Modular enterprise UI components (no emojis)
    ├── src/hooks/useChat.ts# Chat state and interaction hook
    └── src/index.css       # Clean, accessible design system styling
```

---

## Quickstart

### 1. Environment Setup
```bash
# Copy and update environment variables
cp .env.example .env
# Edit .env and configure your SARVAM_API_KEY
```

### 2. Index the Document
```bash
python -m src.app.cli index --file "data/raw/Future-Ready Children_ FAQ.docx"
```

### 3. Start the Backend API (FastAPI)
```bash
source .venv/bin/activate
uvicorn src.app.api:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Start the Frontend (React + TypeScript)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running Tests
```bash
.venv/bin/pytest -v
```
