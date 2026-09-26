"""Production FastAPI backend exposing the RAG pipeline with PostgreSQL + pgvector."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import setup_logger
from src.db.connection import close_db_pool, get_db_pool, init_db
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.llm.sarvam_client import MockLLMClient, SarvamGLMClient
from src.models.query import QueryRequest
from src.models.response import Citation, RAGResponse
from src.services.indexing_service import IndexingService
from src.services.postgres_semantic_cache import PostgresSemanticCache
from src.services.rag_service import RAGService
from src.vectorstore.postgres_store import PostgresVectorStore

logger = setup_logger("rag_api")

app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing enterprise RAG API services...")

    # 1. Initialize PostgreSQL Connection Pool & Schema DDL
    db_pool = await get_db_pool()
    await init_db(db_pool)

    # 2. Storage & Embedding Providers
    postgres_store = PostgresVectorStore(pool=db_pool)
    embedding_provider = LocalSentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
    )
    indexing_service = IndexingService(
        vector_store=postgres_store,
        embedding_provider=embedding_provider,
    )

    # 3. PostgreSQL Semantic Cache (LRU-bounded, WAL concurrency safe)
    semantic_cache = PostgresSemanticCache(
        pool=db_pool,
        similarity_threshold=settings.semantic_cache_threshold,
        max_size=settings.semantic_cache_max_size,
    )

    # 4. Shared Persistent Async HTTP Client for LLM requests
    http_client = httpx.AsyncClient(
        timeout=settings.sarvam_timeout_seconds,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    )

    app_state["settings"] = settings
    app_state["db_pool"] = db_pool
    app_state["postgres_store"] = postgres_store
    app_state["embedding_provider"] = embedding_provider
    app_state["indexing_service"] = indexing_service
    app_state["semantic_cache"] = semantic_cache
    app_state["http_client"] = http_client

    logger.info("RAG API services fully initialized with PostgreSQL and connection pool.")
    yield

    logger.info("Shutting down RAG API services...")
    await http_client.aclose()
    await close_db_pool()
    app_state.clear()
    logger.info("RAG API shutdown complete.")


app = FastAPI(
    title="Future-Ready Children RAG API",
    version="2.0.0",
    description="Enterprise-grade RAG pipeline using PostgreSQL, pgvector, and Sarvam AI GLM-5.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    top_k: int = Field(default=4, ge=1, le=10, description="Top context chunks to retrieve")
    filter_section: str | None = Field(default=None, description="Optional section filter")
    generate_talking_points: bool = Field(default=False)
    api_key: str | None = Field(default=None, description="Optional override API key")


class ChatResponse(BaseModel):
    query: str
    answer: str
    champion_talking_point: str | None
    citations: list[Citation]
    model_used: str
    is_fallback: bool
    fallback_contacts: list[str]


class StatusResponse(BaseModel):
    status: str
    indexed_chunks: int
    model_name: str
    has_api_key: bool
    db_backend: str


class SectionListResponse(BaseModel):
    sections: list[str]


@app.get("/healthz", tags=["Ops"])
@app.get("/api/health", tags=["Ops"])
async def health_check():
    """Kubernetes/Container liveness probe."""
    return {"status": "ok"}


@app.get("/readyz", tags=["Ops"])
async def readiness_check():
    """Readiness probe checking database connectivity."""
    db_pool = app_state.get("db_pool")
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database pool not initialized")
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1;")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unready: {e}")


@app.get("/api/status", response_model=StatusResponse)
async def get_system_status():
    postgres_store: PostgresVectorStore = app_state["postgres_store"]
    settings = app_state["settings"]
    has_key = bool(settings.sarvam_api_key and not settings.sarvam_api_key.startswith("your_sarvam"))

    chunk_count = await postgres_store.acount()

    return StatusResponse(
        status="ready" if chunk_count > 0 else "empty",
        indexed_chunks=chunk_count,
        model_name=settings.sarvam_model,
        has_api_key=has_key,
        db_backend="PostgreSQL + pgvector",
    )


@app.get("/api/sections", response_model=SectionListResponse)
async def get_sections():
    postgres_store: PostgresVectorStore = app_state["postgres_store"]
    sections = await postgres_store.aget_sections()
    return SectionListResponse(sections=sections)


def _get_llm_client(request_api_key: str | None) -> tuple[Any, str]:
    settings = app_state["settings"]
    shared_http = app_state["http_client"]
    effective_key = (request_api_key or settings.sarvam_api_key).strip()

    if effective_key and not effective_key.startswith("your_sarvam"):
        client = SarvamGLMClient(
            api_key=effective_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_model,
            temperature=settings.sarvam_temperature,
            max_tokens=settings.sarvam_max_tokens,
            timeout=settings.sarvam_timeout_seconds,
            async_client=shared_http,
        )
        return client, settings.sarvam_model
    return MockLLMClient(), "mock-glm5.3"


@app.post("/api/query", response_model=ChatResponse)
async def query_faq(request: ChatRequest):
    postgres_store: PostgresVectorStore = app_state["postgres_store"]
    embedding_provider = app_state["embedding_provider"]
    semantic_cache = app_state["semantic_cache"]

    llm_client, _ = _get_llm_client(request.api_key)

    rag_service = RAGService(
        llm_client=llm_client,
        postgres_store=postgres_store,
        embedding_provider=embedding_provider,
        semantic_cache=semantic_cache,
    )

    try:
        query_req = QueryRequest(
            query=request.query,
            top_k=request.top_k,
            filter_section=request.filter_section,
            generate_talking_points=request.generate_talking_points,
        )
        response: RAGResponse = await rag_service.aanswer_query(query_req)
        return ChatResponse(
            query=response.query,
            answer=response.answer,
            champion_talking_point=response.champion_talking_point,
            citations=response.citations,
            model_used=response.model_used,
            is_fallback=response.is_fallback,
            fallback_contacts=response.fallback_contacts,
        )
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def stream_query(request: ChatRequest):
    """Streams response tokens asynchronously via Server-Sent Events."""
    postgres_store: PostgresVectorStore = app_state["postgres_store"]
    embedding_provider = app_state["embedding_provider"]
    semantic_cache = app_state["semantic_cache"]

    llm_client, _ = _get_llm_client(request.api_key)

    rag_service = RAGService(
        llm_client=llm_client,
        postgres_store=postgres_store,
        embedding_provider=embedding_provider,
        semantic_cache=semantic_cache,
    )
    query_req = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        filter_section=request.filter_section,
    )

    async def async_event_generator():
        try:
            async for token in rag_service.aanswer_query_stream(query_req):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            logger.error(f"Error during async stream generation: {err}", exc_info=True)
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return StreamingResponse(
        async_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reindex")
async def reindex_document(x_admin_key: str | None = Header(default=None)):
    """Admin endpoint to re-index the FAQ document into PostgreSQL."""
    indexing_service: IndexingService = app_state["indexing_service"]

    doc_path = Path("data/raw/Future-Ready Children_ FAQ.docx")
    if not doc_path.exists():
        doc_path = Path("data/raw/Future-Ready Children_ FAQ.pdf")

    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="No source document found in data/raw/")

    chunks = await indexing_service.aindex_document(doc_path, clear_existing=True)
    return {
        "message": "Successfully re-indexed document into PostgreSQL + pgvector",
        "total_chunks": len(chunks),
    }
