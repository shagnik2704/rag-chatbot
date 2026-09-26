"""FastAPI backend application exposing the RAG pipeline."""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import setup_logger
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.llm.sarvam_client import MockLLMClient, SarvamGLMClient
from src.models.query import QueryRequest
from src.models.response import Citation, RAGResponse
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.indexing_service import IndexingService
from src.services.rag_service import RAGService
from src.services.semantic_cache import SemanticCache
from src.vectorstore.chroma_store import ChromaVectorStore

logger = setup_logger("rag_api")

# Singleton state container
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    embedding_provider = LocalSentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
    )
    bm25_index = BM25Index()

    # Reconstruct BM25 index from persistent vector store if populated
    if vector_store.count() > 0:
        dummy_vec = [0.0] * embedding_provider.dimension
        stored = vector_store.similarity_search_by_vector(dummy_vec, top_k=vector_store.count())
        bm25_index.index_chunks([chunk for chunk, _ in stored])

    indexing_service = IndexingService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        rrf_k=settings.rrf_k,
    )

    semantic_cache = SemanticCache()

    app_state["settings"] = settings
    app_state["vector_store"] = vector_store
    app_state["embedding_provider"] = embedding_provider
    app_state["bm25_index"] = bm25_index
    app_state["indexing_service"] = indexing_service
    app_state["retriever"] = retriever
    app_state["semantic_cache"] = semantic_cache

    logger.info("RAG API dependencies initialized successfully.")
    yield
    app_state.clear()


app = FastAPI(
    title="Future-Ready Children RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    filter_section: str | None = None
    generate_talking_points: bool = True
    api_key: str | None = None


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


class SectionListResponse(BaseModel):
    sections: list[str]


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
async def get_system_status():
    vector_store: ChromaVectorStore = app_state["vector_store"]
    settings = app_state["settings"]
    has_key = bool(settings.sarvam_api_key and not settings.sarvam_api_key.startswith("your_sarvam"))

    return StatusResponse(
        status="ready" if vector_store.count() > 0 else "empty",
        indexed_chunks=vector_store.count(),
        model_name=settings.sarvam_model,
        has_api_key=has_key,
    )


@app.get("/api/sections", response_model=SectionListResponse)
async def get_sections():
    vector_store: ChromaVectorStore = app_state["vector_store"]
    embedding_provider: LocalSentenceTransformerEmbeddings = app_state["embedding_provider"]

    if vector_store.count() == 0:
        return SectionListResponse(sections=[])

    dummy_vec = [0.0] * embedding_provider.dimension
    stored = vector_store.similarity_search_by_vector(dummy_vec, top_k=vector_store.count())
    sections = sorted(
        list({chunk.metadata.section for chunk, _ in stored if chunk.metadata.section})
    )
    return SectionListResponse(sections=sections)


@app.post("/api/query", response_model=ChatResponse)
async def query_faq(request: ChatRequest):
    retriever: HybridRetriever = app_state["retriever"]
    semantic_cache: SemanticCache = app_state["semantic_cache"]
    settings = app_state["settings"]

    effective_key = (request.api_key or settings.sarvam_api_key).strip()

    if effective_key and not effective_key.startswith("your_sarvam"):
        llm_client = SarvamGLMClient(
            api_key=effective_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_model,
            temperature=settings.sarvam_temperature,
            max_tokens=settings.sarvam_max_tokens,
            timeout=settings.sarvam_timeout_seconds,
        )
    else:
        llm_client = MockLLMClient()

    rag_service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
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
    retriever: HybridRetriever = app_state["retriever"]
    semantic_cache: SemanticCache = app_state["semantic_cache"]
    settings = app_state["settings"]

    effective_key = (request.api_key or settings.sarvam_api_key).strip()

    if effective_key and not effective_key.startswith("your_sarvam"):
        llm_client = SarvamGLMClient(
            api_key=effective_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_model,
            temperature=settings.sarvam_temperature,
            max_tokens=settings.sarvam_max_tokens,
            timeout=settings.sarvam_timeout_seconds,
        )
    else:
        llm_client = MockLLMClient()

    rag_service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
        semantic_cache=semantic_cache,
    )
    query_req = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        filter_section=request.filter_section,
    )

    def event_generator():
        try:
            for token in rag_service.answer_query_stream(query_req):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            logger.error(f"Error during stream generation: {err}", exc_info=True)
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reindex")
async def reindex_document():
    indexing_service: IndexingService = app_state["indexing_service"]
    doc_path = Path("data/raw/Future-Ready Children_ FAQ.docx")
    if not doc_path.exists():
        doc_path = Path("data/raw/Future-Ready Children_ FAQ.pdf")

    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="No source document found in data/raw/")

    chunks = indexing_service.index_document(doc_path, clear_existing=True)
    return {"message": "Successfully re-indexed document", "total_chunks": len(chunks)}
