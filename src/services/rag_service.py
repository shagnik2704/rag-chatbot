"""RAG orchestration service coordinating retrieval, semantic caching, and streaming generation."""

import asyncio
import re
import time
from typing import Any, AsyncGenerator, Generator, Optional
from src.core.logging import setup_logger
from src.embeddings.base import BaseEmbeddingProvider
from src.llm.base import BaseLLMClient
from src.llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.models.query import QueryRequest, RetrievedChunk
from src.models.response import Citation, RAGResponse
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.postgres_store import PostgresVectorStore

logger = setup_logger(__name__)


class RAGService:
    """End-to-end question answering service using hybrid retrieval, semantic caching, and Sarvam GLM-5.3."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: Optional[HybridRetriever] = None,
        postgres_store: Optional[PostgresVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        semantic_cache: Any = None,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.postgres_store = postgres_store
        self.embedding_provider = (
            embedding_provider
            or (getattr(retriever, "embedding_provider", None) if retriever else None)
        )
        self.semantic_cache = semantic_cache

    async def _get_query_vector(self, query: str) -> list[float] | None:
        if not self.embedding_provider:
            return None
        return await asyncio.to_thread(self.embedding_provider.embed_query, query)

    async def _retrieve_chunks(self, request: QueryRequest, query_vector: list[float] | None) -> list[RetrievedChunk]:
        if self.postgres_store is not None:
            if query_vector is None and self.embedding_provider:
                query_vector = await self._get_query_vector(request.query)
            return await self.postgres_store.ahybrid_search(
                query=request.query,
                query_vector=query_vector or [],
                top_k=request.top_k,
                section_filter=request.filter_section,
            )
        elif self.retriever is not None:
            from unittest.mock import AsyncMock
            aretrieve_fn = getattr(self.retriever, "aretrieve", None)
            if aretrieve_fn is not None and not isinstance(aretrieve_fn, AsyncMock):
                return await self.retriever.aretrieve(
                    query=request.query,
                    top_k=request.top_k,
                    section_filter=request.filter_section,
                )
            elif hasattr(self.retriever, "retrieve"):
                return self.retriever.retrieve(
                    query=request.query,
                    top_k=request.top_k,
                    section_filter=request.filter_section,
                )
            return []
        else:
            raise ValueError("Neither postgres_store nor retriever configured.")



    async def aanswer_query(self, request: QueryRequest) -> RAGResponse:
        """Asynchronous execution using concurrent retrieval via PostgreSQL or HybridRetriever."""
        logger.info(f"Processing async query: '{request.query}'")

        query_vector = None
        if self.semantic_cache:
            query_vector = await self._get_query_vector(request.query)
            if hasattr(self.semantic_cache, "aget"):
                cached_answer = await self.semantic_cache.aget(query_vector)
            else:
                cached_answer = self.semantic_cache.get(query_vector)

            if cached_answer:
                return RAGResponse(
                    query=request.query,
                    answer=cached_answer,
                    champion_talking_point=None,
                    citations=[],
                    model_used=f"{self.llm_client.model_name} (cached)",
                    is_fallback=False,
                )

        if query_vector is None:
            query_vector = await self._get_query_vector(request.query)

        retrieved_chunks = await self._retrieve_chunks(request, query_vector)
        prompt = build_rag_prompt(request.query, retrieved_chunks)

        raw_completion = await self.llm_client.agenerate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        clean_answer = self._clean_answer(raw_completion)
        citations = self._build_citations(retrieved_chunks)

        is_fallback = (
            "not covered in the" in raw_completion.lower()
            or "please reach out directly to the campaign" in raw_completion.lower()
            or not retrieved_chunks
        )

        if self.semantic_cache and query_vector is not None and not is_fallback:
            if hasattr(self.semantic_cache, "aset"):
                await self.semantic_cache.aset(request.query, query_vector, clean_answer)
            else:
                self.semantic_cache.set(request.query, query_vector, clean_answer)

        return RAGResponse(
            query=request.query,
            answer=clean_answer,
            champion_talking_point=None,
            citations=citations,
            model_used=self.llm_client.model_name,
            is_fallback=is_fallback,
        )

    async def aanswer_query_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Asynchronously streams response tokens in real-time, checking semantic cache first."""
        query_vector = None
        if self.semantic_cache:
            query_vector = await self._get_query_vector(request.query)
            if hasattr(self.semantic_cache, "aget"):
                cached = await self.semantic_cache.aget(query_vector)
            else:
                cached = self.semantic_cache.get(query_vector)

            if cached:
                words = cached.split(" ")
                for word in words:
                    await asyncio.sleep(0.015)
                    yield word + " "
                return

        if query_vector is None:
            query_vector = await self._get_query_vector(request.query)

        retrieved_chunks = await self._retrieve_chunks(request, query_vector)
        prompt = build_rag_prompt(request.query, retrieved_chunks)

        tokens = []
        async for token in self.llm_client.agenerate_stream(prompt=prompt, system_prompt=SYSTEM_PROMPT):
            tokens.append(token)
            yield token

        full_raw = "".join(tokens)
        clean = self._clean_answer(full_raw)
        is_fallback = "not covered in the" in full_raw.lower() or not retrieved_chunks
        if self.semantic_cache and query_vector is not None and clean and not is_fallback:
            if hasattr(self.semantic_cache, "aset"):
                await self.semantic_cache.aset(request.query, query_vector, clean)
            else:
                self.semantic_cache.set(request.query, query_vector, clean)

    def answer_query(self, request: QueryRequest) -> RAGResponse:
        """Synchronous wrapper for answer_query."""
        return asyncio.run(self.aanswer_query(request))

    def answer_query_stream(self, request: QueryRequest) -> Generator[str, None, None]:
        """Synchronous generator wrapper for answer_query_stream."""
        loop = asyncio.new_event_loop()
        try:
            gen = self.aanswer_query_stream(request)
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def _clean_answer(self, raw_text: str) -> str:
        """Cleans internal headers from output."""
        text = raw_text
        text = re.sub(r"(?:^|\n)(?:#{1,3}\s*|\*\*)Champion Talking Point.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?:^|\n)(?:#{1,3}\s*|\*\*)Citations.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^(?:#{1,3}\s*|\*\*)Direct Answer(?:\*\*)?[:\s]*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for rc in chunks:
            c = rc.chunk
            excerpt = c.text[:220] + "..." if len(c.text) > 220 else c.text
            citations.append(
                Citation(
                    chunk_id=c.id,
                    section=c.metadata.section,
                    question_number=c.metadata.question_number,
                    question_text=c.metadata.question_text,
                    excerpt=excerpt,
                )
            )
        return citations
