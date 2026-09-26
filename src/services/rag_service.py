"""RAG orchestration service coordinating retrieval, semantic caching, and streaming generation."""

import re
import time
from typing import Generator
from src.core.logging import setup_logger
from src.llm.base import BaseLLMClient
from src.llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.models.query import QueryRequest, RetrievedChunk
from src.models.response import Citation, RAGResponse
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.semantic_cache import SemanticCache

logger = setup_logger(__name__)


class RAGService:
    """End-to-end question answering service using hybrid retrieval, semantic caching, and Sarvam GLM-5.3."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: BaseLLMClient,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.semantic_cache = semantic_cache

    def answer_query(self, request: QueryRequest) -> RAGResponse:
        """Executes full RAG workflow with semantic caching."""
        logger.info(f"Processing query: '{request.query}'")

        # 1. Check Semantic Cache
        query_vector = None
        if self.semantic_cache:
            query_vector = self.retriever.embedding_provider.embed_query(request.query)
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

        # 2. Hybrid Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            section_filter=request.filter_section,
        )

        # 3. Build Grounded Prompt
        prompt = build_rag_prompt(request.query, retrieved_chunks)

        # 4. LLM Generation
        raw_completion = self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        # 5. Clean output
        clean_answer = self._clean_answer(raw_completion)
        citations = self._build_citations(retrieved_chunks)

        is_fallback = (
            "not covered in the" in raw_completion.lower()
            or "please reach out directly to the campaign" in raw_completion.lower()
            or not retrieved_chunks
        )

        # 6. Store in Semantic Cache
        if self.semantic_cache and query_vector is not None and not is_fallback:
            self.semantic_cache.set(request.query, query_vector, clean_answer)

        return RAGResponse(
            query=request.query,
            answer=clean_answer,
            champion_talking_point=None,
            citations=citations,
            model_used=self.llm_client.model_name,
            is_fallback=is_fallback,
        )

    async def aanswer_query(self, request: QueryRequest) -> RAGResponse:
        """Asynchronous execution using concurrent retrieval via asyncio.gather."""
        logger.info(f"Processing async query: '{request.query}'")

        query_vector = None
        if self.semantic_cache:
            query_vector = self.retriever.embedding_provider.embed_query(request.query)
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

        # Asynchronous concurrent retrieval
        retrieved_chunks = await self.retriever.aretrieve(
            query=request.query,
            top_k=request.top_k,
            section_filter=request.filter_section,
        )

        prompt = build_rag_prompt(request.query, retrieved_chunks)
        raw_completion = self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        clean_answer = self._clean_answer(raw_completion)
        citations = self._build_citations(retrieved_chunks)

        is_fallback = (
            "not covered in the" in raw_completion.lower()
            or "please reach out directly to the campaign" in raw_completion.lower()
            or not retrieved_chunks
        )

        if self.semantic_cache and query_vector is not None and not is_fallback:
            self.semantic_cache.set(request.query, query_vector, clean_answer)

        return RAGResponse(
            query=request.query,
            answer=clean_answer,
            champion_talking_point=None,
            citations=citations,
            model_used=self.llm_client.model_name,
            is_fallback=is_fallback,
        )

    def answer_query_stream(self, request: QueryRequest) -> Generator[str, None, None]:
        """Streams response tokens in real-time, checking semantic cache first."""
        query_vector = None
        if self.semantic_cache:
            query_vector = self.retriever.embedding_provider.embed_query(request.query)
            cached = self.semantic_cache.get(query_vector)
            if cached:
                words = cached.split(" ")
                for word in words:
                    time.sleep(0.015)
                    yield word + " "
                return

        retrieved_chunks = self.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            section_filter=request.filter_section,
        )
        prompt = build_rag_prompt(request.query, retrieved_chunks)

        tokens = []
        for token in self.llm_client.generate_stream(prompt=prompt, system_prompt=SYSTEM_PROMPT):
            tokens.append(token)
            yield token

        full_raw = "".join(tokens)
        clean = self._clean_answer(full_raw)
        is_fallback = "not covered in the" in full_raw.lower() or not retrieved_chunks
        if self.semantic_cache and query_vector is not None and clean and not is_fallback:
            self.semantic_cache.set(request.query, query_vector, clean)

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
