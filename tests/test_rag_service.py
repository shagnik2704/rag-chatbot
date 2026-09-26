"""Unit and integration tests for RAGService."""

from unittest.mock import MagicMock
from src.llm.sarvam_client import MockLLMClient
from src.models.chunk import ChunkMetadata, DocumentChunk
from src.models.query import QueryRequest, RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.rag_service import RAGService


def test_rag_service_generates_grounded_response():
    mock_retriever = MagicMock(spec=HybridRetriever)
    chunk = DocumentChunk(
        id="faq-q12",
        text="[Section: C. SUPPORTING THE CAMPAIGN]\nQuestion 12: How much does it cost to support a school?\n\nAnswer: A contribution of ₹35,000 / US$365 can support approximately 500 children in one school.",
        metadata=ChunkMetadata(
            section="C. SUPPORTING THE CAMPAIGN",
            question_number=12,
            question_text="How much does it cost to support a school?",
            source_document="faq.docx",
        ),
    )
    retrieved_chunk = RetrievedChunk(chunk=chunk, dense_score=0.9, sparse_score=5.2, rrf_score=0.03, rank=1)
    mock_retriever.retrieve.return_value = [retrieved_chunk]

    canned_llm_text = (
        "**Direct Answer**:\nA contribution of ₹35,000 / US$365 supports approximately 500 children in one school.\n\n"
        "**Champion Talking Point**:\nFor just ₹35,000, we can bring IEEE-standard technology education to 500 children who otherwise wouldn't have this chance.\n\n"
        "**Citations**:\n[Section: C. SUPPORTING THE CAMPAIGN | Q12]"
    )
    llm = MockLLMClient(canned_response=canned_llm_text)

    rag = RAGService(retriever=mock_retriever, llm_client=llm)
    req = QueryRequest(query="How much to support a school?", top_k=2)
    response = rag.answer_query(req)

    assert "₹35,000" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].question_number == 12
    assert response.is_fallback is False


def test_rag_service_fallback_detection():
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = []

    fallback_text = (
        "This specific detail is not covered in the Champion FAQ document. "
        "Please reach out directly to the campaign team: sujathan@wheelsglobal.org"
    )
    llm = MockLLMClient(canned_response=fallback_text)

    rag = RAGService(retriever=mock_retriever, llm_client=llm)
    req = QueryRequest(query="Can we build swimming pools in schools?", top_k=2)
    response = rag.answer_query(req)

    assert response.is_fallback is True
    assert "sujathan@wheelsglobal.org" in response.fallback_contacts
