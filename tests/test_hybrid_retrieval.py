"""Unit tests for BM25 and Hybrid Retrieval."""

from unittest.mock import MagicMock
from src.embeddings.base import BaseEmbeddingProvider
from src.models.chunk import ChunkMetadata, DocumentChunk
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.base import BaseVectorStore


def _create_dummy_chunk(chunk_id: str, q_num: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            section="SECTION A",
            question_number=q_num,
            question_text=f"Question {q_num}",
            source_document="doc.docx",
        ),
    )


def test_bm25_exact_keyword_matching():
    index = BM25Index()
    c1 = _create_dummy_chunk("c1", 1, "The cYAAG platform mobilizes changemakers at scale.")
    c2 = _create_dummy_chunk("c2", 2, "Spoken Tutorial pedagogy is recognized as IEEE Standard P2955.")
    c3 = _create_dummy_chunk("c3", 3, "A school sponsorship costs 35000 rupees.")

    index.index_chunks([c1, c2, c3])

    # Search for acronym cYAAG
    results_cyaag = index.search("cYAAG platform", top_k=2)
    assert len(results_cyaag) > 0
    assert results_cyaag[0][0].id == "c1"

    # Search for IEEE standard
    results_ieee = index.search("IEEE standard", top_k=2)
    assert len(results_ieee) > 0
    assert results_ieee[0][0].id == "c2"


def test_hybrid_retriever_rrf_fusion():
    c1 = _create_dummy_chunk("c1", 1, "Chunk 1: High dense match, low sparse")
    c2 = _create_dummy_chunk("c2", 2, "Chunk 2: High sparse match, low dense")

    mock_vector_store = MagicMock(spec=BaseVectorStore)
    # Dense returns c1 rank 1, c2 rank 2
    mock_vector_store.similarity_search_by_vector.return_value = [(c1, 0.95), (c2, 0.60)]

    mock_embedder = MagicMock(spec=BaseEmbeddingProvider)
    mock_embedder.embed_query.return_value = [0.1, 0.2]

    bm25 = BM25Index()
    bm25.index_chunks([c1, c2])

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_provider=mock_embedder,
        bm25_index=bm25,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
    )

    results = retriever.retrieve("Chunk 1", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert results[0].rrf_score > results[1].rrf_score
