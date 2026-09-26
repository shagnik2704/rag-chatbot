"""In-memory BM25 lexical index for exact keyword and acronym matching."""

import re
from rank_bm25 import BM25Okapi
from src.core.exceptions import RetrievalError
from src.core.logging import setup_logger
from src.models.chunk import DocumentChunk

logger = setup_logger(__name__)


class BM25Index:
    """Sparse lexical search index based on BM25Okapi."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Simple, fast regex tokenization preserving alphanumeric keywords and acronyms."""
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Builds the BM25 index from a collection of DocumentChunks."""
        if not chunks:
            self._chunks = []
            self._bm25 = None
            return

        self._chunks = list(chunks)
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 lexical index with {len(self._chunks)} chunks.")

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Searches the lexical index and returns top_k matching chunks with BM25 scores."""
        if not self._bm25 or not self._chunks:
            return []

        try:
            tokens = self._tokenize(query)
            if not tokens:
                return []

            scores = self._bm25.get_scores(tokens)
            # Pair chunks with scores
            chunk_scores = list(zip(self._chunks, scores))
            # Filter out chunks with 0 score (no keyword match)
            matching = [(c, float(s)) for c, s in chunk_scores if s > 0.0]
            matching.sort(key=lambda x: x[1], reverse=True)
            return matching[:top_k]
        except Exception as e:
            raise RetrievalError(f"Error during BM25 search: {e}") from e

    def count(self) -> int:
        return len(self._chunks)
