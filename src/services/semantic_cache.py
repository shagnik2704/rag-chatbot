"""Thread-safe semantic cache using cosine similarity over query embeddings."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import numpy as np
from src.core.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class CacheEntry:
    query: str
    query_vector: list[float]
    answer: str


class SemanticCache:
    """In-memory semantic vector cache with optional disk persistence."""

    def __init__(
        self,
        similarity_threshold: float = 0.93,
        persist_path: Path | str = "storage/semantic_cache.json",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.persist_path = Path(persist_path)
        self._lock = threading.Lock()
        self._entries: list[CacheEntry] = []
        self._vector_matrix: np.ndarray | None = None
        self._load()

    def get(self, query_vector: list[float]) -> str | None:
        """Looks up a semantically similar cached response.

        Returns:
            Cached answer string if similarity >= threshold, else None.
        """
        with self._lock:
            if not self._entries or self._vector_matrix is None:
                return None

            q_vec = np.array(query_vector, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            # Matrix-vector multiplication for instant batch cosine similarity
            similarities = np.dot(self._vector_matrix, q_vec)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.similarity_threshold:
                hit = self._entries[best_idx]
                logger.info(
                    f"Semantic cache hit (sim: {best_sim:.3f}) for cached query: '{hit.query}'"
                )
                return hit.answer

            return None

    def set(self, query: str, query_vector: list[float], answer: str) -> None:
        """Caches a new query and answer pair."""
        if not answer.strip():
            return

        with self._lock:
            q_vec = np.array(query_vector, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

            entry = CacheEntry(
                query=query.strip(),
                query_vector=q_vec.tolist(),
                answer=answer.strip(),
            )
            self._entries.append(entry)

            # Update matrix
            vec_2d = q_vec.reshape(1, -1)
            if self._vector_matrix is None:
                self._vector_matrix = vec_2d
            else:
                self._vector_matrix = np.vstack([self._vector_matrix, vec_2d])

            self._save()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._vector_matrix = None
            if self.persist_path.exists():
                self.persist_path.unlink()

    def _save(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(e) for e in self._entries]
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist semantic cache: {e}")

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [CacheEntry(**item) for item in data]
            if self._entries:
                vecs = [e.query_vector for e in self._entries]
                self._vector_matrix = np.array(vecs, dtype=np.float32)
            logger.info(f"Loaded {len(self._entries)} entries from semantic cache disk file.")
        except Exception as e:
            logger.warning(f"Failed to load semantic cache: {e}")
