"""Domain schemas for chunks and metadata."""

from typing import Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attributes associated with a document chunk."""

    section: str = Field(..., description="High-level section header (e.g. 'A. UNDERSTANDING THE CAMPAIGN')")
    question_number: int | None = Field(None, description="Question number if applicable (e.g. 12)")
    question_text: str | None = Field(None, description="The specific question string")
    source_document: str = Field(..., description="Original filename or path of the document")
    page_numbers: list[int] = Field(default_factory=list, description="Pages containing this chunk")
    chunk_type: str = Field(default="qa_pair", description="Type of chunk: 'qa_pair', 'section_overview', 'footer'")
    tags: list[str] = Field(default_factory=list, description="Topic keywords or entities")

    def to_flat_dict(self) -> dict[str, Any]:
        """Flattens metadata for vector databases like Chroma which require scalar attributes."""
        return {
            "section": self.section,
            "question_number": self.question_number if self.question_number is not None else -1,
            "question_text": self.question_text or "",
            "source_document": self.source_document,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
            "chunk_type": self.chunk_type,
            "tags": ",".join(self.tags),
        }

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        """Reconstructs ChunkMetadata from flat ChromaDB metadata."""
        pages_str = data.get("page_numbers", "")
        pages = [int(p) for p in pages_str.split(",") if p.strip().isdigit()]
        q_num = data.get("question_number")
        if q_num == -1:
            q_num = None
        tags_str = data.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        return cls(
            section=data.get("section", ""),
            question_number=q_num,
            question_text=data.get("question_text") or None,
            source_document=data.get("source_document", ""),
            page_numbers=pages,
            chunk_type=data.get("chunk_type", "qa_pair"),
            tags=tags,
        )


class DocumentChunk(BaseModel):
    """Discrete textual chunk indexed in vector and lexical databases."""

    id: str = Field(..., description="Unique chunk identifier (e.g. 'faq-q12')")
    text: str = Field(..., description="The complete text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Rich metadata payload")
    embedding: list[float] | None = Field(default=None, description="Vector embedding representation")
