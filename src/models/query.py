"""Domain schemas for queries and retrieval results."""

from pydantic import BaseModel, Field
from src.models.chunk import DocumentChunk


class QueryRequest(BaseModel):
    """Encapsulates a user inquiry and retrieval preferences."""

    query: str = Field(..., min_length=1, description="The user question to answer")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve")
    filter_section: str | None = Field(default=None, description="Optional section filter")
    generate_talking_points: bool = Field(default=True, description="Whether to include champion talking points")


class RetrievedChunk(BaseModel):
    """A scored and ranked chunk retrieved from the hybrid index."""

    chunk: DocumentChunk = Field(..., description="The underlying document chunk")
    dense_score: float = Field(default=0.0, description="Dense vector similarity score")
    sparse_score: float = Field(default=0.0, description="BM25 keyword relevance score")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion combined score")
    rank: int = Field(default=1, description="Final rank after fusion (1-indexed)")
