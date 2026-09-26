"""Application configuration managed via Pydantic Settings."""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for RAG chatbot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Sarvam AI API Configuration
    sarvam_api_key: str = Field(
        default="",
        description="API key for Sarvam AI",
    )
    sarvam_base_url: str = Field(
        default="https://api.sarvam.ai",
        description="Base URL for Sarvam AI API",
    )
    sarvam_model: str = Field(
        default="glm5.3",
        description="Model identifier for Sarvam inference",
    )
    sarvam_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation",
    )
    sarvam_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens to generate",
    )
    sarvam_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Timeout for Sarvam API requests in seconds",
    )

    # Embedding & Storage Configuration
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model ID for dense embeddings",
    )
    chroma_persist_directory: Path = Field(
        default=Path("storage/chroma_db"),
        description="Directory path for persistent ChromaDB storage",
    )
    chroma_collection_name: str = Field(
        default="future_ready_faq",
        description="ChromaDB collection name",
    )

    # Retrieval Configuration (Hybrid: Dense + BM25)
    top_k_retrieval: int = Field(
        default=4,
        gt=0,
        description="Number of top context chunks to retrieve",
    )
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense vector retrieval in hybrid fusion",
    )
    sparse_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 sparse retrieval in hybrid fusion",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="Constant parameter for Reciprocal Rank Fusion",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @field_validator("chroma_persist_directory", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v) if isinstance(v, str) else v


def get_settings() -> Settings:
    """Factory function for retrieving cached application settings."""
    return Settings()
