"""Ingestion module providing loaders, normalizers, and chunkers."""

from src.ingestion.loaders import (
    BaseDocumentLoader,
    DocxDocumentLoader,
    DocumentLoaderFactory,
    PDFDocumentLoader,
    TextDocumentLoader,
)
from src.ingestion.normalizers import clean_text
from src.ingestion.qa_chunker import BaseChunker, QAStructuralChunker

__all__ = [
    "BaseDocumentLoader",
    "DocxDocumentLoader",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "DocumentLoaderFactory",
    "clean_text",
    "BaseChunker",
    "QAStructuralChunker",
]
