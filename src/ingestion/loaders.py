"""Multi-format document loaders for PDF, DOCX, and Text files."""

from abc import ABC, abstractmethod
from pathlib import Path
import docx
import pypdf

from src.core.exceptions import DocumentIngestionError
from src.core.logging import setup_logger
from src.ingestion.normalizers import clean_text

logger = setup_logger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract interface for file loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """Loads and extracts raw text from the specified file path."""
        pass


class DocxDocumentLoader(BaseDocumentLoader):
    """Loads and extracts text from Word (.docx) documents."""

    def load(self, file_path: Path) -> str:
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            return clean_text(full_text)
        except Exception as e:
            raise DocumentIngestionError(f"Failed to load DOCX document '{file_path}': {e}") from e


class PDFDocumentLoader(BaseDocumentLoader):
    """Loads and extracts text from PDF documents using pypdf."""

    def load(self, file_path: Path) -> str:
        try:
            reader = pypdf.PdfReader(str(file_path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            full_text = "\n\n".join(pages_text)
            return clean_text(full_text)
        except Exception as e:
            raise DocumentIngestionError(f"Failed to load PDF document '{file_path}': {e}") from e


class TextDocumentLoader(BaseDocumentLoader):
    """Loads plain text / markdown files."""

    def load(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return clean_text(f.read())
        except Exception as e:
            raise DocumentIngestionError(f"Failed to load text document '{file_path}': {e}") from e


class DocumentLoaderFactory:
    """Factory to dispatch appropriate loader based on file suffix."""

    _LOADERS: dict[str, type[BaseDocumentLoader]] = {
        ".docx": DocxDocumentLoader,
        ".pdf": PDFDocumentLoader,
        ".txt": TextDocumentLoader,
        ".md": TextDocumentLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseDocumentLoader:
        suffix = file_path.suffix.lower()
        loader_cls = cls._LOADERS.get(suffix)
        if not loader_cls:
            supported = ", ".join(cls._LOADERS.keys())
            raise DocumentIngestionError(
                f"Unsupported file type '{suffix}' for file '{file_path}'. Supported types: {supported}"
            )
        return loader_cls()
