"""Structural Q&A Chunker that parses hierarchical FAQ documents into semantic chunks."""

from abc import ABC, abstractmethod
from pathlib import Path
import re

from src.core.exceptions import ChunkingError
from src.core.logging import setup_logger
from src.models.chunk import ChunkMetadata, DocumentChunk

logger = setup_logger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, source_doc: str) -> list[DocumentChunk]:
        """Splits raw text into a list of DocumentChunks."""
        pass


class QAStructuralChunker(BaseChunker):
    """Parses structured Q&A / FAQ documents into discrete semantic chunks.

    Preserves hierarchical section headers, question IDs, question text, and answers.
    Also handles document preambles and footer notices.
    """

    # Matches section titles like 'A. UNDERSTANDING THE CAMPAIGN'
    SECTION_REGEX = re.compile(r"^[A-Z]\.\s+([^\n]+)$", re.MULTILINE)

    # Matches question headers like '1. What is...' or '12. How much...'
    QUESTION_REGEX = re.compile(r"(?:^|\n)(\d+)\.\s+([^\n]+)", re.MULTILINE)

    def chunk(self, text: str, source_doc: str) -> list[DocumentChunk]:
        if not text.strip():
            raise ChunkingError(f"Cannot chunk empty text from '{source_doc}'")

        chunks: list[DocumentChunk] = []

        # Find all section headers and their start positions
        section_matches = list(self.SECTION_REGEX.finditer(text))
        # Find all question headers and their start positions (filter out calendar years like 2026, 2027)
        raw_question_matches = list(self.QUESTION_REGEX.finditer(text))
        question_matches = [
            m for m in raw_question_matches if int(m.group(1)) <= 200
        ]

        if not question_matches:
            logger.warning(
                "No numbered Q&A pattern found in document. Falling back to paragraph chunking.",
                extra={"source": source_doc},
            )
            return self._fallback_paragraph_chunk(text, source_doc)

        # 1. Capture Preamble (text before the first section or question)
        first_content_start = (
            min(section_matches[0].start(), question_matches[0].start())
            if section_matches
            else question_matches[0].start()
        )
        preamble_text = text[:first_content_start].strip()
        if preamble_text:
            chunks.append(
                DocumentChunk(
                    id="faq-preamble",
                    text=f"[Document Overview & Guidelines]\n{preamble_text}",
                    metadata=ChunkMetadata(
                        section="PREAMBLE & GUIDELINES",
                        question_number=None,
                        question_text="Campaign Overview & Guidelines",
                        source_document=source_doc,
                        chunk_type="preamble",
                        tags=["guidelines", "contacts", "overview", "confidentiality"],
                    ),
                )
            )

        # Helper to determine current section given a char index
        def get_section_at(idx: int) -> str:
            current_section = "GENERAL"
            for sm in section_matches:
                if sm.start() <= idx:
                    current_section = sm.group(0).strip()
                else:
                    break
            return current_section

        # 2. Iterate through questions and construct each QA chunk
        for i, q_match in enumerate(question_matches):
            q_num = int(q_match.group(1))
            q_text = q_match.group(2).strip()

            content_start = q_match.end()
            # The answer spans until the next question or the end of the text
            content_end = (
                question_matches[i + 1].start()
                if i + 1 < len(question_matches)
                else len(text)
            )

            raw_answer = text[content_start:content_end].strip()

            # Clean any section header that might sit between this question and the next
            # e.g., if a new section begins right after this question's answer
            clean_answer = self._strip_trailing_section_headers(raw_answer)
            section_name = get_section_at(q_match.start())

            # Format rich chunk text with structured preamble for better embedding & generation
            chunk_body = (
                f"[Section: {section_name}]\n"
                f"Question {q_num}: {q_text}\n\n"
                f"Answer:\n{clean_answer}"
            )

            # Auto-tagging common keywords
            tags = self._extract_tags(chunk_body)

            chunks.append(
                DocumentChunk(
                    id=f"faq-q{q_num}",
                    text=chunk_body,
                    metadata=ChunkMetadata(
                        section=section_name,
                        question_number=q_num,
                        question_text=q_text,
                        source_document=source_doc,
                        chunk_type="qa_pair",
                        tags=tags,
                    ),
                )
            )

        logger.info(
            f"Successfully parsed {len(chunks)} chunks ({len(question_matches)} questions) from '{source_doc}'."
        )
        return chunks

    def _strip_trailing_section_headers(self, answer_text: str) -> str:
        """Removes next section title if accidentally included in answer span."""
        lines = answer_text.split("\n")
        filtered_lines = []
        for line in lines:
            if self.SECTION_REGEX.match(line.strip()):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    def _extract_tags(self, content: str) -> list[str]:
        """Extracts significant keywords for tagging."""
        lowered = content.lower()
        keyword_map = {
            "cyaag": "cYAAG",
            "spoken tutorial": "Spoken Tutorial",
            "wheels global": "WHEELS Global Foundation",
            "edupyramids": "EduPyramids",
            "ieee": "IEEE Standard",
            "foss": "FOSS",
            "cost": "Pricing & Contribution",
            "35,000": "School Cost",
            "maharashtra": "State Implementation",
            "madhya pradesh": "State Implementation",
            "mp": "State Implementation",
            "champion": "Champion Role",
            "team": "Team Mobilization",
            "points": "Gamification & Points",
            "timeline": "Campaign Timeline",
            "inr": "Donation Currency & Policy",
            "usd": "Donation Currency & Policy",
            "bank account": "Donation Currency & Policy",
        }
        tags = set()
        for key, tag in keyword_map.items():
            if key in lowered:
                tags.add(tag)
        return sorted(list(tags))

    def _fallback_paragraph_chunk(self, text: str, source_doc: str) -> list[DocumentChunk]:
        """Fallback for non-FAQ structured text."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for i, p in enumerate(paragraphs):
            chunks.append(
                DocumentChunk(
                    id=f"chunk-{i+1}",
                    text=p,
                    metadata=ChunkMetadata(
                        section="GENERAL",
                        question_number=None,
                        question_text=None,
                        source_document=source_doc,
                        chunk_type="paragraph",
                        tags=[],
                    ),
                )
            )
        return chunks
