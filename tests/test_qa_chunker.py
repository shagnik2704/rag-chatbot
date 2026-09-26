"""Unit tests for QAStructuralChunker."""

import pytest
from src.ingestion.qa_chunker import QAStructuralChunker


SAMPLE_FAQ_TEXT = """Future-Ready Children
Champion FAQ & Talking Points
PLEASE DO NOT SHARE THE ENTIRE DOCUMENT!
Contact: saisudha@edupyramids.org

A. UNDERSTANDING THE CAMPAIGN

1. What is the Future-Ready Children campaign?
Future-Ready Children is an initiative of WHEELS Global Foundation, in partnership with EduPyramids.
The campaign aims to reach 5 million children across 10,000 schools.

2. Why is this campaign important?
Every child has potential. But realising that potential requires opportunity.

C. SUPPORTING THE CAMPAIGN

12. How much does it cost to support a school?
A contribution of ₹35,000 / US$365 can support approximately 500 children in one school.
"""


def test_qa_chunker_parses_sample_faq():
    chunker = QAStructuralChunker()
    chunks = chunker.chunk(SAMPLE_FAQ_TEXT, source_doc="test_doc.txt")

    assert len(chunks) == 4  # 1 preamble + 3 questions

    # Preamble chunk
    preamble = chunks[0]
    assert preamble.id == "faq-preamble"
    assert preamble.metadata.section == "PREAMBLE & GUIDELINES"
    assert "PLEASE DO NOT SHARE" in preamble.text

    # Question 1
    q1 = chunks[1]
    assert q1.id == "faq-q1"
    assert q1.metadata.question_number == 1
    assert q1.metadata.section == "A. UNDERSTANDING THE CAMPAIGN"
    assert "WHEELS Global Foundation" in q1.text

    # Question 12
    q12 = chunks[3]
    assert q12.id == "faq-q12"
    assert q12.metadata.question_number == 12
    assert q12.metadata.section == "C. SUPPORTING THE CAMPAIGN"
    assert "₹35,000" in q12.text
    assert "School Cost" in q12.metadata.tags


def test_qa_chunker_filters_out_years():
    sample_with_years = """A. TIMELINE
17. What is the timeline?
Fundraising: 15 August 2026 - 26 January 2027.
2027. Monitoring and reporting continues until 2028.
18. How will impact be measured?
Impact is measured through key metrics.
"""
    chunker = QAStructuralChunker()
    chunks = chunker.chunk(sample_with_years, source_doc="test_timeline.txt")

    question_numbers = [c.metadata.question_number for c in chunks if c.metadata.question_number]
    assert 2027 not in question_numbers
    assert 17 in question_numbers
    assert 18 in question_numbers
