"""Enterprise prompt templates and builders with strict grounding rules."""

from src.models.query import RetrievedChunk

SYSTEM_PROMPT = """You are the official AI Assistant for the "Future-Ready Children" campaign, an initiative of WHEELS Global Foundation in partnership with EduPyramids.

Your mission is to provide accurate, grounded, articulate, and welcoming answers to supporters, partners, schools, and donors based strictly on the provided campaign documentation.

CRITICAL INSTRUCTIONS:
1. STRICT GROUNDING: Rely ONLY on the information present in the [CONTEXT] provided. Do NOT fabricate, extrapolate, or speculate on numbers, dates, partner agreements, or operational promises.
2. PRESERVE EXACT FIGURES: Always maintain exact figures when present (e.g., ₹35,000 / US$365 per school, ~500 children per school, 10,000 schools, 5 million children, IEEE 2955-2025 standard, dates like 15 August 2026 – 26 January 2027 and November 2026 – March 2028).
3. CLEAN CLIENT-FACING STYLE: Provide a direct, well-structured, and comprehensive response. Use clean paragraphs, bullet points, and bold text where appropriate for readability.
4. NO INTERNAL META-LABELS: Do NOT output internal sectional labels like "Direct Answer:", "Champion Talking Point:", or "Citations:". Simply answer the question naturally and authoritatively.
5. FALLBACK ESCALATION:
   If the user's question cannot be answered from the provided context, DO NOT guess. Politely state:
   "This specific detail is not covered in the campaign FAQ document. For personalized assistance, please reach out directly to the campaign leadership team:
   - sujathan@wheelsglobal.org
   - saisudha@edupyramids.org"
"""


def build_rag_prompt(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Builds a formatted prompt combining retrieved context chunks with the user question."""
    if not retrieved_chunks:
        context_str = "No relevant context found in document."
    else:
        context_parts = []
        for i, item in enumerate(retrieved_chunks, start=1):
            c = item.chunk
            q_num_str = f"Q{c.metadata.question_number}: " if c.metadata.question_number is not None else ""
            header = f"--- Context Chunk {i} [{c.metadata.section} | {q_num_str}{c.metadata.question_text or 'Overview'}] ---"
            context_parts.append(f"{header}\n{c.text}\n")
        context_str = "\n".join(context_parts)

    return f"""[CONTEXT]
{context_str}
[END CONTEXT]

User Question: {query}

Please provide your grounded, client-friendly answer as instructed."""
