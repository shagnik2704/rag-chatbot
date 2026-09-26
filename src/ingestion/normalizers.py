"""Text normalization and cleaning utilities."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalizes whitespace and standardizes special characters.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text preserving unicode symbols like currency.
    """
    if not text:
        return ""

    # Normalize unicode to NFKC (preserves special symbols like ₹ and letters)
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotation marks
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Standardize dashes while preserving em/en dash context
    text = text.replace("—", " — ").replace("–", " - ")

    # Normalize excessive horizontal whitespace (tabs, consecutive spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize consecutive blank lines to double newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
