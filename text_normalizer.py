"""Text normalization module for cleaning transcripts."""

from __future__ import annotations

import re

# Common filler words and verbal pauses that should be removed from transcripts
FILLER_WORDS = {
    "um",
    "uh",
    "ah",
    "er",
    "mmm",
    "umm",
    "uhh",
    "hmm",
    "like",
    "so",
    "actually",
    "basically",
    "ok",
    "okay",
}

# Corporate terminology and abbreviations mapped to their preferred forms
TERMINOLOGY_MAP = {
    "setting up": "deployment",
    "set up": "deploy",
    "setup": "deployment",
    "config": "configuration",
    "configs": "configurations",
    "ai": "artificial intelligence",
    "api": "application programming interface",
    "db": "database",
    "prod": "production",
    "dev": "development",
}


def normalize_text(text: str) -> str:
    """Normalize raw text by cleaning fillers, casing, punctuation, whitespace, and terminology."""

    # Standardize casing
    cleaned = text.lower()

    # Remove stray punctuation
    cleaned = re.sub(r"[^\w\s]", "", cleaned)

    # Collapse repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    def remove_fillers(token: str) -> bool:
        return token not in FILLER_WORDS

    tokens = filter(remove_fillers, cleaned.split())
    cleaned = " ".join(tokens)

    for src, tgt in TERMINOLOGY_MAP.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", tgt, cleaned, flags=re.IGNORECASE)

    return cleaned
