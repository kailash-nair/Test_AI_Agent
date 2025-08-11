"""Text normalization module for cleaning transcripts."""

from __future__ import annotations

import re

FILLER_WORDS = {
    "um",
    "uh",
    "ah",
    "er",
    "mmm",
}

TERMINOLOGY_MAP = {
    "setting up": "deployment",
    "set up": "deploy",
    "setup": "deployment",
}


def normalize_text(text: str) -> str:
    """Remove filler words and standardize corporate terminology."""

    def remove_fillers(token: str) -> bool:
        return token.lower() not in FILLER_WORDS

    tokens = filter(remove_fillers, re.split(r"\s+", text))
    cleaned = " ".join(tokens)
    for src, tgt in TERMINOLOGY_MAP.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", tgt, cleaned, flags=re.IGNORECASE)
    return cleaned
