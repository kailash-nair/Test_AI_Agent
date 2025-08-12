"""Issue summarization module."""

from __future__ import annotations

from typing import List, Optional

from transformers import pipeline

# Preload the text generation pipeline once at module import to avoid
# re-initializing the model on every summarize call.
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
tokenizer = summarizer.tokenizer


def _chunk_text(text: str, max_tokens: int = 350) -> List[str]:
    """Split ``text`` into chunks of approximately ``max_tokens`` tokens.

    The chunking is performed using the model's tokenizer to stay within the
    model's context window and avoid truncation.
    """

    tokenized = tokenizer(
        text,
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"][0]

    chunks: List[str] = []
    for i in range(0, len(tokenized), max_tokens):
        piece = tokenized[i : i + max_tokens]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))

    return chunks


def summarize(
    text: str,
    meeting_date: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> str:
    """Generate structured summary from transcript."""
    base_prompt = (
        "You are a corporate meeting assistant. Using the transcript, summarize each "
        "issue discussed and list decisions and **action items** with owners and "
        "deadlines. Format with headings and bullet points. "
    )
    if meeting_date:
        base_prompt += f"Meeting date: {meeting_date}. "
    if attendees:
        base_prompt += f"Attendees: {', '.join(attendees)}. "

    chunks = _chunk_text(text)
    partial_summaries: List[str] = []
    for chunk in chunks:
        prompt = base_prompt + "Transcript: "
        partial = summarizer(prompt + chunk, max_length=512)[0]["generated_text"]
        partial_summaries.append(partial)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    final_prompt = (
        base_prompt
        + "Partial summaries: "
        + " ".join(partial_summaries)
    )
    final_summary = summarizer(final_prompt, max_length=512)[0]["generated_text"]
    return final_summary
