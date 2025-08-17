"""Issue summarization module."""

from __future__ import annotations

from typing import List, Optional

from transformers import pipeline

# Preload the text generation pipeline once at module import to avoid
# re-initializing the model on every summarize call.
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
tokenizer = summarizer.tokenizer


def _generate(prompt: str) -> str:
    """Run the underlying model with safe tokenization and decoding."""

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(summarizer.model.device) for k, v in inputs.items()}
    output_ids = summarizer.model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


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
        prompt = base_prompt + "Transcript: " + chunk
        partial_summaries.append(_generate(prompt))

    combined = " ".join(partial_summaries)
    while True:
        final_prompt = base_prompt + "Partial summaries: " + combined
        token_len = len(tokenizer(final_prompt, return_tensors="pt")["input_ids"][0])
        if token_len <= tokenizer.model_max_length:
            return _generate(final_prompt)

        combined_chunks = _chunk_text(combined)
        new_partials: List[str] = []
        for part in combined_chunks:
            prompt = base_prompt + "Partial summaries: " + part
            new_partials.append(_generate(prompt))
        combined = " ".join(new_partials)
