"""Issue summarization module."""

from __future__ import annotations

from typing import List, Optional

from transformers import pipeline


def summarize(
    text: str,
    meeting_date: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> str:
    """Generate structured summary from transcript."""
    summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = (
        "You are a corporate meeting assistant. Using the transcript, summarize each "
        "issue discussed and list decisions and **action items** with owners and "
        "deadlines. Format with headings and bullet points. "
    )
    if meeting_date:
        prompt += f"Meeting date: {meeting_date}. "
    if attendees:
        prompt += f"Attendees: {', '.join(attendees)}. "
    prompt += "Transcript: "
    output = summarizer(prompt + text, max_length=512)[0]["generated_text"]
    return output
