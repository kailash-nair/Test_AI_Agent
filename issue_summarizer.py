"""Issue summarization module."""

from __future__ import annotations

from typing import List

from transformers import pipeline


def summarize(text: str, meeting_date: str, attendees: List[str]) -> str:
    """Generate structured summary from transcript."""
    summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = (
        "You are a corporate meeting assistant. Using the transcript, summarize each "
        "issue discussed and list decisions and **action items** with owners and "
        "deadlines. Format with headings and bullet points. Meeting date: "
        f"{meeting_date}. Attendees: {', '.join(attendees)}. Transcript: "
    )
    output = summarizer(prompt + text, max_length=512)[0]["generated_text"]
    return output
