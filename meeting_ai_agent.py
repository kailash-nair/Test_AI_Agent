"""AI meeting summarizer script."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

from audio_extractor import extract_audio
from whisper_transcriber import transcribe
from text_normalizer import normalize_text
from issue_summarizer import summarize


@dataclass
class SummaryRequest:
    video_path: str
    meeting_date: Optional[str] = None
    attendees: Optional[List[str]] = None
    save_transcript: str | None = None
    save_summary: str | None = None
    model_name: str = "small"


def run(args: SummaryRequest) -> tuple[str, str]:
    audio_path = os.path.splitext(args.video_path)[0] + ".wav"
    extract_audio(args.video_path, audio_path)
    transcript = transcribe(audio_path, model_name=args.model_name)
    cleaned = normalize_text(transcript)
    summary = summarize(cleaned, args.meeting_date, args.attendees)
    if args.save_transcript:
        with open(args.save_transcript, "w", encoding="utf-8") as f:
            f.write(transcript)
    if args.save_summary:
        with open(args.save_summary, "w", encoding="utf-8") as f:
            f.write(summary)
    return transcript, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting summarizer AI agent")
    parser.add_argument("video", nargs="?", help="Path to Teams video recording")
    parser.add_argument("--date", help="Meeting date")
    parser.add_argument(
        "--attendees",
        nargs="*",
        help="List of attendees",
    )
    parser.add_argument("--save-transcript", help="File to save the transcript")
    parser.add_argument("--save-summary", help="File to save the summary")
    parser.add_argument(
        "--model-name",
        default="small",
        help="Whisper model to use for transcription",
    )
    args_ns = parser.parse_args()
    if not args_ns.video:
        args_ns.video = input("Video file: ").strip()
    args = SummaryRequest(
        video_path=args_ns.video,
        meeting_date=args_ns.date,
        attendees=args_ns.attendees,
        save_transcript=args_ns.save_transcript,
        save_summary=args_ns.save_summary,
        model_name=args_ns.model_name,
    )
    transcript, summary = run(args)
    print("Transcript:", transcript)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
