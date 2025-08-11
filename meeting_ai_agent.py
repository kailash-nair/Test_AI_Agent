"""AI meeting summarizer script."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

from audio_extractor import extract_audio
from whisper_transcriber import transcribe
from text_normalizer import normalize_text
from issue_summarizer import summarize


@dataclass
class SummaryRequest:
    video_path: str
    meeting_date: str
    attendees: List[str]
    output_path: str | None = None


def run(args: SummaryRequest) -> str:
    audio_path = os.path.splitext(args.video_path)[0] + ".wav"
    extract_audio(args.video_path, audio_path)
    transcript = transcribe(audio_path)
    cleaned = normalize_text(transcript)
    summary = summarize(cleaned, args.meeting_date, args.attendees)
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting summarizer AI agent")
    parser.add_argument("video", help="Path to Teams video recording")
    parser.add_argument("--date", required=True, help="Meeting date")
    parser.add_argument(
        "--attendees",
        nargs="+",
        required=True,
        help="List of attendees",
    )
    parser.add_argument("--output", help="Output file for the summary")
    args_ns = parser.parse_args()
    args = SummaryRequest(
        video_path=args_ns.video,
        meeting_date=args_ns.date,
        attendees=args_ns.attendees,
        output_path=args_ns.output,
    )
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
