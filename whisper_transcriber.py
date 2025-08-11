"""Whisper transcription module with Malayalam support."""

import whisper


def transcribe(audio_path: str) -> str:
    """Transcribe Malayalam audio to English using Whisper."""
    model = whisper.load_model("small")
    # Specify language to ensure Malayalam recognition
    result = model.transcribe(audio_path, task="translate", language="ml")
    return result["text"].strip()
