"""Whisper transcription module with Malayalam support."""

from __future__ import annotations


def transcribe(audio_path: str, model_name: str = "small") -> str:
    """Transcribe Malayalam audio to English using Whisper.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe.
    model_name:
        Name of the Whisper model to load.  Defaults to ``"small"``.

    The function requires the ``whisper`` package.  If it is not installed a
    :class:`RuntimeError` is raised with instructions on how to install it.
    """

    try:
        import whisper
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The 'whisper' package is required for transcription. "
            "Install it with 'pip install -U openai-whisper'."
        ) from exc

    model = whisper.load_model(model_name)
    # Specify language to ensure Malayalam recognition
    result = model.transcribe(audio_path, task="translate", language="ml")
    return result["text"].strip()
