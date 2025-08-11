"""Audio extraction module for meeting AI agent.

This implementation avoids depending on ``moviepy`` which is not available in
the execution environment.  Instead, the widely available ``ffmpeg`` command
line tool is used to extract the audio track from a video file.
"""

from __future__ import annotations

import subprocess
import shutil


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract the audio track from ``video_path`` and store it at
    ``audio_path``.

    The function relies on the ``ffmpeg`` binary.  If ``ffmpeg`` is not found on
    the system, a :class:`RuntimeError` is raised with installation guidance.
    """

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to extract audio but was not found. "
            "Please install ffmpeg and ensure it is in your PATH."
        )

    cmd = [ffmpeg_path, "-y", "-i", video_path, audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path
