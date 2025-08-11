"""Audio extraction module for meeting AI agent."""

from moviepy.editor import VideoFileClip


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract audio track from the video file."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path
