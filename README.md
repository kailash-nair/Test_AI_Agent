# Test_AI_Agent

This repository contains a modular Python meeting assistant. The main script
`meeting_ai_agent.py` orchestrates dedicated pipeline modules to process a
Microsoft Teams meeting recording:

1. **Audio Extraction** – `audio_extractor.py` isolates the audio track from the
   video file.
2. **Speech-to-Text Translation** – `whisper_transcriber.py` transcribes
   Malayalam speech to English using Whisper with the language explicitly set to
   Malayalam for accurate recognition.
3. **Register Transformation** – `text_normalizer.py` removes filler words and
   normalizes corporate terminology.
4. **Issue-wise Summarization** – `issue_summarizer.py` uses a text generation
   model to produce structured summaries of issues, decisions and action items.

### Usage

```bash
python meeting_ai_agent.py <video_file> --date YYYY-MM-DD --attendees Alice Bob --output summary.txt
```

The script prints the structured summary and optionally writes it to a file when
`--output` is provided.
