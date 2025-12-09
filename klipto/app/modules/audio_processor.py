
import os
from typing import Dict, Any
from faster_whisper import WhisperModel
from app.config import settings

class AudioProcessor:
    """Handles audio transcription using Faster Whisper (Local)."""

    def __init__(self):
        # Use CPU with int8 for the sandbox environment
        # In production with GPU, use device="cuda", compute_type="float16"
        self.model_size = settings.WHISPER_MODEL # e.g. "base" or "small"
        print(f"Loading faster-whisper model: {self.model_size}")

        # Check if CUDA is available, otherwise CPU
        # For this sandbox, force CPU to avoid errors if GPU libs are missing drivers
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribes audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing {audio_path}...")
        segments, info = self.model.transcribe(audio_path, beam_size=5)

        # faster-whisper returns a generator, we must consume it
        segment_list = list(segments)

        # Format similar to OpenAI response for compatibility
        formatted_segments = []
        full_text = ""

        for segment in segment_list:
            full_text += segment.text + " "
            formatted_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })

        return {
            'text': full_text.strip(),
            'segments': formatted_segments,
            'language': info.language
        }
