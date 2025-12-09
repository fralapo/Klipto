"""
Audio Processor Module
Handles audio transcription using Faster Whisper (local) or cloud APIs.
"""

import os
from typing import Dict, Any, Optional, List
from app.config import settings


class AudioProcessor:
    """
    Handles audio transcription using Faster Whisper.

    Supports multiple Whisper model sizes:
    - tiny: Fastest, ~1GB VRAM, lower accuracy
    - base: Fast, ~1GB VRAM, good accuracy (default)
    - small: Medium, ~2GB VRAM, better accuracy
    - medium: Slow, ~5GB VRAM, high accuracy
    - large-v3: Slowest, ~10GB VRAM, best accuracy

    Can run on CPU (slower) or GPU with CUDA (faster).
    """

    # Model configurations
    MODEL_INFO = {
        "tiny": {"vram": "~1GB", "speed": "fastest", "accuracy": "low"},
        "base": {"vram": "~1GB", "speed": "fast", "accuracy": "good"},
        "small": {"vram": "~2GB", "speed": "medium", "accuracy": "better"},
        "medium": {"vram": "~5GB", "speed": "slow", "accuracy": "high"},
        "large-v3": {"vram": "~10GB", "speed": "slowest", "accuracy": "best"},
    }

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the audio processor.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        from faster_whisper import WhisperModel

        self.model_size = model_size or settings.WHISPER_MODEL or "base"
        self.device = device or self._detect_device()
        self.compute_type = self._get_compute_type()

        print(f"🎤 Loading Faster-Whisper model: {self.model_size}")
        print(f"   Device: {self.device}, Compute: {self.compute_type}")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

        print(f"✅ Whisper model loaded successfully")

    def _detect_device(self) -> str:
        """Detects the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                print("🎮 CUDA GPU detected")
                return "cuda"
        except ImportError:
            pass

        print("💻 Using CPU (no GPU detected or torch not available)")
        return "cpu"

    def _get_compute_type(self) -> str:
        """Gets the optimal compute type for the device."""
        if self.device == "cuda":
            return "float16"  # Faster on GPU
        return "int8"  # More efficient on CPU

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        word_timestamps: bool = False,
        vad_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribes audio file to text with timestamps.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code (e.g., 'en', 'it', 'es') or None for auto-detect
            task: 'transcribe' or 'translate' (to English)
            beam_size: Beam size for decoding (higher = more accurate, slower)
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Use voice activity detection to filter silence

        Returns:
            Dict with 'text', 'segments', 'language', and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        print(f"🎵 Transcribing: {audio_path} ({file_size:.1f} MB)")

        # Transcribe with faster-whisper
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Minimum silence to split
                speech_pad_ms=400,  # Padding around speech
            ) if vad_filter else None
        )

        # Consume generator and format segments
        segments = []
        full_text_parts = []

        for segment in segments_gen:
            segment_data = {
                'start': round(segment.start, 2),
                'end': round(segment.end, 2),
                'text': segment.text.strip()
            }

            # Add word-level timestamps if requested
            if word_timestamps and segment.words:
                segment_data['words'] = [
                    {
                        'word': word.word,
                        'start': round(word.start, 2),
                        'end': round(word.end, 2),
                        'probability': round(word.probability, 3)
                    }
                    for word in segment.words
                ]

            segments.append(segment_data)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        result = {
            'text': full_text,
            'segments': segments,
            'language': info.language,
            'language_probability': round(info.language_probability, 3),
            'duration': round(info.duration, 2),
            'num_segments': len(segments),
            'model': self.model_size,
            'device': self.device
        }

        print(f"✅ Transcription complete:")
        print(f"   Language: {info.language} ({info.language_probability:.1%} confidence)")
        print(f"   Duration: {info.duration:.1f}s")
        print(f"   Segments: {len(segments)}")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Returns information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            **self.MODEL_INFO.get(self.model_size, {})
        }

    @classmethod
    def list_models(cls) -> List[Dict[str, str]]:
        """Returns list of available Whisper models with their specs."""
        return [
            {"id": model, **info}
            for model, info in cls.MODEL_INFO.items()
        ]
