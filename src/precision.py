"""
Precision cutting module with audio validation.
Implements waveform analysis, VAD, and zero-crossing refinement.
"""

import json
import struct
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Optional imports with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


@dataclass
class WordTimestamp:
    """Word with precise timestamps."""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[int] = None
    # Refined timestamps after validation
    refined_start: Optional[float] = None
    refined_end: Optional[float] = None
    # Validation metrics
    speech_ratio: Optional[float] = None
    start_diff_ms: Optional[float] = None
    end_diff_ms: Optional[float] = None
    is_valid: bool = True


@dataclass
class CutSegment:
    """Segment to cut with validation data."""
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)
    # Refined after validation
    refined_start: Optional[float] = None
    refined_end: Optional[float] = None
    # Metrics
    avg_confidence: float = 1.0
    speech_coverage: float = 1.0
    boundary_quality: str = "unknown"


class TranscriptionParser:
    """
    Universal parser for word-level transcriptions.
    Supports: Deepgram, Whisper, AssemblyAI, Google, AWS.
    """
    
    @staticmethod
    def parse(json_path: Path, format_type: str = "auto") -> list[WordTimestamp]:
        """Parse transcription JSON to WordTimestamp list."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Auto-detect format
        if format_type == "auto":
            format_type = TranscriptionParser._detect_format(data)
        
        parser_map = {
            "deepgram": TranscriptionParser._parse_deepgram,
            "whisper": TranscriptionParser._parse_whisper,
            "assemblyai": TranscriptionParser._parse_assemblyai,
            "google": TranscriptionParser._parse_google,
        }
        
        parser = parser_map.get(format_type, TranscriptionParser._parse_deepgram)
        return parser(data)
    
    @staticmethod
    def _detect_format(data: dict) -> str:
        """Auto-detect transcription format."""
        # Deepgram format
        if "results" in data and "channels" in data.get("results", {}):
            return "deepgram"
        
        # Our processed Deepgram format
        if "words" in data and "utterances" in data:
            return "deepgram"
        
        # Whisper format
        if "segments" in data:
            first_seg = data["segments"][0] if data["segments"] else {}
            if "words" in first_seg:
                return "whisper"
        
        # AssemblyAI format (timestamps in milliseconds)
        if "words" in data:
            first_word = data["words"][0] if data["words"] else {}
            if isinstance(first_word.get("start"), int) and first_word.get("start", 0) > 100:
                return "assemblyai"
        
        # Google format
        if "results" in data and isinstance(data["results"], list):
            return "google"
        
        return "deepgram"
    
    @staticmethod
    def _parse_deepgram(data: dict) -> list[WordTimestamp]:
        """Parse Deepgram format."""
        words = []
        
        # Handle raw Deepgram response
        if "results" in data:
            channels = data["results"].get("channels", [{}])
            channel = channels[0] if channels else {}
            alternatives = channel.get("alternatives", [{}])
            alt = alternatives[0] if alternatives else {}
            word_list = alt.get("words", [])
        else:
            # Our processed format
            word_list = data.get("words", [])
        
        for w in word_list:
            words.append(WordTimestamp(
                word=w.get("punctuated_word", w.get("word", "")),
                start=float(w.get("start", 0)),
                end=float(w.get("end", 0)),
                confidence=float(w.get("confidence", 1.0)),
                speaker=w.get("speaker"),
            ))
        
        return words
    
    @staticmethod
    def _parse_whisper(data: dict) -> list[WordTimestamp]:
        """Parse Whisper/whisper-timestamped format."""
        words = []
        
        for segment in data.get("segments", []):
            for w in segment.get("words", []):
                words.append(WordTimestamp(
                    word=w.get("word", "").strip(),
                    start=float(w.get("start", 0)),
                    end=float(w.get("end", 0)),
                    confidence=float(w.get("probability", w.get("confidence", 1.0))),
                ))
        
        return words
    
    @staticmethod
    def _parse_assemblyai(data: dict) -> list[WordTimestamp]:
        """Parse AssemblyAI format (timestamps in ms)."""
        words = []
        
        for w in data.get("words", []):
            words.append(WordTimestamp(
                word=w.get("text", ""),
                start=w.get("start", 0) / 1000.0,  # ms -> s
                end=w.get("end", 0) / 1000.0,
                confidence=float(w.get("confidence", 1.0)),
                speaker=w.get("speaker"),
            ))
        
        return words
    
    @staticmethod
    def _parse_google(data: dict) -> list[WordTimestamp]:
        """Parse Google Speech-to-Text format."""
        words = []
        
        for result in data.get("results", []):
            for alt in result.get("alternatives", []):
                for w in alt.get("words", []):
                    # Google uses "Xs" format for time
                    start_str = w.get("startTime", "0s").rstrip("s")
                    end_str = w.get("endTime", "0s").rstrip("s")
                    
                    words.append(WordTimestamp(
                        word=w.get("word", ""),
                        start=float(start_str),
                        end=float(end_str),
                        confidence=float(alt.get("confidence", 1.0)),
                        speaker=w.get("speakerTag"),
                    ))
        
        return words


class AudioAnalyzer:
    """
    Audio analysis for timestamp validation.
    Uses librosa for waveform analysis.
    """
    
    def __init__(self, audio_path: Path, sample_rate: int = 22050):
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required: pip install librosa")
        
        self.audio_path = Path(audio_path)
        self.sr = sample_rate
        self.y, _ = librosa.load(str(audio_path), sr=sample_rate)
        self.duration = len(self.y) / self.sr
    
    def get_energy_envelope(
        self,
        start: float,
        end: float,
        frame_length: int = 512,
        hop_length: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate RMS energy envelope for segment."""
        start_sample = max(0, int(start * self.sr))
        end_sample = min(len(self.y), int(end * self.sr))
        
        segment = self.y[start_sample:end_sample]
        
        if len(segment) < frame_length:
            return np.array([start]), np.array([0.0])
        
        rms = librosa.feature.rms(
            y=segment,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sr,
            hop_length=hop_length
        ) + start
        
        return times, rms
    
    def find_zero_crossings(
        self,
        center: float,
        window_ms: float = 50
    ) -> np.ndarray:
        """Find zero-crossing points near a timestamp."""
        window_samples = int(window_ms / 1000 * self.sr)
        center_sample = int(center * self.sr)
        
        start = max(0, center_sample - window_samples)
        end = min(len(self.y), center_sample + window_samples)
        
        segment = self.y[start:end]
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        
        # Convert to time
        return (start + zero_crossings) / self.sr
    
    def find_speech_boundaries(
        self,
        start: float,
        end: float,
        threshold_db: float = -40,
        padding: float = 0.1
    ) -> tuple[float, float]:
        """Find actual speech boundaries using energy."""
        search_start = max(0, start - padding)
        search_end = min(self.duration, end + padding)
        
        times, rms = self.get_energy_envelope(search_start, search_end)
        
        if len(rms) == 0:
            return start, end
        
        # Convert threshold
        threshold_linear = librosa.db_to_amplitude(threshold_db)
        
        # Find active regions
        active = rms > threshold_linear
        
        if not np.any(active):
            return start, end
        
        active_indices = np.where(active)[0]
        actual_start = times[active_indices[0]]
        actual_end = times[active_indices[-1]]
        
        return actual_start, actual_end
    
    def get_optimal_cut_point(
        self,
        target_time: float,
        search_window_ms: float = 30,
        prefer_silence: bool = True
    ) -> float:
        """Find optimal cut point near target time."""
        window = search_window_ms / 1000
        
        # Get energy around target
        times, rms = self.get_energy_envelope(
            target_time - window,
            target_time + window
        )
        
        if len(rms) == 0:
            return target_time
        
        if prefer_silence:
            # Find lowest energy point
            min_idx = np.argmin(rms)
            optimal = times[min_idx]
        else:
            # Find zero crossing closest to target
            zc = self.find_zero_crossings(target_time, search_window_ms)
            if len(zc) > 0:
                distances = np.abs(zc - target_time)
                optimal = zc[np.argmin(distances)]
            else:
                optimal = target_time
        
        return optimal


class VADValidator:
    """
    Voice Activity Detection for timestamp validation.
    Uses WebRTC VAD.
    """
    
    def __init__(self, audio_path: Path, aggressiveness: int = 2):
        if not WEBRTCVAD_AVAILABLE:
            raise ImportError("webrtcvad required: pip install webrtcvad")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000  # WebRTC requires 8000, 16000, or 32000
        
        # Load and convert audio
        if LIBROSA_AVAILABLE:
            y, _ = librosa.load(str(audio_path), sr=self.sample_rate)
            self.audio_int16 = (y * 32767).astype(np.int16)
        else:
            # Fallback: use ffmpeg
            self.audio_int16 = self._load_with_ffmpeg(audio_path)
        
        self.duration = len(self.audio_int16) / self.sample_rate
    
    def _load_with_ffmpeg(self, audio_path: Path) -> np.ndarray:
        """Load audio using ffmpeg as fallback."""
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate), "-ac", "1",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True)
        return np.frombuffer(result.stdout, dtype=np.int16)
    
    def get_speech_ratio(
        self,
        start: float,
        end: float,
        frame_duration_ms: int = 30
    ) -> float:
        """Get ratio of frames containing speech."""
        frame_samples = int(self.sample_rate * frame_duration_ms / 1000)
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        
        speech_frames = 0
        total_frames = 0
        
        for i in range(start_sample, end_sample - frame_samples, frame_samples):
            frame = self.audio_int16[i:i + frame_samples]
            
            if len(frame) < frame_samples:
                continue
            
            frame_bytes = struct.pack(f'{len(frame)}h', *frame)
            
            try:
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    speech_frames += 1
                total_frames += 1
            except Exception:
                continue
        
        return speech_frames / max(total_frames, 1)
    
    def validate_words(
        self,
        words: list[WordTimestamp],
        min_speech_ratio: float = 0.3
    ) -> list[WordTimestamp]:
        """Validate word timestamps using VAD."""
        for word in words:
            word.speech_ratio = self.get_speech_ratio(word.start, word.end)
            word.is_valid = word.speech_ratio >= min_speech_ratio
        
        return words


class BoundaryRefiner:
    """
    Refines cut boundaries for clean audio transitions.
    """
    
    def __init__(self, audio_analyzer: Optional[AudioAnalyzer] = None):
        self.analyzer = audio_analyzer
    
    def refine_word(
        self,
        word: WordTimestamp,
        search_window_ms: float = 30,
        padding_ms: float = 10
    ) -> WordTimestamp:
        """Refine word boundaries using audio analysis."""
        if self.analyzer is None:
            word.refined_start = word.start
            word.refined_end = word.end
            return word
        
        padding = padding_ms / 1000
        
        # Find optimal start point
        optimal_start = self.analyzer.get_optimal_cut_point(
            word.start,
            search_window_ms,
            prefer_silence=True
        )
        
        # Find optimal end point
        optimal_end = self.analyzer.get_optimal_cut_point(
            word.end,
            search_window_ms,
            prefer_silence=True
        )
        
        # Apply padding
        word.refined_start = max(0, optimal_start - padding)
        word.refined_end = min(self.analyzer.duration, optimal_end + padding)
        
        # Calculate differences
        word.start_diff_ms = abs(word.start - word.refined_start) * 1000
        word.end_diff_ms = abs(word.end - word.refined_end) * 1000
        
        return word
    
    def refine_segment(
        self,
        segment: CutSegment,
        search_window_ms: float = 50,
        padding_ms: float = 100
    ) -> CutSegment:
        """Refine segment boundaries."""
        padding = padding_ms / 1000
        
        if self.analyzer is None:
            segment.refined_start = max(0, segment.start - padding)
            segment.refined_end = segment.end + padding
            segment.boundary_quality = "unvalidated"
            return segment
        
        # Find optimal boundaries
        optimal_start = self.analyzer.get_optimal_cut_point(
            segment.start,
            search_window_ms,
            prefer_silence=True
        )
        
        optimal_end = self.analyzer.get_optimal_cut_point(
            segment.end,
            search_window_ms,
            prefer_silence=True
        )
        
        segment.refined_start = max(0, optimal_start - padding)
        segment.refined_end = min(self.analyzer.duration, optimal_end + padding)
        
        # Assess boundary quality
        start_energy = self._get_boundary_energy(segment.refined_start)
        end_energy = self._get_boundary_energy(segment.refined_end)
        
        if start_energy < 0.01 and end_energy < 0.01:
            segment.boundary_quality = "excellent"
        elif start_energy < 0.05 and end_energy < 0.05:
            segment.boundary_quality = "good"
        else:
            segment.boundary_quality = "acceptable"
        
        return segment
    
    def _get_boundary_energy(self, time: float, window_ms: float = 10) -> float:
        """Get energy at a boundary point."""
        if self.analyzer is None:
            return 0.0
        
        window = window_ms / 1000
        _, rms = self.analyzer.get_energy_envelope(
            max(0, time - window),
            min(self.analyzer.duration, time + window)
        )
        
        return float(np.mean(rms)) if len(rms) > 0 else 0.0


class PrecisionCutter:
    """
    FFmpeg-based precision cutter with validation.
    """
    
    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        audio_path: Optional[Path] = None
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        self.video_info = self._get_video_info()
        self.duration = self.video_info.get("duration", 0)
        self.fps = self.video_info.get("fps", 30)
        
        # Initialize analyzers if audio available
        self.audio_analyzer = None
        self.vad_validator = None
        self.boundary_refiner = None
        
        if audio_path and Path(audio_path).exists():
            self._init_analyzers(audio_path)
    
    def _get_video_info(self) -> dict:
        """Get video metadata."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(self.video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"duration": 0, "fps": 30}
        
        data = json.loads(result.stdout)
        
        duration = float(data.get("format", {}).get("duration", 0))
        
        # Find video stream for fps
        fps = 30
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "30/1")
                try:
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = int(num) / int(den)
                    else:
                        fps = float(fps_str)
                except (ValueError, ZeroDivisionError):
                    fps = 30
                break
        
        return {"duration": duration, "fps": fps}
    
    def _init_analyzers(self, audio_path: Path):
        """Initialize audio analyzers."""
        try:
            if LIBROSA_AVAILABLE:
                self.audio_analyzer = AudioAnalyzer(audio_path)
                self.boundary_refiner = BoundaryRefiner(self.audio_analyzer)
        except Exception as e:
            print(f"Warning: Could not initialize audio analyzer: {e}")
        
        try:
            if WEBRTCVAD_AVAILABLE:
                self.vad_validator = VADValidator(audio_path)
        except Exception as e:
            print(f"Warning: Could not initialize VAD: {e}")
        
        if self.boundary_refiner is None:
            self.boundary_refiner = BoundaryRefiner(None)
    
    def create_segment_from_words(
        self,
        words: list[WordTimestamp],
        start_idx: int,
        end_idx: int
    ) -> CutSegment:
        """Create a cut segment from word range."""
        segment_words = words[start_idx:end_idx + 1]
        
        if not segment_words:
            raise ValueError("No words in range")
        
        segment = CutSegment(
            start=segment_words[0].start,
            end=segment_words[-1].end,
            words=segment_words,
            avg_confidence=sum(w.confidence for w in segment_words) / len(segment_words),
        )
        
        return segment
    
    def validate_segment(self, segment: CutSegment) -> CutSegment:
        """Validate and refine segment boundaries."""
        # VAD validation
        if self.vad_validator:
            segment.speech_coverage = self.vad_validator.get_speech_ratio(
                segment.start, segment.end
            )
        
        # Boundary refinement
        if self.boundary_refiner:
            segment = self.boundary_refiner.refine_segment(segment)
        else:
            segment.refined_start = segment.start
            segment.refined_end = segment.end
            segment.boundary_quality = "unvalidated"
        
        return segment
    
    def cut_segment(
        self,
        segment: CutSegment,
        output_name: str,
        method: str = "accurate",
        validate: bool = True,
        # Encoding options
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 18,
        preset: str = "medium",
        audio_bitrate: str = "192k",
    ) -> dict:
        """
        Cut a segment with precision.
        
        Methods:
        - "fast": Stream copy (may be imprecise)
        - "accurate": Re-encode for frame accuracy
        - "hybrid": Input seek + re-encode
        """
        # Validate if requested
        if validate:
            segment = self.validate_segment(segment)
        
        # Use refined boundaries if available
        start = segment.refined_start if segment.refined_start else segment.start
        end = segment.refined_end if segment.refined_end else segment.end
        duration = end - start
        
        output_path = self.output_dir / f"{output_name}.mp4"
        
        # Build FFmpeg command based on method
        if method == "fast":
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(self.video_path),
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ]
        
        elif method == "accurate":
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-ss", str(start),
                "-t", str(duration),
                "-c:v", video_codec,
                "-preset", preset,
                "-crf", str(crf),
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-async", "1",
                str(output_path)
            ]
        
        elif method == "hybrid":
            # Seek to nearest keyframe before, then precise cut
            keyframe_seek = max(0, start - 5)  # 5 seconds before
            offset = start - keyframe_seek
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(keyframe_seek),
                "-i", str(self.video_path),
                "-ss", str(offset),
                "-t", str(duration),
                "-c:v", video_codec,
                "-preset", "fast",
                "-crf", str(crf),
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-avoid_negative_ts", "make_zero",
                "-async", "1",
                str(output_path)
            ]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        success = result.returncode == 0 and output_path.exists()
        
        return {
            "success": success,
            "path": output_path if success else None,
            "segment": {
                "original_start": segment.start,
                "original_end": segment.end,
                "refined_start": start,
                "refined_end": end,
                "duration": duration,
                "boundary_quality": segment.boundary_quality,
                "speech_coverage": segment.speech_coverage,
            },
            "error": result.stderr if not success else None,
        }
    
    def cut_clips_from_analysis(
        self,
        analysis: dict,
        method: str = "hybrid",
        validate: bool = True,
        words: Optional[list[WordTimestamp]] = None,
    ) -> list[dict]:
        """Cut all clips from LLM analysis."""
        results = []
        clips = analysis.get("clips", [])
        
        for i, clip in enumerate(clips):
            # Create segment
            segment = CutSegment(
                start=clip["start_time"],
                end=clip["end_time"],
            )
            
            # If words provided, find matching words
            if words:
                segment.words = [
                    w for w in words
                    if w.start >= segment.start and w.end <= segment.end
                ]
                if segment.words:
                    segment.avg_confidence = sum(
                        w.confidence for w in segment.words
                    ) / len(segment.words)
            
            output_name = f"{self.video_path.stem}_clip{i+1:02d}"
            
            result = self.cut_segment(
                segment=segment,
                output_name=output_name,
                method=method,
                validate=validate,
            )
            
            result["clip_info"] = clip
            results.append(result)
        
        return results


class VideoWordCutter:
    """
    Complete system for precision video cutting.
    Integrates parsing, validation, refinement, and cutting.
    """
    
    def __init__(
        self,
        video_path: Path,
        transcription_path: Path,
        output_dir: Path,
        transcription_format: str = "auto"
    ):
        self.video_path = Path(video_path)
        self.transcription_path = Path(transcription_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse transcription
        self.words = TranscriptionParser.parse(
            transcription_path,
            transcription_format
        )
        
        # Extract audio for analysis
        self.audio_path = self.output_dir / f"{self.video_path.stem}_audio.wav"
        self._extract_audio()
        
        # Initialize cutter with analyzers
        self.cutter = PrecisionCutter(
            video_path=self.video_path,
            output_dir=self.output_dir,
            audio_path=self.audio_path
        )
    
    def _extract_audio(self):
        """Extract audio track for analysis."""
        if self.audio_path.exists():
            return
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            str(self.audio_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
    
    def validate_all_words(
        self,
        use_vad: bool = True,
        use_waveform: bool = True,
        min_speech_ratio: float = 0.3
    ) -> list[WordTimestamp]:
        """Validate and refine all word timestamps."""
        # VAD validation
        if use_vad and self.cutter.vad_validator:
            self.words = self.cutter.vad_validator.validate_words(
                self.words,
                min_speech_ratio
            )
        
        # Waveform refinement
        if use_waveform and self.cutter.boundary_refiner:
            for word in self.words:
                self.cutter.boundary_refiner.refine_word(word)
        
        return self.words
    
    def get_validation_report(self) -> dict:
        """Generate validation report for all words."""
        total = len(self.words)
        valid = sum(1 for w in self.words if w.is_valid)
        
        avg_start_diff = 0
        avg_end_diff = 0
        refined_count = 0
        
        for word in self.words:
            if word.start_diff_ms is not None:
                avg_start_diff += word.start_diff_ms
                avg_end_diff += word.end_diff_ms
                refined_count += 1
        
        if refined_count > 0:
            avg_start_diff /= refined_count
            avg_end_diff /= refined_count
        
        return {
            "total_words": total,
            "valid_words": valid,
            "validation_rate": valid / total * 100 if total > 0 else 0,
            "refined_words": refined_count,
            "avg_start_diff_ms": round(avg_start_diff, 2),
            "avg_end_diff_ms": round(avg_end_diff, 2),
            "invalid_words": [
                {"word": w.word, "start": w.start, "speech_ratio": w.speech_ratio}
                for w in self.words if not w.is_valid
            ][:10],  # First 10 invalid
        }
    
    def cut_phrase(
        self,
        start_word_idx: int,
        end_word_idx: int,
        output_name: str,
        method: str = "hybrid",
        validate: bool = True,
        padding_ms: float = 100
    ) -> dict:
        """Cut a phrase from word index range."""
        segment = self.cutter.create_segment_from_words(
            self.words,
            start_word_idx,
            end_word_idx
        )
        
        return self.cutter.cut_segment(
            segment=segment,
            output_name=output_name,
            method=method,
            validate=validate,
        )
    
    def cut_by_text(
        self,
        search_text: str,
        output_name: str,
        method: str = "hybrid",
        context_words: int = 0
    ) -> dict:
        """Cut segment containing specific text."""
        # Find matching words
        search_lower = search_text.lower()
        full_text = " ".join(w.word for w in self.words).lower()
        
        if search_lower not in full_text:
            return {"success": False, "error": "Text not found"}
        
        # Find word indices
        start_idx = None
        end_idx = None
        current_text = ""
        
        for i, word in enumerate(self.words):
            current_text += word.word.lower() + " "
            
            if search_lower in current_text and start_idx is None:
                # Backtrack to find start
                temp_text = ""
                for j in range(i, -1, -1):
                    temp_text = self.words[j].word.lower() + " " + temp_text
                    if search_lower in temp_text:
                        start_idx = j
                        break
            
            if start_idx is not None and end_idx is None:
                # Check if we've passed the search text
                remaining = current_text[current_text.find(search_lower):]
                if len(remaining.split()) > len(search_text.split()):
                    end_idx = i
                    break
        
        if end_idx is None:
            end_idx = len(self.words) - 1
        
        # Apply context
        start_idx = max(0, start_idx - context_words)
        end_idx = min(len(self.words) - 1, end_idx + context_words)
        
        return self.cut_phrase(start_idx, end_idx, output_name, method)


# Utility functions

def format_timestamp_ffmpeg(seconds: float) -> str:
    """Format seconds to FFmpeg timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def estimate_keyframe_position(time: float, fps: float = 30, gop: int = 250) -> float:
    """Estimate nearest keyframe position."""
    keyframe_interval = gop / fps
    return (time // keyframe_interval) * keyframe_interval
