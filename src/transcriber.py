"""
Transcriber module using Deepgram API.
Aligned with official documentation: https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded
"""

import httpx
import json
import hashlib
from pathlib import Path
from typing import Optional
from config import DEEPGRAM_API_KEY, TRANSCRIPTS_DIR
from deepgram_cache import DeepgramModelCache


class Transcriber:
    """Deepgram transcription with full API support."""
    
    BASE_URL = "https://api.deepgram.com/v1/listen"
    
    def __init__(
        self,
        model: str = "nova-2",
        language: str = "it",
        cache_enabled: bool = True
    ):
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY not configured")
        
        self.api_key = DEEPGRAM_API_KEY
        self.model = model
        self.language = language
        self.cache_enabled = cache_enabled
        self.cache_dir = TRANSCRIPTS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dynamic model cache
        self.model_cache = DeepgramModelCache(self.api_key)
        
        # Validate model compatibility
        if not self.model_cache.is_combination_supported(self.model, self.language):
             print(f"⚠️  WARNING: Model '{self.model}' + language '{self.language}' might not be supported.")
             fallback = "nova-2"
             print(f"   Falling back to '{fallback}' (General Purpose).")
             self.model = fallback
    
    def transcribe(
        self,
        audio_path: Optional[Path] = None,
        audio_url: Optional[str] = None,
        # Core parameters
        model: Optional[str] = None,
        language: Optional[str] = None,
        detect_language: bool = False,
        # Formatting
        punctuate: bool = True,
        paragraphs: bool = True,
        utterances: bool = True,
        utt_split: float = 0.8,
        smart_format: bool = True,
        # Features
        diarize: bool = True,
        filler_words: bool = False,
        profanity_filter: bool = False,
        numerals: bool = True,
        measurements: bool = True,
        # Analysis
        sentiment: bool = False,
        summarize: bool = False,
        topics: bool = False,
        # Advanced
        keywords: Optional[list[str]] = None,
        search: Optional[list[str]] = None,
        replace: Optional[list[str]] = None,
        redact: Optional[list[str]] = None,
        multichannel: bool = False,
        # Callback
        callback: Optional[str] = None,
        callback_method: str = "POST",
    ) -> dict:
        """
        Transcribe audio with full Deepgram API support.
        
        Args:
            audio_path: Local audio file path
            audio_url: Remote audio URL
            model: Deepgram model (nova-3, nova-2, enhanced, base, whisper-*)
            language: Language code (it, en, es, etc.) or 'multi' for multilingual
            detect_language: Auto-detect language
            punctuate: Add punctuation
            paragraphs: Group into paragraphs
            utterances: Group into utterances (speaker turns)
            utt_split: Utterance split threshold (0.0-1.0)
            smart_format: Format numbers, dates, etc.
            diarize: Speaker diarization
            filler_words: Include filler words (um, uh)
            profanity_filter: Filter profanity
            numerals: Convert numbers to digits
            measurements: Format measurements
            sentiment: Analyze sentiment
            summarize: Generate summary
            topics: Extract topics
            keywords: Boost specific keywords
            search: Search for specific terms
            replace: Replace terms (format: "term:replacement")
            redact: Redact sensitive info (pci, ssn, numbers)
            multichannel: Process channels separately
            callback: Webhook URL for async processing
            callback_method: Webhook method (POST/PUT)
        
        Returns:
            Structured transcription result
        """
        if not audio_path and not audio_url:
            raise ValueError("Provide either audio_path or audio_url")
        
        # Check cache
        cache_key = self._get_cache_key(audio_path, audio_url, model or self.model)
        if self.cache_enabled:
            cached = self._load_cache(cache_key)
            if cached:
                return cached
        
        # Build query parameters
        params = {
            "model": model or self.model,
            "punctuate": str(punctuate).lower(),
            "paragraphs": str(paragraphs).lower(),
            "utterances": str(utterances).lower(),
            "utt_split": str(utt_split),
            "smart_format": str(smart_format).lower(),
            "diarize": str(diarize).lower(),
            "filler_words": str(filler_words).lower(),
            "profanity_filter": str(profanity_filter).lower(),
            "numerals": str(numerals).lower(),
            "measurements": str(measurements).lower(),
            "multichannel": str(multichannel).lower(),
        }
        
        # Language handling
        if detect_language:
            params["detect_language"] = "true"
        else:
            params["language"] = language or self.language
        
        # Analysis features
        if sentiment:
            params["sentiment"] = "true"
        if summarize:
            params["summarize"] = "v2"
        if topics:
            params["topics"] = "true"
        
        # Keywords/search
        if keywords:
            params["keywords"] = keywords
        if search:
            params["search"] = search
        if replace:
            params["replace"] = replace
        if redact:
            params["redact"] = redact
        
        # Callback
        if callback:
            params["callback"] = callback
            params["callback_method"] = callback_method
        
        # Make request
        headers = {
            "Authorization": f"Token {self.api_key}",
        }
        
        with httpx.Client(timeout=300.0) as client:
            if audio_url:
                # URL-based transcription
                headers["Content-Type"] = "application/json"
                response = client.post(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    json={"url": audio_url}
                )
            else:
                # File-based transcription
                audio_path = Path(audio_path)
                content_type = self._get_content_type(audio_path)
                headers["Content-Type"] = content_type
                
                with open(audio_path, "rb") as f:
                    response = client.post(
                        self.BASE_URL,
                        params=params,
                        headers=headers,
                        content=f.read()
                    )
        
        response.raise_for_status()
        raw_result = response.json()
        
        # Process response
        result = self._process_response(raw_result)
        
        # Cache result
        if self.cache_enabled:
            self._save_cache(cache_key, result)
        
        return result
    
    def _process_response(self, raw: dict) -> dict:
        """Process Deepgram response into structured format."""
        results = raw.get("results", {})
        channels = results.get("channels", [{}])
        channel = channels[0] if channels else {}
        alternatives = channel.get("alternatives", [{}])
        alternative = alternatives[0] if alternatives else {}
        
        # Extract words with full metadata
        words = []
        for w in alternative.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": w.get("start", 0),
                "end": w.get("end", 0),
                "confidence": w.get("confidence", 0),
                "speaker": w.get("speaker"),
                "punctuated_word": w.get("punctuated_word", w.get("word", "")),
            })
        
        # Extract utterances
        utterances = []
        for u in results.get("utterances", []):
            utterances.append({
                "text": u.get("transcript", ""),
                "start": u.get("start", 0),
                "end": u.get("end", 0),
                "speaker": u.get("speaker"),
                "confidence": u.get("confidence", 0),
                "words": u.get("words", []),
            })
        
        # Fallback: create utterances from words if not provided
        if not utterances and words:
            utterances = self._create_utterances_from_words(words)
        
        # Extract paragraphs
        paragraphs = []
        for p in alternative.get("paragraphs", {}).get("paragraphs", []):
            paragraphs.append({
                "text": " ".join(s.get("text", "") for s in p.get("sentences", [])),
                "start": p.get("start", 0),
                "end": p.get("end", 0),
                "speaker": p.get("speaker"),
                "sentences": p.get("sentences", []),
            })
        
        # Metadata
        metadata = raw.get("metadata", {})
        
        # Build result
        result = {
            "full_text": alternative.get("transcript", ""),
            "words": words,
            "utterances": utterances,
            "paragraphs": paragraphs,
            "duration": metadata.get("duration", 0),
            "channels": metadata.get("channels", 1),
            "language": results.get("detected_language") or self.language,
            "confidence": alternative.get("confidence", 0),
            "metadata": {
                "request_id": metadata.get("request_id"),
                "model_info": metadata.get("model_info", {}),
                "created": metadata.get("created"),
                "sha256": metadata.get("sha256"),
            },
        }
        
        # Optional analysis results
        if "summaries" in results:
            result["summary"] = results["summaries"]
        if "sentiments" in results:
            result["sentiments"] = results["sentiments"]
        if "topics" in results:
            result["topics"] = results["topics"]
        
        return result
    
    def _create_utterances_from_words(
        self,
        words: list[dict],
        pause_threshold: float = 0.8
    ) -> list[dict]:
        """Create utterances from words based on pauses and speaker changes."""
        if not words:
            return []
        
        utterances = []
        current_utterance = {
            "words": [words[0]],
            "start": words[0]["start"],
            "speaker": words[0].get("speaker"),
        }
        
        for i, word in enumerate(words[1:], 1):
            prev_word = words[i - 1]
            pause = word["start"] - prev_word["end"]
            speaker_change = word.get("speaker") != prev_word.get("speaker")
            
            if pause > pause_threshold or speaker_change:
                # Complete current utterance
                current_utterance["end"] = prev_word["end"]
                current_utterance["text"] = " ".join(
                    w.get("punctuated_word", w["word"]) 
                    for w in current_utterance["words"]
                )
                current_utterance["confidence"] = sum(
                    w["confidence"] for w in current_utterance["words"]
                ) / len(current_utterance["words"])
                utterances.append(current_utterance)
                
                # Start new utterance
                current_utterance = {
                    "words": [word],
                    "start": word["start"],
                    "speaker": word.get("speaker"),
                }
            else:
                current_utterance["words"].append(word)
        
        # Complete final utterance
        if current_utterance["words"]:
            current_utterance["end"] = current_utterance["words"][-1]["end"]
            current_utterance["text"] = " ".join(
                w.get("punctuated_word", w["word"]) 
                for w in current_utterance["words"]
            )
            current_utterance["confidence"] = sum(
                w["confidence"] for w in current_utterance["words"]
            ) / len(current_utterance["words"])
            utterances.append(current_utterance)
        
        return utterances
    
    def _get_content_type(self, path: Path) -> str:
        """Get MIME type for audio file."""
        ext = path.suffix.lower()
        types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".mp4": "audio/mp4",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return types.get(ext, "audio/wav")
    
    def _get_cache_key(
        self,
        audio_path: Optional[Path],
        audio_url: Optional[str],
        model: str
    ) -> str:
        """Generate cache key."""
        if audio_path:
            source = str(Path(audio_path).resolve())
        else:
            source = audio_url
        
        key_data = f"{source}:{model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[dict]:
        """Load cached transcription."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def _save_cache(self, cache_key: str, data: dict) -> None:
        """Save transcription to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def list_models(self) -> dict:
        """List available Deepgram models from dynamic cache."""
        data = self.model_cache.get_models()
        # Convert list of dicts to simple dict for compatibility if needed, 
        # but better to return the full list or let main.py handle it.
        # For backward compatibility with existing main.py logic (if not updated yet):
        return {m["name"]: m.get("description", "") for m in data.get("models", [])}
    
    def set_model(self, model: str) -> None:
        """Set default model."""
        self.model = model
    
    def set_language(self, language: str) -> None:
        """Set default language."""
        self.language = language


# Utility functions

def generate_srt(transcription: dict, max_chars: int = 42) -> str:
    """Generate SRT subtitle file from transcription."""
    lines = []
    index = 1
    
    for utterance in transcription.get("utterances", []):
        text = utterance["text"]
        start = utterance["start"]
        end = utterance["end"]
        
        # Split long utterances
        if len(text) > max_chars:
            words = utterance.get("words", [])
            if words:
                chunks = _split_utterance_words(words, max_chars)
                for chunk_start, chunk_end, chunk_text in chunks:
                    lines.append(f"{index}")
                    lines.append(f"{_format_srt_time(chunk_start)} --> {_format_srt_time(chunk_end)}")
                    lines.append(chunk_text)
                    lines.append("")
                    index += 1
            else:
                lines.append(f"{index}")
                lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
                lines.append(text)
                lines.append("")
                index += 1
        else:
            lines.append(f"{index}")
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(text)
            lines.append("")
            index += 1
    
    return "\n".join(lines)


def _split_utterance_words(
    words: list[dict],
    max_chars: int
) -> list[tuple[float, float, str]]:
    """Split utterance words into chunks."""
    chunks = []
    current_words = []
    current_text = ""
    
    for word in words:
        word_text = word.get("punctuated_word", word["word"])
        test_text = f"{current_text} {word_text}".strip()
        
        if len(test_text) > max_chars and current_words:
            chunks.append((
                current_words[0]["start"],
                current_words[-1]["end"],
                current_text
            ))
            current_words = [word]
            current_text = word_text
        else:
            current_words.append(word)
            current_text = test_text
    
    if current_words:
        chunks.append((
            current_words[0]["start"],
            current_words[-1]["end"],
            current_text
        ))
    
    return chunks


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_word_level_srt(transcription: dict, words_per_chunk: int = 3) -> str:
    """Generate word-level SRT for karaoke-style subtitles."""
    words = transcription.get("words", [])
    lines = []
    index = 1
    
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i:i + words_per_chunk]
        if not chunk:
            continue
        
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        text = " ".join(w.get("punctuated_word", w["word"]) for w in chunk)
        
        lines.append(f"{index}")
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")
        index += 1
    
    return "\n".join(lines)

def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
    if seconds is None:
        return "00:00"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return f"{int(m):02d}:{int(s):02d}"

def format_transcript_for_analysis(transcript: dict) -> str:
    """Formats transcript for LLM analysis."""
    lines = []
    utterances = transcript.get("utterances", [])
    
    for u in utterances:
        start = format_timestamp(u.get("start", 0))
        end = format_timestamp(u.get("end", 0))
        speaker = f"[Speaker {u.get('speaker', '?')}]"
        text = u.get("text", "")
        lines.append(f"[{start}-{end}] {speaker}: {text}")
        
    return "\n".join(lines)
