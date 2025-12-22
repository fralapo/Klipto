"""
Enhanced Multi-Agent Video Analyzer v3.1
Fully integrated TOON format for token-efficient LLM communication.

Features:
- Local scene detection from transcript structure
- TOON-based prompts (70%+ token savings)
- TOON-based LLM responses
- Automatic intro/outro exclusion
- Sentence-level boundary alignment
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from openai import OpenAI

# Local imports
from config import (
    OPENROUTER_API_KEY,
    ANALYSIS_DIR,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
)
from transcriber import format_timestamp
import toon


# ═══════════════════════════════════════════════════════════════════════════
# TOON UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class TranscriptToToon:
    """
    Converts Deepgram transcript structures to TOON format.
    Optimized for LLM context efficiency.
    """
    
    @staticmethod
    def utterances(
        transcript: Union[dict, Path, str],
        max_text_length: int = 200,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """Convert utterances to TOON tabular format."""
        if isinstance(transcript, (str, Path)):
            transcript = json.loads(Path(transcript).read_text(encoding="utf-8"))
        
        utterances_list = transcript.get("utterances", [])
        
        data = []
        for i, u in enumerate(utterances_list):
            u_start = u.get("start", 0)
            u_end = u.get("end", 0)
            
            if start_time is not None and u_end < start_time:
                continue
            if end_time is not None and u_start > end_time:
                continue
            
            text = u.get("text", "")
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            data.append({
                "idx": i,
                "start": round(u_start, 2),
                "end": round(u_end, 2),
                "spk": u.get("speaker", 0),
                "text": text,
            })
        
        if not data:
            return "utterances[0]:"
        
        # Use tab delimiter for better tokenization
        return toon.encode({"utterances": data}, delimiter="\t")
    
    @staticmethod
    def words(
        transcript: Union[dict, Path, str],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_words: int = 500,
    ) -> str:
        """Convert words to TOON tabular format."""
        if isinstance(transcript, (str, Path)):
            transcript = json.loads(Path(transcript).read_text(encoding="utf-8"))
        
        words_list = transcript.get("words", [])
        
        data = []
        for i, w in enumerate(words_list):
            if len(data) >= max_words:
                break
                
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            
            if start_time is not None and w_end < start_time:
                continue
            if end_time is not None and w_start > end_time:
                continue
            
            data.append({
                "idx": i,
                "w": w.get("punctuated_word", w.get("word", "")),
                "s": round(w_start, 2),
                "e": round(w_end, 2),
            })
        
        if not data:
            return "words[0]:"
        
        return toon.encode({"words": data}, delimiter="\t")
    
    @staticmethod
    def sentences(transcript: Union[dict, Path, str]) -> str:
        """Extract sentences from paragraphs and convert to TOON."""
        if isinstance(transcript, (str, Path)):
            transcript = json.loads(Path(transcript).read_text(encoding="utf-8"))
        
        paragraphs = transcript.get("paragraphs", [])
        
        data = []
        idx = 0
        for para in paragraphs:
            speaker = para.get("speaker", 0)
            for sent in para.get("sentences", []):
                data.append({
                    "idx": idx,
                    "start": round(sent.get("start", 0), 2),
                    "end": round(sent.get("end", 0), 2),
                    "spk": speaker,
                    "text": sent.get("text", ""),
                })
                idx += 1
        
        if not data:
            return "sentences[0]:"
        
        return toon.encode({"sentences": data}, delimiter="\t")
    
    @staticmethod
    def scenes(scenes: List[Dict[str, Any]]) -> str:
        """Convert detected scenes to TOON tabular format."""
        data = []
        for scene in scenes:
            speakers = scene.get("speakers", [])
            if isinstance(speakers, list):
                speakers_str = ",".join(map(str, speakers))
            else:
                speakers_str = str(speakers)
            
            data.append({
                "id": scene.get("id", 0),
                "start": round(scene.get("start_time", 0), 2),
                "end": round(scene.get("end_time", 0), 2),
                "dur": round(scene.get("end_time", 0) - scene.get("start_time", 0), 1),
                "spks": speakers_str,
                "type": scene.get("scene_type", "unknown"),
            })
        
        if not data:
            return "scenes[0]:"
        
        return toon.encode({"scenes": data}, delimiter="\t")
    
    @staticmethod
    def scenes_with_text(
        scenes: List[Dict[str, Any]],
        utterances: List[dict],
        max_text_length: int = 300,
    ) -> str:
        """Convert scenes with transcript text to TOON."""
        data = []
        for scene in scenes:
            # Get utterances for this scene
            start_idx = scene.get("start_utterance_idx", 0)
            end_idx = scene.get("end_utterance_idx", 0)
            scene_utts = utterances[start_idx:end_idx + 1]
            text = " ".join(u.get("text", "") for u in scene_utts)
            
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            data.append({
                "id": scene.get("id", 0),
                "start": round(scene.get("start_time", 0), 2),
                "end": round(scene.get("end_time", 0), 2),
                "type": scene.get("scene_type", "")[:15],
                "title": scene.get("title", "")[:30],
                "viral": scene.get("viral_potential", 5),
                "text": text,
            })
        
        if not data:
            return "scenes[0]:"
        
        return toon.encode({"scenes": data}, delimiter="\t")
    
    @staticmethod
    def clips(clips: List[Dict[str, Any]]) -> str:
        """Convert clip proposals to TOON tabular format."""
        data = []
        for clip in clips:
            data.append({
                "id": clip.get("id", 0),
                "scene": clip.get("scene_id", 0),
                "start": round(clip.get("start_time", 0), 2),
                "end": round(clip.get("end_time", 0), 2),
                "score": clip.get("virality_score", 0),
                "hook": clip.get("hook_text", "")[:50],
            })
        
        if not data:
            return "clips[0]:"
        
        return toon.encode({"clips": data}, delimiter="\t")
    
    @staticmethod
    def clips_for_review(
        clips: List[Dict[str, Any]],
        sentences: List[dict],
    ) -> str:
        """Convert clips with context for supervisor review."""
        data = []
        for clip in clips:
            # Find first and last sentences
            clip_start = clip.get("start_time", 0)
            clip_end = clip.get("end_time", 0)
            
            first_sent = ""
            last_sent = ""
            
            for sent in sentences:
                if sent["start"] >= clip_start - 0.5 and sent["end"] <= clip_end + 0.5:
                    if not first_sent:
                        first_sent = sent["text"][:40]
                    last_sent = sent["text"][:40]
            
            data.append({
                "id": clip.get("id", 0),
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "dur": round(clip_end - clip_start, 1),
                "score": clip.get("virality_score", 0),
                "first": first_sent,
                "last": last_sent,
            })
        
        if not data:
            return "clips[0]:"
        
        return toon.encode({"clips": data}, delimiter="\t")


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedScene:
    """Scene detected from transcript structure."""
    id: int
    start_time: float
    end_time: float
    start_utterance_idx: int
    end_utterance_idx: int
    speakers: List[int]
    text_preview: str
    # Classification (filled by LLM)
    scene_type: str = "unknown"
    title: str = ""
    description: str = ""
    mood: str = ""
    viral_potential: int = 0
    is_excluded: bool = False
    exclusion_reason: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_utterance_idx": self.start_utterance_idx,
            "end_utterance_idx": self.end_utterance_idx,
            "speakers": self.speakers,
            "scene_type": self.scene_type,
            "title": self.title,
            "description": self.description,
            "mood": self.mood,
            "viral_potential": self.viral_potential,
            "is_excluded": self.is_excluded,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class ClipCandidate:
    """A potential clip identified by the selector."""
    id: int
    scene_id: int
    start_time: float
    end_time: float
    hook_text: str
    title: str
    why_viral: str
    virality_score: int
    # Validation
    issues: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class FinalClip:
    """Validated and refined final clip."""
    clip_number: int
    start_time: float
    end_time: float
    duration: float
    first_word_idx: int
    last_word_idx: int
    first_words: str
    last_words: str
    scene_id: int
    title: str
    hook_text: str
    virality_score: int
    why_viral: str
    
    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# SCENE DETECTOR (LOCAL - NO LLM)
# ═══════════════════════════════════════════════════════════════════════════

class SceneDetector:
    """
    Detects scene boundaries from transcript structure.
    Uses speaker changes, pauses, and pattern matching.
    """
    
    INTRO_PATTERNS = [
        r"ciao\s+ragazzi",
        r"benvenuti",
        r"come\s+promesso",
        r"oggi\s+siamo",
        r"in\s+questo\s+video",
        r"eccoci\s+qui",
        r"buongiorno",
        r"bentornati",
    ]
    
    OUTRO_PATTERNS = [
        r"spolliciate",
        r"mettete\s+(la\s+)?campanell[ae]",
        r"iscriviti",
        r"seguitemi",
        r"seguimi",
        r"canale\s+youtube",
        r"live\s+su\s+twitch",
        r"finisce\s+(la\s+mia\s+)?avventura",
        r"ci\s+vediamo",
        r"alla\s+prossima",
        r"lasciate\s+un\s+like",
        r"commentate",
    ]
    
    SHOUTOUT_PATTERNS = [
        r"seguitelo",
        r"seguitela",
        r"direttamente\s+da",
        r"top\s+streamer",
        r"ragazzi.*eccolo",
        r"ringraziamo",
        r"un\s+applauso",
    ]
    
    def __init__(
        self,
        min_scene_duration: float = 8.0,
        max_pause_within_scene: float = 2.0,
    ):
        self.min_scene_duration = min_scene_duration
        self.max_pause_within_scene = max_pause_within_scene
    
    def detect_scenes(self, transcript: dict) -> List[DetectedScene]:
        """Detect scene boundaries from transcript."""
        utterances = transcript.get("utterances", [])
        
        if not utterances:
            return []
        
        # Find break points
        break_points = self._find_break_points(utterances)
        
        # Create scenes
        scenes = self._create_scenes(utterances, break_points)
        
        # Merge tiny scenes
        scenes = self._merge_tiny_scenes(scenes, utterances)
        
        # Pre-classify (intro/outro detection)
        scenes = self._preclassify_scenes(scenes, utterances)
        
        return scenes
    
    def _find_break_points(self, utterances: List[dict]) -> List[int]:
        """Find indices where scenes should break."""
        break_points = [0]
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            pause = curr["start"] - prev["end"]
            
            # Long pause = likely new scene
            if pause > self.max_pause_within_scene:
                break_points.append(i)
                continue
            
            # Medium pause + speaker change
            prev_speaker = prev.get("speaker", -1)
            curr_speaker = curr.get("speaker", -1)
            
            if pause > 1.0 and prev_speaker != curr_speaker:
                if self._is_sustained_speaker_change(utterances, i, curr_speaker):
                    break_points.append(i)
        
        return break_points
    
    def _is_sustained_speaker_change(
        self,
        utterances: List[dict],
        start_idx: int,
        new_speaker: int,
        lookahead: int = 3
    ) -> bool:
        """Check if speaker change is sustained."""
        count = 0
        for i in range(start_idx, min(start_idx + lookahead, len(utterances))):
            if utterances[i].get("speaker") == new_speaker:
                count += 1
        return count >= 2
    
    def _create_scenes(
        self,
        utterances: List[dict],
        break_points: List[int]
    ) -> List[DetectedScene]:
        """Create scene objects from break points."""
        scenes = []
        
        for i, start_idx in enumerate(break_points):
            end_idx = break_points[i + 1] - 1 if i + 1 < len(break_points) else len(utterances) - 1
            
            start_time = utterances[start_idx]["start"]
            end_time = utterances[end_idx]["end"]
            
            speakers = list(set(
                u.get("speaker", 0)
                for u in utterances[start_idx:end_idx + 1]
            ))
            
            text_parts = [u["text"] for u in utterances[start_idx:end_idx + 1]]
            full_text = " ".join(text_parts)
            preview = full_text[:150] + "..." if len(full_text) > 150 else full_text
            
            scenes.append(DetectedScene(
                id=len(scenes) + 1,
                start_time=start_time,
                end_time=end_time,
                start_utterance_idx=start_idx,
                end_utterance_idx=end_idx,
                speakers=speakers,
                text_preview=preview,
            ))
        
        return scenes
    
    def _merge_tiny_scenes(
        self,
        scenes: List[DetectedScene],
        utterances: List[dict]
    ) -> List[DetectedScene]:
        """Merge scenes that are too short."""
        if len(scenes) <= 1:
            return scenes
        
        merged = []
        current = scenes[0]
        
        for next_scene in scenes[1:]:
            if current.duration < self.min_scene_duration:
                # Merge with next
                current = DetectedScene(
                    id=current.id,
                    start_time=current.start_time,
                    end_time=next_scene.end_time,
                    start_utterance_idx=current.start_utterance_idx,
                    end_utterance_idx=next_scene.end_utterance_idx,
                    speakers=list(set(current.speakers + next_scene.speakers)),
                    text_preview=current.text_preview,
                )
            else:
                merged.append(current)
                current = next_scene
        
        merged.append(current)
        
        # Re-number
        for i, scene in enumerate(merged):
            scene.id = i + 1
        
        return merged
    
    def _preclassify_scenes(
        self,
        scenes: List[DetectedScene],
        utterances: List[dict]
    ) -> List[DetectedScene]:
        """Pre-classify scenes as intro/outro/shoutout."""
        total_duration = utterances[-1]["end"] if utterances else 0
        
        for scene in scenes:
            scene_utts = utterances[scene.start_utterance_idx:scene.end_utterance_idx + 1]
            full_text = " ".join(u["text"] for u in scene_utts).lower()
            
            # Check intro (first 15% of video)
            if scene.start_time < total_duration * 0.15:
                for pattern in self.INTRO_PATTERNS:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        scene.scene_type = "intro"
                        scene.is_excluded = True
                        scene.exclusion_reason = "Detected as intro"
                        break
            
            # Check outro (last 15% of video)
            if scene.end_time > total_duration * 0.85:
                for pattern in self.OUTRO_PATTERNS:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        scene.scene_type = "outro"
                        scene.is_excluded = True
                        scene.exclusion_reason = "Detected as outro/CTA"
                        break
            
            # Check shoutout (don't exclude, just flag)
            if not scene.is_excluded:
                shoutout_count = sum(
                    1 for p in self.SHOUTOUT_PATTERNS
                    if re.search(p, full_text, re.IGNORECASE)
                )
                if shoutout_count >= 2:
                    scene.scene_type = "shoutout"
        
        return scenes


# ═══════════════════════════════════════════════════════════════════════════
# SENTENCE BOUNDARY EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class SentenceBoundaryExtractor:
    """Extracts sentence boundaries for precise clip cutting."""
    
    def __init__(self, transcript: dict):
        self.words = transcript.get("words", [])
        self.paragraphs = transcript.get("paragraphs", [])
        self.sentences = self._extract_sentences()
    
    def _extract_sentences(self) -> List[dict]:
        """Extract all sentences with boundaries."""
        sentences = []
        for para in self.paragraphs:
            for sent in para.get("sentences", []):
                sentences.append({
                    "text": sent["text"],
                    "start": sent["start"],
                    "end": sent["end"],
                    "speaker": para.get("speaker", 0),
                })
        return sentences
    
    def get_sentences_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[dict]:
        """Get sentences within a time range."""
        return [
            {"index": i, **sent}
            for i, sent in enumerate(self.sentences)
            if sent["start"] >= start_time - 0.5 and sent["end"] <= end_time + 0.5
        ]
    
    def find_clean_boundaries(
        self,
        target_start: float,
        target_end: float,
        max_drift: float = 3.0
    ) -> Tuple[float, float, int, int]:
        """Find clean sentence boundaries near target times."""
        # Find start sentence
        start_sent = None
        for i, sent in enumerate(self.sentences):
            if abs(sent["start"] - target_start) <= max_drift:
                start_sent = {"index": i, **sent}
                break
            if sent["start"] > target_start + max_drift:
                if i > 0:
                    start_sent = {"index": i - 1, **self.sentences[i - 1]}
                break
        
        if not start_sent:
            # Fallback
            for i, sent in enumerate(self.sentences):
                if sent["start"] <= target_start <= sent["end"]:
                    start_sent = {"index": i, **sent}
                    break
        
        # Find end sentence
        end_sent = None
        for i in range(len(self.sentences) - 1, -1, -1):
            sent = self.sentences[i]
            if abs(sent["end"] - target_end) <= max_drift:
                end_sent = {"index": i, **sent}
                break
            if sent["end"] < target_end - max_drift:
                if i < len(self.sentences) - 1:
                    end_sent = {"index": i + 1, **self.sentences[i + 1]}
                break
        
        if not end_sent:
            for i, sent in enumerate(self.sentences):
                if sent["start"] <= target_end <= sent["end"]:
                    end_sent = {"index": i, **sent}
                    break
        
        if not start_sent or not end_sent:
            return target_start, target_end, -1, -1
        
        return (
            start_sent["start"],
            end_sent["end"],
            start_sent["index"],
            end_sent["index"]
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOON RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════

class ToonResponseParser:
    """
    Parses TOON-formatted LLM responses with JSON fallback.
    """
    
    @staticmethod
    def parse(response: str, strict: bool = False) -> List[dict]:
        """
        Parse LLM response, trying TOON first then JSON.
        
        Returns:
            List of dicts parsed from response
        """
        response = response.strip()
        
        # Try to extract from markdown code block
        if "```toon" in response:
            match = re.search(r'```toon\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1).strip()
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1).strip()
        
        # Try TOON decode
        try:
            result = toon.decode(response, strict=False)
            if isinstance(result, dict):
                # Find the array in the result
                for key, value in result.items():
                    if isinstance(value, list):
                        return value
                return [result]
            elif isinstance(result, list):
                return result
        except Exception:
            pass
        
        # Try JSON fallback
        try:
            import json
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                for key in ["clips", "scenes", "approved", "classifications", "items"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        except Exception:
            pass
        
        if strict:
            raise ValueError(f"Could not parse response: {response[:200]}...")
        
        return []
    
    @staticmethod
    def parse_supervision_response(response: str) -> Tuple[List[dict], List[dict]]:
        """
        Parse supervision response with approved/rejected sections.
        
        Returns:
            Tuple of (approved_list, rejected_list)
        """
        response = response.strip()
        
        # Clean markdown
        if "```" in response:
            match = re.search(r'```(?:toon)?\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1).strip()
        
        approved = []
        rejected = []
        
        # Try to parse as TOON with two sections
        try:
            # Split by approved/rejected headers
            parts = re.split(r'\n(?=approved\[|rejected\[)', response)
            
            for part in parts:
                part = part.strip()
                if part.startswith("approved["):
                    result = toon.decode(part, strict=False)
                    if isinstance(result, dict):
                        for v in result.values():
                            if isinstance(v, list):
                                approved = v
                                break
                    elif isinstance(result, list):
                        approved = result
                        
                elif part.startswith("rejected["):
                    result = toon.decode(part, strict=False)
                    if isinstance(result, dict):
                        for v in result.values():
                            if isinstance(v, list):
                                rejected = v
                                break
                    elif isinstance(result, list):
                        rejected = result
            
            if approved or rejected:
                return approved, rejected
        except Exception:
            pass
        
        # Try JSON fallback
        try:
            import json
            data = json.loads(response)
            approved = data.get("approved", [])
            rejected = data.get("rejected", [])
            return approved, rejected
        except Exception:
            pass
        
        return [], []


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED MULTI-AGENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedMultiAgentAnalyzer:
    """
    Multi-agent analyzer with TOON integration for token efficiency.
    
    Pipeline:
    1. LOCAL: Detect scenes from transcript structure
    2. LLM (TOON): Classify and score scenes
    3. LLM (TOON): Select best clips
    4. LLM (TOON): Supervise and validate
    5. LOCAL: Align to word boundaries
    """
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
    
    DEFAULT_MODELS = {
        "classifier": "deepseek/deepseek-chat",
        "selector": "deepseek/deepseek-chat",
        "supervisor": "deepseek/deepseek-chat",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[Dict[str, str]] = None,
        app_name: str = "Klipto",
        app_url: str = "https://github.com/klipto",
        debug: bool = False,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.models = {**self.DEFAULT_MODELS, **(models or {})}
        self.app_name = app_name
        self.app_url = app_url
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        self.client = OpenAI(
            base_url=self.OPENROUTER_API_URL,
            api_key=self.api_key,
        )
        
        # Cost tracking
        self.costs = {"classifier": 0, "selector": 0, "supervisor": 0, "total": 0}
        self.tokens = {"classifier": 0, "selector": 0, "supervisor": 0, "total": 0}
    
    def analyze(
        self,
        transcript: dict,
        video_title: str,
        max_clips: int = 5,
        language: str = "it",
        target_platform: str = "all",
    ) -> dict:
        """Run complete analysis pipeline."""
        
        print(f"\n{'═'*70}")
        print("🤖 ENHANCED MULTI-AGENT ANALYZER v3.1 (TOON)")
        print(f"{'═'*70}")
        print(f"   Video: {video_title}")
        print(f"   Duration: {format_timestamp(transcript.get('duration', 0))}")
        print(f"   Target: {max_clips} clips for {target_platform}")
        
        duration = transcript.get("duration", 0)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Local Scene Detection
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("📍 PHASE 1: Scene Detection (Local)")
        print(f"{'─'*70}")
        
        detector = SceneDetector()
        scenes = detector.detect_scenes(transcript)
        
        print(f"   Detected {len(scenes)} scenes:")
        for scene in scenes:
            status = "❌ EXCLUDED" if scene.is_excluded else "✓"
            print(f"   [{scene.id}] {format_timestamp(scene.start_time)}-"
                  f"{format_timestamp(scene.end_time)} ({scene.duration:.1f}s) "
                  f"spk:{scene.speakers} {status}")
        
        viable_scenes = [s for s in scenes if not s.is_excluded]
        excluded_count = len(scenes) - len(viable_scenes)
        print(f"\n   Viable: {len(viable_scenes)} | Excluded: {excluded_count}")
        
        if not viable_scenes:
            return self._build_empty_result(video_title, duration, target_platform, language)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Scene Classification (LLM + TOON)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("📊 PHASE 2: Scene Classification (LLM + TOON)")
        print(f"{'─'*70}")
        
        classified_scenes = self._classify_scenes(
            scenes=viable_scenes,
            transcript=transcript,
            video_title=video_title,
            language=language,
        )
        
        # Sort by viral potential
        classified_scenes.sort(key=lambda s: s.viral_potential, reverse=True)
        
        print(f"   Classified {len(classified_scenes)} scenes:")
        for scene in classified_scenes[:10]:  # Show top 10
            print(f"   [{scene.id}] {scene.scene_type}: {scene.title[:35]}... "
                  f"(viral: {scene.viral_potential}/10)")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: Clip Selection (LLM + TOON)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("🎬 PHASE 3: Clip Selection (LLM + TOON)")
        print(f"{'─'*70}")
        
        sentence_extractor = SentenceBoundaryExtractor(transcript)
        
        clip_candidates = self._select_clips(
            scenes=classified_scenes,
            transcript=transcript,
            sentence_extractor=sentence_extractor,
            max_clips=max_clips + 3,  # Extra for filtering
            language=language,
            target_platform=target_platform,
        )
        
        print(f"   Selected {len(clip_candidates)} candidates:")
        for clip in clip_candidates:
            print(f"   [{clip.id}] {format_timestamp(clip.start_time)}-"
                  f"{format_timestamp(clip.end_time)} ({clip.duration:.1f}s) "
                  f"score:{clip.virality_score}/10")
        
        if not clip_candidates:
            return self._build_empty_result(video_title, duration, target_platform, language)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: Supervision (LLM + TOON)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("🔍 PHASE 4: Supervision (LLM + TOON)")
        print(f"{'─'*70}")
        
        validated_clips = self._supervise_clips(
            clips=clip_candidates,
            transcript=transcript,
            sentence_extractor=sentence_extractor,
            max_clips=max_clips,
            language=language,
        )
        
        print(f"   Approved: {len(validated_clips)}/{len(clip_candidates)}")
        
        if not validated_clips:
            # Fallback: use top candidates without supervision
            print("   ⚠️ No clips approved, using top candidates")
            validated_clips = clip_candidates[:max_clips]
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: Word Alignment (Local)
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("📐 PHASE 5: Word Alignment (Local)")
        print(f"{'─'*70}")
        
        final_clips = self._align_to_words(
            clips=validated_clips,
            words=transcript.get("words", []),
        )
        
        print(f"   Aligned {len(final_clips)} clips to word boundaries")
        
        # ═══════════════════════════════════════════════════════════════
        # Build Result
        # ═══════════════════════════════════════════════════════════════
        
        result = self._build_result(
            video_title=video_title,
            duration=duration,
            target_platform=target_platform,
            language=language,
            scenes=scenes,
            viable_scenes=viable_scenes,
            clip_candidates=clip_candidates,
            final_clips=final_clips,
        )
        
        self._print_final_summary(result, final_clips)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Scene Classification
    # ═══════════════════════════════════════════════════════════════════════
    
    def _classify_scenes(
        self,
        scenes: List[DetectedScene],
        transcript: dict,
        video_title: str,
        language: str,
    ) -> List[DetectedScene]:
        """Classify scenes using TOON format."""
        
        utterances = transcript.get("utterances", [])
        
        # Build scene data with transcripts
        scene_data = []
        for scene in scenes:
            scene_utts = utterances[scene.start_utterance_idx:scene.end_utterance_idx + 1]
            text = " ".join(u["text"] for u in scene_utts)[:400]
            
            scene_data.append({
                "id": scene.id,
                "start": round(scene.start_time, 2),
                "end": round(scene.end_time, 2),
                "dur": round(scene.duration, 1),
                "spks": ",".join(map(str, scene.speakers)),
                "text": text,
            })
        
        # Prepare data wrapped in "scenes" key for TOON root object
        toon_data = {"scenes": scene_data}
        scenes_toon = toon.encode(toon_data, delimiter="\t")
        
        system_prompt = self._get_classifier_system_prompt(language)
        
        user_prompt = self._get_classifier_user_prompt(
            scenes_toon=scenes_toon,
            video_title=video_title,
            language=language,
        )
        
        response = self._call_llm(
            model=self.models["classifier"],
            system=system_prompt,
            user=user_prompt,
            phase="classifier",
        )
        
        # Parse TOON response
        classifications = ToonResponseParser.parse(response)
        class_map = {}
        for c in classifications:
            try:
                # Handle int, string "1", string " 1 "
                cid = int(str(c.get("id", 0)).strip())
                class_map[cid] = c
            except ValueError:
                continue
        
        for scene in scenes:
            if scene.id in class_map:
                c = class_map[scene.id]
                try:
                    scene.scene_type = str(c.get("type", c.get("scene_type", "other")))
                    scene.title = str(c.get("title", ""))
                    scene.mood = str(c.get("mood", ""))
                    
                    val = c.get("viral", c.get("viral_potential", 5))
                    scene.viral_potential = int(str(val).strip()) if val is not None else 5
                except (ValueError, TypeError):
                    if self.debug: print(f"   ⚠️ Error parsing fields for scene {scene.id}")
                    scene.viral_potential = 5
        
        return scenes
    
    def _get_classifier_system_prompt(self, language: str) -> str:
        if language == "it":
            return """Sei un analista video esperto. Classifica ogni scena.

TIPI: interview, monologue, shoutout, reaction, transition, other
MOOD: funny, serious, emotional, chaotic, calm, exciting
VIRAL (1-10): Potenziale virale della scena

OUTPUT SOLO in formato TOON:
```toon
classifications[N]{id,type,title,mood,viral}:
  1	interview	"Intervista cosplayer"	funny	8
  2	shoutout	"Presentazione streamer"	calm	4
```"""
        else:
            return """You are an expert video analyst. Classify each scene.

TYPES: interview, monologue, shoutout, reaction, transition, other
MOOD: funny, serious, emotional, chaotic, calm, exciting
VIRAL (1-10): Viral potential of the scene

OUTPUT ONLY in TOON format:
```toon
classifications[N]{id,type,title,mood,viral}:
  1	interview	"Cosplayer interview"	funny	8
  2	shoutout	"Streamer presentation"	calm	4
```"""
    
    def _get_classifier_user_prompt(
        self,
        scenes_toon: str,
        video_title: str,
        language: str,
    ) -> str:
        if language == "it":
            return f"""Classifica queste scene del video "{video_title}".

SCENE DA CLASSIFICARE:
```toon
{scenes_toon}
```

Per ogni scena, valuta:
1. Tipo di contenuto (interview, monologue, shoutout, etc.)
2. Mood/tono (funny, serious, etc.)
3. Potenziale virale (1-10): 9-10=iconico, 7-8=interessante, 5-6=medio, 1-4=basso

Rispondi SOLO in formato TOON."""
        else:
            return f"""Classify these scenes from video "{video_title}".

SCENES TO CLASSIFY:
```toon
{scenes_toon}
```

For each scene, evaluate:
1. Content type (interview, monologue, shoutout, etc.)
2. Mood/tone (funny, serious, etc.)
3. Viral potential (1-10): 9-10=iconic, 7-8=interesting, 5-6=medium, 1-4=low

Respond ONLY in TOON format."""

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Clip Selection
    # ═══════════════════════════════════════════════════════════════════════
    
    def _select_clips(
        self,
        scenes: List[DetectedScene],
        transcript: dict,
        sentence_extractor: SentenceBoundaryExtractor,
        max_clips: int,
        language: str,
        target_platform: str,
    ) -> List[ClipCandidate]:
        """Select clips using TOON format."""
        
        utterances = transcript.get("utterances", [])
        
        # Take top scenes by viral potential
        top_scenes = scenes[:max_clips * 2]
        
        # Build scene data
        scene_data = []
        for scene in top_scenes:
            scene_utts = utterances[scene.start_utterance_idx:scene.end_utterance_idx + 1]
            text = " ".join(u["text"] for u in scene_utts)[:350]
            
            scene_data.append({
                "id": scene.id,
                "start": round(scene.start_time, 2),
                "end": round(scene.end_time, 2),
                "dur": round(scene.duration, 1),
                "type": scene.scene_type,
                "title": scene.title[:30],
                "viral": scene.viral_potential,
                "text": text,
            })
        
        # Prepare TOON data wrapped in "top_scenes"
        toon_data = {"top_scenes": scene_data}
        scenes_toon = toon.encode(toon_data, delimiter="\t")
        
        system_prompt = self._get_selector_system_prompt(language, target_platform)
        
        user_prompt = self._get_selector_user_prompt(
            scenes_toon=scenes_toon,
            max_clips=max_clips,
            language=language,
        )
        
        response = self._call_llm(
            model=self.models["selector"],
            system=system_prompt,
            user=user_prompt,
            phase="selector",
        )
        
        # Parse response
        clips_data = ToonResponseParser.parse(response)
        
        candidates = []
        for item in clips_data:
            try:
                if not isinstance(item, dict): continue
                
                # Safe conversions
                start = float(str(item.get("start", 0)).strip())
                end = float(str(item.get("end", 0)).strip())
                
                # Align to sentence boundaries
                aligned_start, aligned_end, _, _ = sentence_extractor.find_clean_boundaries(
                    start, end, max_drift=2.0
                )
                
                cid_val = item.get("id", len(candidates) + 1)
                cid = int(str(cid_val).strip()) if cid_val is not None else len(candidates) + 1
                
                sid_val = item.get("scene", item.get("scene_id", 0))
                sid = int(str(sid_val).strip()) if sid_val is not None else 0
                
                score_val = item.get("score", item.get("virality_score", 5))
                score = int(str(score_val).strip()) if score_val is not None else 5
                
                candidates.append(ClipCandidate(
                    id=cid,
                    scene_id=sid,
                    start_time=aligned_start,
                    end_time=aligned_end,
                    hook_text=str(item.get("hook", "")),
                    title=str(item.get("title", "")),
                    why_viral=str(item.get("why", item.get("why_viral", ""))),
                    virality_score=score,
                ))
            except (ValueError, TypeError) as e:
                if self.debug: print(f"   ⚠️ Skipping invalid clip candidate: {item} ({e})")
                continue
        
        # Filter by duration
        valid_candidates = [
            c for c in candidates
            if MIN_CLIP_DURATION <= c.duration <= MAX_CLIP_DURATION * 1.1
        ]
        
        return valid_candidates
    
    def _get_selector_system_prompt(self, language: str, platform: str) -> str:
        if language == "it":
            return f"""Sei un esperto di contenuti virali per {platform}.

REGOLE:
1. Hook FORTE nei primi 3 secondi
2. Contenuto AUTOCONTENUTO (comprensibile da solo)
3. Durata {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION} secondi
4. Inizia/finisce con frasi COMPLETE
5. NON mischiare scene/interviste diverse

OUTPUT SOLO in formato TOON:
```toon
clips[N]{{id,scene,start,end,hook,title,why,score}}:
  1	3	25.5	55.2	"Hook iniziale"	"Titolo"	"Motivo viralità"	9
```"""
        else:
            return f"""You are a viral content expert for {platform}.

RULES:
1. STRONG hook in first 3 seconds
2. SELF-CONTAINED content (understandable alone)
3. Duration {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION} seconds
4. Start/end with COMPLETE sentences
5. DON'T mix different scenes/interviews

OUTPUT ONLY in TOON format:
```toon
clips[N]{{id,scene,start,end,hook,title,why,score}}:
  1	3	25.5	55.2	"Opening hook"	"Title"	"Viral reason"	9
```"""
    
    def _get_selector_user_prompt(
        self,
        scenes_toon: str,
        max_clips: int,
        language: str,
    ) -> str:
        if language == "it":
            return f"""Seleziona le {max_clips} migliori clip da queste scene.

SCENE TOP (ordinate per potenziale):
```toon
{scenes_toon}
```

Per ogni clip specifica:
- id: numero progressivo
- scene: ID della scena di origine
- start, end: timestamp esatti (in secondi)
- hook: prima frase accattivante
- title: titolo breve
- why: perché diventerebbe virale
- score: punteggio 1-10

Rispondi SOLO in formato TOON."""
        else:
            return f"""Select the {max_clips} best clips from these scenes.

TOP SCENES (sorted by potential):
```toon
{scenes_toon}
```

For each clip specify:
- id: progressive number
- scene: source scene ID
- start, end: exact timestamps (in seconds)
- hook: catchy first sentence
- title: short title
- why: why it would go viral
- score: score 1-10

Respond ONLY in TOON format."""

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Supervision
    # ═══════════════════════════════════════════════════════════════════════
    
    def _supervise_clips(
        self,
        clips: List[ClipCandidate],
        transcript: dict,
        sentence_extractor: SentenceBoundaryExtractor,
        max_clips: int,
        language: str,
    ) -> List[ClipCandidate]:
        """Supervise clips using TOON format."""
        
        # Build clip data with context
        clip_data = []
        for clip in clips:
            # Get sentences in range
            sents = sentence_extractor.get_sentences_in_range(
                clip.start_time, clip.end_time
            )
            
            first_sent = sents[0]["text"][:50] if sents else ""
            last_sent = sents[-1]["text"][:50] if sents else ""
            
            clip_data.append({
                "id": clip.id,
                "start": round(clip.start_time, 2),
                "end": round(clip.end_time, 2),
                "dur": round(clip.duration, 1),
                "score": clip.virality_score,
                "first": first_sent,
                "last": last_sent,
            })
        
        # Prepare TOON data wrapped in "clips"
        toon_data = {"clips": clip_data}
        clips_toon = toon.encode(toon_data, delimiter="\t")
        
        system_prompt = self._get_supervisor_system_prompt(language)
        
        user_prompt = self._get_supervisor_user_prompt(
            clips_toon=clips_toon,
            max_clips=max_clips,
            language=language,
        )
        
        response = self._call_llm(
            model=self.models["supervisor"],
            system=system_prompt,
            user=user_prompt,
            phase="supervisor",
        )
        
        # Parse response
        approved, rejected = ToonResponseParser.parse_supervision_response(response)
        
        if self.debug:
            print(f"   Supervisor approved: {len(approved)}, rejected: {len(rejected)}")
        
        # Apply approvals
        # Apply approvals
        approved_ids = {}
        for a in approved:
            try:
                aid = int(str(a.get("id", 0)).strip())
                approved_ids[aid] = a
            except ValueError:
                continue

        validated = []
        
        for clip in clips:
            if clip.id in approved_ids:
                approval = approved_ids[clip.id]
                
                try:
                    # Apply adjustments
                    adj_start = approval.get("adj_start", approval.get("adjusted_start"))
                    adj_end = approval.get("adj_end", approval.get("adjusted_end"))
                    
                    if adj_start is not None and str(adj_start).lower() != "null":
                        clip.start_time = float(str(adj_start).strip())
                    if adj_end is not None and str(adj_end).lower() != "null":
                        clip.end_time = float(str(adj_end).strip())
                    
                    new_score = approval.get("score", approval.get("final_score"))
                    if new_score is not None and str(new_score).lower() != "null":
                        clip.virality_score = int(str(new_score).strip())
                    
                    validated.append(clip)
                except (ValueError, TypeError) as e:
                    if self.debug: print(f"   ⚠️ Error applying supervision to clip {clip.id}: {e}")
                    # Keep clip but maybe warn? Or skip validation update and assume original is valid?
                    # Let's keep original if update fails but mark as validated since it was in approved list.
                    validated.append(clip)
        
        # Sort by score and limit
        validated.sort(key=lambda c: c.virality_score, reverse=True)
        return validated[:max_clips]
    
    def _get_supervisor_system_prompt(self, language: str) -> str:
        if language == "it":
            return f"""Sei un supervisore qualità per clip video.

CHECKLIST VALIDAZIONE:
1. Prima frase = hook forte?
2. Ultima frase = conclusione naturale?
3. Contenuto autocontenuto?
4. Durata appropriata ({MIN_CLIP_DURATION}-{MAX_CLIP_DURATION}s)?
5. Non taglia a metà pensiero?

Sii RIGOROSO - meglio rifiutare clip dubbie.

OUTPUT in formato TOON:
```toon
approved[N]{{id,adj_start,adj_end,score,notes}}:
  1	null	null	9	"Perfetta"
  3	26.5	null	8	"Aggiustato inizio"

rejected[N]{{id,reason}}:
  2	"Finisce a metà frase"
```"""
        else:
            return f"""You are a quality supervisor for video clips.

VALIDATION CHECKLIST:
1. First sentence = strong hook?
2. Last sentence = natural ending?
3. Self-contained content?
4. Appropriate duration ({MIN_CLIP_DURATION}-{MAX_CLIP_DURATION}s)?
5. Doesn't cut mid-thought?

Be STRICT - better to reject questionable clips.

OUTPUT in TOON format:
```toon
approved[N]{{id,adj_start,adj_end,score,notes}}:
  1	null	null	9	"Perfect"
  3	26.5	null	8	"Adjusted start"

rejected[N]{{id,reason}}:
  2	"Ends mid-sentence"
```"""
    
    def _get_supervisor_user_prompt(
        self,
        clips_toon: str,
        max_clips: int,
        language: str,
    ) -> str:
        if language == "it":
            return f"""Valida queste clip e seleziona le TOP {max_clips}.

CLIP DA VALIDARE:
```toon
{clips_toon}
```

Per ogni clip:
- APPROVA se passa tutti i controlli
- RIFIUTA con motivazione se ha problemi
- Puoi aggiustare leggermente start/end se necessario

Rispondi in formato TOON con sezioni 'approved' e 'rejected'."""
        else:
            return f"""Validate these clips and select TOP {max_clips}.

CLIPS TO VALIDATE:
```toon
{clips_toon}
```

For each clip:
- APPROVE if it passes all checks
- REJECT with reason if it has issues
- You can slightly adjust start/end if needed

Respond in TOON format with 'approved' and 'rejected' sections."""

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: Word Alignment
    # ═══════════════════════════════════════════════════════════════════════
    
    def _align_to_words(
        self,
        clips: List[ClipCandidate],
        words: List[dict],
    ) -> List[FinalClip]:
        """Align clip boundaries to word timestamps."""
        final_clips = []
        
        for i, clip in enumerate(clips):
            # Find words in range
            clip_words = [
                (idx, w) for idx, w in enumerate(words)
                if w["start"] >= clip.start_time - 0.5
                and w["end"] <= clip.end_time + 0.5
            ]
            
            if not clip_words:
                continue
            
            first_idx, first_word = clip_words[0]
            last_idx, last_word = clip_words[-1]
            
            # Add padding
            final_start = max(0, first_word["start"] - 0.15)
            final_end = last_word["end"] + 0.25
            
            # Get text excerpts
            first_words = " ".join(
                w.get("punctuated_word", w["word"])
                for _, w in clip_words[:10]
            )
            last_words = " ".join(
                w.get("punctuated_word", w["word"])
                for _, w in clip_words[-10:]
            )
            
            final_clips.append(FinalClip(
                clip_number=i + 1,
                start_time=round(final_start, 3),
                end_time=round(final_end, 3),
                duration=round(final_end - final_start, 2),
                first_word_idx=first_idx,
                last_word_idx=last_idx,
                first_words=first_words,
                last_words=last_words,
                scene_id=clip.scene_id,
                title=clip.title,
                hook_text=clip.hook_text,
                virality_score=clip.virality_score,
                why_viral=clip.why_viral,
            ))
        
        return final_clips

    # ═══════════════════════════════════════════════════════════════════════
    # LLM Communication
    # ═══════════════════════════════════════════════════════════════════════
    
    def _call_llm(
        self,
        model: str,
        system: str,
        user: str,
        phase: str,
    ) -> str:
        """Call LLM and track usage."""
        
        if self.debug:
            print(f"   Calling {phase} ({model})...")
            print(f"   System prompt: {len(system)} chars")
            print(f"   User prompt: {len(user)} chars")
        
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.app_url,
                    "X-Title": self.app_name,
                },
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content
            
            # Track usage
            if response.usage:
                tokens = response.usage.total_tokens
                self.tokens[phase] = tokens
                self.tokens["total"] += tokens
                
                # Estimate cost (~$0.30/1M tokens average)
                cost = tokens * 0.0000003
                self.costs[phase] = cost
                self.costs["total"] += cost
                
                if self.debug:
                    print(f"   {phase}: {tokens} tokens, ${cost:.6f}")
            
            return content
            
        except Exception as e:
            error_str = str(e)
            if "No endpoints found" in error_str:
                raise RuntimeError(
                    f"\n❌ ERRORE POLICY OPENROUTER:\n"
                    f"   Il modello '{model}' non è disponibile.\n"
                    f"   👉 Vai su https://openrouter.ai/settings/privacy\n"
                    f"   👉 Oppure scegli un altro modello."
                ) from None
            raise RuntimeError(f"LLM call failed ({phase}): {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Result Building
    # ═══════════════════════════════════════════════════════════════════════
    
    def _build_result(
        self,
        video_title: str,
        duration: float,
        target_platform: str,
        language: str,
        scenes: List[DetectedScene],
        viable_scenes: List[DetectedScene],
        clip_candidates: List[ClipCandidate],
        final_clips: List[FinalClip],
    ) -> dict:
        """Build final result dictionary."""
        return {
            "video_title": video_title,
            "video_duration": duration,
            "target_platform": target_platform,
            "pipeline": "enhanced-multiagent-v3.1-toon",
            "analysis_timestamp": datetime.now().isoformat(),
            
            # Scene stats
            "scenes_detected": len(scenes),
            "scenes_excluded": len(scenes) - len(viable_scenes),
            "scenes_viable": len(viable_scenes),
            
            # Clip stats
            "clips_proposed": len(clip_candidates),
            "clips_found": len(final_clips),
            "clips": [c.to_dict() for c in final_clips],
            
            # Cost tracking
            "tokens": self.tokens,
            "costs": self.costs,
            "models_used": self.models,
            "language": language,
        }
    
    def _build_empty_result(
        self,
        video_title: str,
        duration: float,
        target_platform: str,
        language: str,
    ) -> dict:
        """Build result when no clips found."""
        return {
            "video_title": video_title,
            "video_duration": duration,
            "target_platform": target_platform,
            "pipeline": "enhanced-multiagent-v3.1-toon",
            "analysis_timestamp": datetime.now().isoformat(),
            "scenes_detected": 0,
            "clips_found": 0,
            "clips": [],
            "error": "No viable scenes found",
            "tokens": self.tokens,
            "costs": self.costs,
            "language": language,
        }
    
    def _print_final_summary(self, result: dict, clips: List[FinalClip]):
        """Print final summary."""
        print(f"\n{'═'*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'═'*70}")
        print(f"   Scenes: {result['scenes_detected']} detected, "
              f"{result['scenes_excluded']} excluded, "
              f"{result['scenes_viable']} viable")
        print(f"   Clips: {result['clips_proposed']} proposed → "
              f"{result['clips_found']} final")
        
        if clips:
            print(f"\n   📋 FINAL CLIPS:")
            for clip in clips:
                print(f"\n   [{clip.clip_number}] {clip.title}")
                print(f"       ⏱️  {format_timestamp(clip.start_time)} → "
                      f"{format_timestamp(clip.end_time)} ({clip.duration}s)")
                print(f"       🎣 Hook: \"{clip.hook_text[:50]}...\"")
                print(f"       ⭐ Score: {clip.virality_score}/10")
        
        print(f"\n   💰 Token usage:")
        for phase, tokens in self.tokens.items():
            if phase != "total" and tokens > 0:
                print(f"      {phase}: {tokens} tokens (${self.costs.get(phase, 0):.6f})")
        print(f"      TOTAL: {self.tokens['total']} tokens (${self.costs['total']:.6f})")
    
    def save_analysis(self, result: dict, video_id: str = "analysis") -> str:
        """Save analysis to file."""
        output_path = ANALYSIS_DIR / f"{video_id}_analysis_v31.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n   💾 Saved: {output_path}")
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_video_enhanced(
    transcript: dict,
    video_title: str,
    max_clips: int = 5,
    language: str = "it",
    target_platform: str = "all",
    debug: bool = False,
) -> dict:
    """
    Analyze video with enhanced multi-agent pipeline.
    Drop-in replacement for analyze_for_clips().
    """
    analyzer = EnhancedMultiAgentAnalyzer(debug=debug)
    return analyzer.analyze(
        transcript=transcript,
        video_title=video_title,
        max_clips=max_clips,
        language=language,
        target_platform=target_platform,
    )


def estimate_token_savings(transcript: dict) -> dict:
    """
    Estimate token savings from using TOON vs JSON.
    
    Returns dict with savings info for utterances and words.
    """
    import json as json_module
    
    results = {}
    
    for field in ["utterances", "words"]:
        array = transcript.get(field, [])
        if not array:
            continue
        
        # Limit for estimation
        sample = array[:100] if len(array) > 100 else array
        
        # JSON size
        json_str = json_module.dumps(sample, ensure_ascii=False)
        json_chars = len(json_str)
        
        # TOON size  
        if field == "utterances":
            toon_str = TranscriptToToon.utterances({"utterances": sample})
        else:
            toon_str = TranscriptToToon.words({"words": sample})
        toon_chars = len(toon_str)
        
        # Token estimates (rough: 1 token ≈ 3.5 chars for mixed content)
        json_tokens = int(json_chars / 3.5)
        toon_tokens = int(toon_chars / 3.5)
        savings_pct = (1 - toon_tokens / json_tokens) * 100 if json_tokens > 0 else 0
        
        # Extrapolate to full array
        scale = len(array) / len(sample)
        
        results[field] = {
            "items": len(array),
            "json_tokens": int(json_tokens * scale),
            "toon_tokens": int(toon_tokens * scale),
            "savings_percent": round(savings_pct, 1),
        }
    
    return results
