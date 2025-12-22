# enhanced_analyzer.py
"""
Enhanced Multi-Agent Video Analyzer v4.0
- Full TOON v3.0 specification compliance
- 4-phase pipeline: Scene Detection → Classification → Clip Selection → Supervision
- Word-level boundary alignment to prevent truncation
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from openai import OpenAI
import os

# ============================================================================
# TOON v3.0 Encoder/Decoder
# ============================================================================

class ToonV3:
    """TOON v3.0 compliant encoder/decoder"""
    
    @staticmethod
    def _needs_quoting(s: str) -> bool:
        """Check if string needs quoting per TOON v3.0 spec"""
        if not s:
            return True
        if s in ('true', 'false', 'null'):
            return True
        if s[0] in ' \t"\'#[{' or s[-1] in ' \t':
            return True
        if '\n' in s or '\t' in s:
            return True
        # Check if looks like number
        try:
            float(s)
            return True
        except ValueError:
            pass
        return False
    
    @staticmethod
    def _quote_string(s: str) -> str:
        """Quote string with proper escaping"""
        escaped = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        return f'"{escaped}"'
    
    @staticmethod
    def _encode_value(value, indent: int = 0) -> str:
        """Encode a single value to TOON format"""
        prefix = '\t' * indent
        
        if value is None:
            return 'null'
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            if ToonV3._needs_quoting(value):
                return ToonV3._quote_string(value)
            return value
        elif isinstance(value, list):
            return ToonV3._encode_array(value, indent)
        elif isinstance(value, dict):
            return ToonV3._encode_object(value, indent)
        else:
            return ToonV3._quote_string(str(value))
    
    @staticmethod
    def _encode_object(obj: dict, indent: int = 0) -> str:
        """Encode object to TOON format"""
        if not obj:
            return '{}'
        
        lines = []
        prefix = '\t' * indent
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{prefix}{key}")
                lines.append(ToonV3._encode_value(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {ToonV3._encode_value(value, indent)}")
        return '\n'.join(lines)
    
    @staticmethod
    def _is_tabular(arr: list) -> bool:
        """Check if array is suitable for tabular encoding"""
        if len(arr) < 1:
            return False
        if not all(isinstance(item, dict) for item in arr):
            return False
        # All items must have same keys
        keys = set(arr[0].keys())
        if not all(set(item.keys()) == keys for item in arr):
            return False
        # All values must be primitives
        for item in arr:
            for v in item.values():
                if isinstance(v, (dict, list)):
                    return False
        return True
    
    @staticmethod
    def _encode_array(arr: list, indent: int = 0) -> str:
        """Encode array to TOON format, using tabular for homogeneous objects"""
        if not arr:
            return '[]'
        
        prefix = '\t' * indent
        
        # Tabular encoding for arrays of homogeneous objects
        if ToonV3._is_tabular(arr):
            keys = list(arr[0].keys())
            lines = []
            # Header: [count | field1 field2 ...]
            lines.append(f"{prefix}[{len(arr)} | {' '.join(keys)}]")
            # Rows: tab-separated values
            for item in arr:
                values = []
                for k in keys:
                    v = item[k]
                    if v is None:
                        values.append('-')
                    elif isinstance(v, bool):
                        values.append('true' if v else 'false')
                    elif isinstance(v, str):
                        # Escape tabs in values
                        v = v.replace('\t', ' ').replace('\n', ' ')
                        if ToonV3._needs_quoting(v):
                            values.append(ToonV3._quote_string(v))
                        else:
                            values.append(v)
                    else:
                        values.append(str(v))
                lines.append(f"{prefix}\t{chr(9).join(values)}")
            return '\n'.join(lines)
        
        # Non-tabular array
        lines = [f"{prefix}[{len(arr)}]"]
        for item in arr:
            lines.append(ToonV3._encode_value(item, indent + 1))
        return '\n'.join(lines)
    
    @staticmethod
    def encode(data, root_key: str = "data") -> str:
        """Encode data to TOON v3.0 format"""
        if isinstance(data, list):
            return ToonV3._encode_array(data, 0)
        elif isinstance(data, dict):
            return ToonV3._encode_object(data, 0)
        else:
            return ToonV3._encode_value(data, 0)
    
    @staticmethod
    def decode(toon_str: str) -> any:
        """Decode TOON v3.0 format to Python data"""
        lines = toon_str.strip().split('\n')
        if not lines:
            return None
        
        # Simple tabular array parser
        if lines[0].startswith('[') and '|' in lines[0]:
            match = re.match(r'\[(\d+)\s*\|\s*(.+)\]', lines[0].strip())
            if match:
                count = int(match.group(1))
                fields = match.group(2).split()
                result = []
                for i, line in enumerate(lines[1:count+1]):
                    values = line.strip().split('\t')
                    obj = {}
                    for j, field in enumerate(fields):
                        if j < len(values):
                            v = values[j].strip()
                            if v == '-':
                                obj[field] = None
                            elif v == 'true':
                                obj[field] = True
                            elif v == 'false':
                                obj[field] = False
                            elif v.startswith('"') and v.endswith('"'):
                                obj[field] = v[1:-1].replace('\\n', '\n').replace('\\t', '\t')
                            else:
                                try:
                                    obj[field] = float(v) if '.' in v else int(v)
                                except ValueError:
                                    obj[field] = v
                        else:
                            obj[field] = None
                    result.append(obj)
                return result
        
        # Fallback: try JSON
        try:
            return json.loads(toon_str)
        except json.JSONDecodeError:
            return None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Word:
    """Single word with precise timing"""
    word: str
    start: float
    end: float
    confidence: float = 1.0
    punctuated_word: str = ""

@dataclass
class Utterance:
    """Speech segment from one speaker"""
    id: int
    speaker: int
    start: float
    end: float
    transcript: str
    words: list[Word] = field(default_factory=list)

@dataclass
class Scene:
    """Detected scene with classification"""
    id: int
    start: float
    end: float
    start_word_idx: int
    end_word_idx: int
    speaker: int
    transcript: str
    scene_type: str = ""  # interview, intro, outro, transition, monologue
    description: str = ""
    viral_score: int = 0
    participants: list[str] = field(default_factory=list)

@dataclass
class Clip:
    """Final clip with aligned boundaries"""
    id: int
    scene_id: int
    start: float
    end: float
    duration: float
    start_word: str
    end_word: str
    title: str
    description: str
    viral_score: int
    quality_notes: list[str] = field(default_factory=list)


# ============================================================================
# Scene Detector (Local)
# ============================================================================

class SceneDetector:
    """Detects scene boundaries from transcript without LLM"""
    
    def __init__(
        self,
        min_pause_seconds: float = 1.5,
        speaker_change_gap: float = 0.3,
        min_scene_duration: float = 5.0
    ):
        self.min_pause = min_pause_seconds
        self.speaker_gap = speaker_change_gap
        self.min_duration = min_scene_duration
    
    def detect(self, utterances: list[Utterance], words: list[Word]) -> list[Scene]:
        """Detect scenes based on speaker changes and pauses"""
        if not utterances:
            return []
        
        scenes = []
        current_scene_start = 0
        current_scene_start_time = utterances[0].start
        current_speaker = utterances[0].speaker
        current_text = []
        
        for i, utt in enumerate(utterances):
            gap = 0
            if i > 0:
                gap = utt.start - utterances[i-1].end
            
            # New scene on: speaker change, or long pause
            is_speaker_change = utt.speaker != current_speaker
            is_long_pause = gap > self.min_pause
            
            if (is_speaker_change or is_long_pause) and i > 0:
                scene_end = utterances[i-1].end
                scene_duration = scene_end - current_scene_start_time
                
                if scene_duration >= self.min_duration:
                    # Find word indices for this scene
                    start_idx, end_idx = self._find_word_indices(
                        words, current_scene_start_time, scene_end
                    )
                    
                    scenes.append(Scene(
                        id=len(scenes),
                        start=current_scene_start_time,
                        end=scene_end,
                        start_word_idx=start_idx,
                        end_word_idx=end_idx,
                        speaker=current_speaker,
                        transcript=' '.join(current_text)
                    ))
                
                # Start new scene
                current_scene_start_time = utt.start
                current_speaker = utt.speaker
                current_text = []
            
            current_text.append(utt.transcript)
        
        # Add final scene
        if utterances:
            last_utt = utterances[-1]
            scene_duration = last_utt.end - current_scene_start_time
            if scene_duration >= self.min_duration:
                start_idx, end_idx = self._find_word_indices(
                    words, current_scene_start_time, last_utt.end
                )
                scenes.append(Scene(
                    id=len(scenes),
                    start=current_scene_start_time,
                    end=last_utt.end,
                    start_word_idx=start_idx,
                    end_word_idx=end_idx,
                    speaker=current_speaker,
                    transcript=' '.join(current_text)
                ))
        
        return scenes
    
    def _find_word_indices(
        self, words: list[Word], start_time: float, end_time: float
    ) -> tuple[int, int]:
        """Find word indices for a time range"""
        start_idx = 0
        end_idx = len(words) - 1
        
        for i, w in enumerate(words):
            if w.start >= start_time:
                start_idx = i
                break
        
        for i in range(len(words) - 1, -1, -1):
            if words[i].end <= end_time:
                end_idx = i
                break
        
        return start_idx, end_idx


# ============================================================================
# Word Boundary Aligner
# ============================================================================

class WordBoundaryAligner:
    """Aligns clip boundaries to word/sentence edges"""
    
    def __init__(self, words: list[Word]):
        self.words = words
        self.sentence_endings = {'.', '!', '?'}
        self._build_sentence_map()
    
    def _build_sentence_map(self):
        """Build map of sentence boundaries"""
        self.sentences = []
        current_start = 0
        
        for i, word in enumerate(self.words):
            text = word.punctuated_word or word.word
            if text and text[-1] in self.sentence_endings:
                self.sentences.append({
                    'start_idx': current_start,
                    'end_idx': i,
                    'start_time': self.words[current_start].start,
                    'end_time': word.end
                })
                current_start = i + 1
        
        # Add remaining words as final sentence
        if current_start < len(self.words):
            self.sentences.append({
                'start_idx': current_start,
                'end_idx': len(self.words) - 1,
                'start_time': self.words[current_start].start,
                'end_time': self.words[-1].end
            })
    
    def align_to_words(self, start: float, end: float) -> tuple[float, float, str, str]:
        """Align times to exact word boundaries"""
        start_word_idx = 0
        end_word_idx = len(self.words) - 1
        
        # Find first word starting at or after start time
        for i, w in enumerate(self.words):
            if w.start >= start - 0.1:  # 100ms tolerance
                start_word_idx = i
                break
        
        # Find last word ending at or before end time
        for i in range(len(self.words) - 1, -1, -1):
            if self.words[i].end <= end + 0.1:
                end_word_idx = i
                break
        
        if start_word_idx > end_word_idx:
            start_word_idx = end_word_idx
        
        aligned_start = self.words[start_word_idx].start
        aligned_end = self.words[end_word_idx].end
        start_word = self.words[start_word_idx].word
        end_word = self.words[end_word_idx].word
        
        return aligned_start, aligned_end, start_word, end_word
    
    def align_to_sentences(self, start: float, end: float) -> tuple[float, float, str, str]:
        """Align times to sentence boundaries for cleaner cuts"""
        # Find sentence containing start time
        start_sentence = None
        for sent in self.sentences:
            if sent['start_time'] <= start <= sent['end_time']:
                start_sentence = sent
                break
            if sent['start_time'] > start:
                start_sentence = sent
                break
        
        # Find sentence containing end time
        end_sentence = None
        for sent in reversed(self.sentences):
            if sent['start_time'] <= end <= sent['end_time']:
                end_sentence = sent
                break
            if sent['end_time'] < end:
                end_sentence = sent
                break
        
        if not start_sentence:
            start_sentence = self.sentences[0] if self.sentences else None
        if not end_sentence:
            end_sentence = self.sentences[-1] if self.sentences else None
        
        if start_sentence and end_sentence:
            aligned_start = start_sentence['start_time']
            aligned_end = end_sentence['end_time']
            start_word = self.words[start_sentence['start_idx']].word
            end_word = self.words[end_sentence['end_idx']].word
            return aligned_start, aligned_end, start_word, end_word
        
        return self.align_to_words(start, end)


# ============================================================================
# Transcript to TOON Converter
# ============================================================================

class TranscriptToToon:
    """Converts transcript data to TOON v3.0 format"""
    
    @staticmethod
    def scenes(scenes: list[Scene]) -> str:
        """Convert scenes to TOON tabular format for LLM"""
        data = [{
            'id': s.id,
            'start': round(s.start, 2),
            'end': round(s.end, 2),
            'speaker': s.speaker,
            'transcript': s.transcript[:200]  # Truncate for context window
        } for s in scenes]
        return ToonV3.encode(data)
    
    @staticmethod
    def scenes_classified(scenes: list[Scene]) -> str:
        """Convert classified scenes to TOON"""
        data = [{
            'id': s.id,
            'start': round(s.start, 2),
            'end': round(s.end, 2),
            'type': s.scene_type,
            'viral_score': s.viral_score,
            'description': s.description[:100]
        } for s in scenes]
        return ToonV3.encode(data)
    
    @staticmethod
    def clips(clips: list[Clip]) -> str:
        """Convert clips to TOON for review"""
        data = [{
            'id': c.id,
            'scene_id': c.scene_id,
            'start': round(c.start, 2),
            'end': round(c.end, 2),
            'duration': round(c.duration, 2),
            'title': c.title,
            'viral_score': c.viral_score
        } for c in clips]
        return ToonV3.encode(data)


# ============================================================================
# Multi-Agent Analyzer
# ============================================================================

class EnhancedMultiAgentAnalyzer:
    """
    Multi-Agent Video Analyzer v4.0
    
    Pipeline:
    1. Scene Detection (local) - identify natural boundaries
    2. Scene Classification (LLM) - categorize and score scenes
    3. Clip Selection (LLM) - select best moments
    4. Supervision (LLM) - validate and refine
    5. Word Alignment (local) - precise boundary alignment
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
        min_clip_duration: float = 15.0,
        max_clip_duration: float = 60.0,
        max_clips: int = 5,
        debug: bool = False
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.min_duration = min_clip_duration
        self.max_duration = max_clip_duration
        self.max_clips = max_clips
        self.debug = debug
        
        self.scene_detector = SceneDetector()
    
    def analyze(self, transcript_path: str) -> dict:
        """Run full analysis pipeline"""
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Parse transcript
        words = self._parse_words(transcript_data)
        utterances = self._parse_utterances(transcript_data)
        
        if self.debug:
            print(f"[DEBUG] Loaded {len(words)} words, {len(utterances)} utterances")
        
        # Phase 1: Local scene detection
        scenes = self.scene_detector.detect(utterances, words)
        if self.debug:
            print(f"[DEBUG] Detected {len(scenes)} scenes")
        
        # Phase 2: LLM scene classification
        scenes = self._phase2_classify_scenes(scenes)
        
        # Filter out intro/outro
        eligible_scenes = [s for s in scenes if s.scene_type not in ('intro', 'outro', 'transition')]
        if self.debug:
            print(f"[DEBUG] {len(eligible_scenes)} eligible scenes after filtering")
        
        # Phase 3: LLM clip selection
        clips = self._phase3_select_clips(eligible_scenes)
        
        # Phase 4: LLM supervision
        clips = self._phase4_supervise(clips, scenes)
        
        # Phase 5: Word boundary alignment
        aligner = WordBoundaryAligner(words)
        final_clips = self._phase5_align_boundaries(clips, aligner)
        
        return {
            'scenes': [asdict(s) for s in scenes],
            'clips': [asdict(c) for c in final_clips],
            'metadata': {
                'total_scenes': len(scenes),
                'eligible_scenes': len(eligible_scenes),
                'final_clips': len(final_clips)
            }
        }
    
    def _parse_words(self, data: dict) -> list[Word]:
        """Parse words from transcript"""
        words = []
        # Try raw Deepgram structure
        word_data = data.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('words', [])
        
        # Fallback to flattened structure
        if not word_data:
            word_data = data.get('words', [])
        
        for w in word_data:
            words.append(Word(
                word=w.get('word', ''),
                start=w.get('start', 0),
                end=w.get('end', 0),
                confidence=w.get('confidence', 1.0),
                punctuated_word=w.get('punctuated_word', w.get('word', ''))
            ))
        return words
    
    def _parse_utterances(self, data: dict) -> list[Utterance]:
        """Parse utterances from transcript"""
        utterances = []
        # Try raw Deepgram structure
        utt_data = data.get('results', {}).get('utterances', [])
        
        # Fallback to flattened structure
        if not utt_data:
            utt_data = data.get('utterances', [])
        
        for i, u in enumerate(utt_data):
            words = [Word(
                word=w.get('word', ''),
                start=w.get('start', 0),
                end=w.get('end', 0),
                confidence=w.get('confidence', 1.0),
                punctuated_word=w.get('punctuated_word', w.get('word', ''))
            ) for w in u.get('words', [])]
            
            utterances.append(Utterance(
                id=i,
                speaker=u.get('speaker', 0),
                start=u.get('start', 0),
                end=u.get('end', 0),
                transcript=u.get('transcript', ''),
                words=words
            ))
        return utterances
    
    def _phase2_classify_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """Classify scenes using LLM"""
        if not scenes:
            return scenes
        
        scenes_toon = TranscriptToToon.scenes(scenes)
        
        prompt = f"""Analizza queste scene video e classifica ciascuna.

SCENE (formato TOON):
{scenes_toon}

Per ogni scena, rispondi in formato TOON tabulare con questi campi:
- id: numero scena
- type: intro | outro | interview | monologue | transition | highlight
- viral_score: 1-10 (potenziale virale)
- description: breve descrizione (max 50 caratteri)
- participants: chi parla (es. "host,guest")

Criteri di classificazione:
- intro/outro: presentazioni, saluti, call-to-action, richieste iscrizione
- interview: domande e risposte tra persone diverse
- monologue: una persona parla da sola
- highlight: momento divertente, emotivo o memorabile
- transition: passaggio tra argomenti

Rispondi SOLO con la tabella TOON, nessun altro testo."""

        response = self._call_llm(prompt)
        
        # Parse response
        try:
            classifications = ToonV3.decode(response)
            if classifications:
                for cls in classifications:
                    scene_id = cls.get('id', -1)
                    for scene in scenes:
                        if scene.id == scene_id:
                            scene.scene_type = cls.get('type', 'unknown')
                            scene.viral_score = int(cls.get('viral_score', 5))
                            scene.description = cls.get('description', '')
                            break
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Classification parse error: {e}")
        
        return scenes
    
    def _phase3_select_clips(self, scenes: list[Scene]) -> list[Clip]:
        """Select best clips from eligible scenes"""
        if not scenes:
            return []
        
        scenes_toon = TranscriptToToon.scenes_classified(scenes)
        
        prompt = f"""Seleziona le migliori clip da queste scene per video virali brevi (Shorts/TikTok/Reels).

SCENE DISPONIBILI (formato TOON):
{scenes_toon}

REQUISITI CLIP:
- Durata: {self.min_duration}-{self.max_duration} secondi
- Massimo {self.max_clips} clip
- Ogni clip deve essere AUTO-CONTENUTA (inizio e fine logici)
- NON tagliare a metà una battuta o intervista
- Preferisci momenti con hook forte all'inizio

Rispondi in formato TOON tabulare con questi campi:
- id: numero clip (0, 1, 2...)
- scene_id: id della scena di origine
- start: tempo inizio in secondi
- end: tempo fine in secondi
- title: titolo accattivante (max 50 caratteri)
- viral_score: 1-10

IMPORTANTE: I tempi start/end devono essere DENTRO i confini della scena.

Rispondi SOLO con la tabella TOON."""

        response = self._call_llm(prompt)
        
        clips = []
        try:
            clip_data = ToonV3.decode(response)
            if clip_data:
                for c in clip_data:
                    clip = Clip(
                        id=c.get('id', len(clips)),
                        scene_id=c.get('scene_id', 0),
                        start=float(c.get('start', 0)),
                        end=float(c.get('end', 0)),
                        duration=float(c.get('end', 0)) - float(c.get('start', 0)),
                        start_word='',
                        end_word='',
                        title=c.get('title', ''),
                        description='',
                        viral_score=int(c.get('viral_score', 5))
                    )
                    clips.append(clip)
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Clip selection parse error: {e}")
        
        return clips
    
    def _phase4_supervise(self, clips: list[Clip], all_scenes: list[Scene]) -> list[Clip]:
        """Supervise and validate clips"""
        if not clips:
            return clips
        
        clips_toon = TranscriptToToon.clips(clips)
        scenes_toon = TranscriptToToon.scenes_classified(all_scenes)
        
        prompt = f"""Sei un supervisore qualità per clip video. Valida queste clip proposte.

CLIP PROPOSTE:
{clips_toon}

TUTTE LE SCENE DEL VIDEO:
{scenes_toon}

VERIFICA PER OGNI CLIP:
1. La clip è completamente dentro una singola scena? (non mescola interviste diverse)
2. I tempi sono validi? (start < end, durata {self.min_duration}-{self.max_duration}s)
3. La clip evita intro/outro?
4. La clip ha senso come contenuto standalone?

Rispondi in formato TOON tabulare con:
- id: id clip originale
- approved: true/false
- adjusted_start: nuovo start (o stesso se ok)
- adjusted_end: nuovo end (o stesso se ok)
- issue: problema riscontrato (o "none")

Se una clip mescola contenuti di scene diverse, NON approvarla.

Rispondi SOLO con la tabella TOON."""

        response = self._call_llm(prompt)
        
        approved_clips = []
        try:
            validations = ToonV3.decode(response)
            if validations:
                for v in validations:
                    clip_id = v.get('id', -1)
                    if v.get('approved', False):
                        for clip in clips:
                            if clip.id == clip_id:
                                clip.start = float(v.get('adjusted_start', clip.start))
                                clip.end = float(v.get('adjusted_end', clip.end))
                                clip.duration = clip.end - clip.start
                                issue = v.get('issue', 'none')
                                if issue and issue != 'none':
                                    clip.quality_notes.append(issue)
                                approved_clips.append(clip)
                                break
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Supervision parse error: {e}")
            approved_clips = clips  # Fallback: keep all
        
        return approved_clips
    
    def _phase5_align_boundaries(self, clips: list[Clip], aligner: WordBoundaryAligner) -> list[Clip]:
        """Align clip boundaries to word/sentence edges"""
        aligned_clips = []
        
        for clip in clips:
            aligned_start, aligned_end, start_word, end_word = aligner.align_to_sentences(
                clip.start, clip.end
            )
            
            clip.start = aligned_start
            clip.end = aligned_end
            clip.duration = aligned_end - aligned_start
            clip.start_word = start_word
            clip.end_word = end_word
            
            # Validate duration
            if self.min_duration <= clip.duration <= self.max_duration:
                aligned_clips.append(clip)
            elif self.debug:
                print(f"[DEBUG] Clip {clip.id} excluded: duration {clip.duration:.1f}s out of range")
        
        return aligned_clips
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Sei un esperto analista video. Rispondi sempre in formato TOON come richiesto."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM error: {e}")
            return ""


# ============================================================================
# Main Entry Point
# ============================================================================

def analyze_video(
    transcript_path: str,
    output_path: str = None,
    api_key: str = None,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    max_clips: int = 5,
    debug: bool = False
) -> dict:
    """
    Analyze video transcript and generate clip suggestions.
    
    Args:
        transcript_path: Path to Deepgram transcript JSON
        output_path: Optional path to save results
        api_key: OpenRouter API key
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        max_clips: Maximum number of clips to generate
        debug: Enable debug output
    
    Returns:
        Analysis results with scenes and clips
    """
    
    if api_key is None:
        api_key = os.environ.get('OPENROUTER_API_KEY', '')
    
    analyzer = EnhancedMultiAgentAnalyzer(
        api_key=api_key,
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
        max_clips=max_clips,
        debug=debug
    )
    
    results = analyzer.analyze(transcript_path)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if debug:
            print(f"[DEBUG] Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_analyzer.py <transcript.json> [output.json]")
        sys.exit(1)
    
    transcript = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = analyze_video(
        transcript,
        output,
        debug=True
    )
    
    print(f"\nAnalysis complete:")
    print(f"  Scenes detected: {results['metadata']['total_scenes']}")
    print(f"  Eligible scenes: {results['metadata']['eligible_scenes']}")
    print(f"  Final clips: {results['metadata']['final_clips']}")
    
    for clip in results['clips']:
        print(f"\n  Clip {clip['id']}: {clip['title']}")
        print(f"    Time: {clip['start']:.2f}s - {clip['end']:.2f}s ({clip['duration']:.1f}s)")
        print(f"    Words: '{clip['start_word']}' → '{clip['end_word']}'")
        print(f"    Viral score: {clip['viral_score']}/10")
