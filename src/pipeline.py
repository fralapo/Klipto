# pipeline.py
"""
AI YouTube Shorts Generator - Complete Pipeline v4.0

Features:
- Multi-agent video analysis with TOON v3.0
- Interactive dry-run preview
- Batch processing for multiple videos
- HTML/Markdown report generation
- Smart caching system
- Progress tracking
"""

import json
import os
import re
import hashlib
import shutil
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # API
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.0-flash-001"
    
    # Clip settings
    min_clip_duration: float = 15.0
    max_clip_duration: float = 60.0
    max_clips: int = 5
    
    # Directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    
    # Options
    use_cache: bool = True
    generate_report: bool = True
    dry_run: bool = False
    debug: bool = False
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get('OPENROUTER_API_KEY', '')
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        (self.data_dir / "transcripts").mkdir(exist_ok=True)
        (self.data_dir / "videos").mkdir(exist_ok=True)


# ============================================================================
# Console Utilities
# ============================================================================

class Console:
    """Terminal output utilities"""
    
    COLORS = {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'reset': '\033[0m'
    }
    
    _enabled = True
    
    @classmethod
    def disable_colors(cls):
        cls._enabled = False
        for key in cls.COLORS:
            cls.COLORS[key] = ''
    
    @classmethod
    def c(cls, color: str) -> str:
        return cls.COLORS.get(color, '') if cls._enabled else ''
    
    @classmethod
    def header(cls, text: str, width: int = 70):
        print(f"\n{cls.c('bold')}{cls.c('cyan')}{'═' * width}{cls.c('reset')}")
        print(f"{cls.c('bold')}{cls.c('cyan')}{text.center(width)}{cls.c('reset')}")
        print(f"{cls.c('bold')}{cls.c('cyan')}{'═' * width}{cls.c('reset')}\n")
    
    @classmethod
    def section(cls, text: str, emoji: str = ""):
        prefix = f"{emoji} " if emoji else ""
        print(f"\n{cls.c('bold')}{cls.c('yellow')}{prefix}{text}{cls.c('reset')}")
        print(f"{cls.c('dim')}{'─' * 70}{cls.c('reset')}")
    
    @classmethod
    def success(cls, text: str):
        print(f"{cls.c('green')}✓ {text}{cls.c('reset')}")
    
    @classmethod
    def error(cls, text: str):
        print(f"{cls.c('red')}✗ {text}{cls.c('reset')}")
    
    @classmethod
    def warning(cls, text: str):
        print(f"{cls.c('yellow')}⚠ {text}{cls.c('reset')}")
    
    @classmethod
    def info(cls, text: str):
        print(f"{cls.c('cyan')}ℹ {text}{cls.c('reset')}")
    
    @classmethod
    def progress(cls, current: int, total: int, text: str = ""):
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = '█' * filled + '░' * (bar_width - filled)
        percent = current / total * 100
        print(f"\r  [{bar}] {percent:5.1f}% {text}", end='', flush=True)
        if current >= total:
            print()


# ============================================================================
# TOON v3.0 Implementation
# ============================================================================

class ToonV3:
    """TOON v3.0 encoder/decoder"""
    
    @staticmethod
    def _needs_quoting(s: str) -> bool:
        if not s:
            return True
        if s in ('true', 'false', 'null'):
            return True
        if s[0] in ' \t"\'#[{' or s[-1] in ' \t':
            return True
        if '\n' in s or '\t' in s:
            return True
        try:
            float(s)
            return True
        except ValueError:
            pass
        return False
    
    @staticmethod
    def _quote(s: str) -> str:
        escaped = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        return f'"{escaped}"'
    
    @staticmethod
    def _is_tabular(arr: list) -> bool:
        if len(arr) < 1 or not all(isinstance(item, dict) for item in arr):
            return False
        keys = set(arr[0].keys())
        if not all(set(item.keys()) == keys for item in arr):
            return False
        for item in arr:
            for v in item.values():
                if isinstance(v, (dict, list)):
                    return False
        return True
    
    @staticmethod
    def encode(data) -> str:
        if isinstance(data, list) and ToonV3._is_tabular(data):
            keys = list(data[0].keys())
            lines = [f"[{len(data)} | {' '.join(keys)}]"]
            for item in data:
                values = []
                for k in keys:
                    v = item[k]
                    if v is None:
                        values.append('-')
                    elif isinstance(v, bool):
                        values.append('true' if v else 'false')
                    elif isinstance(v, str):
                        v = v.replace('\t', ' ').replace('\n', ' ')
                        values.append(ToonV3._quote(v) if ToonV3._needs_quoting(v) else v)
                    else:
                        values.append(str(v))
                lines.append('\t' + '\t'.join(values))
            return '\n'.join(lines)
        return json.dumps(data, ensure_ascii=False)
    
    @staticmethod
    def decode(toon_str: str):
        lines = toon_str.strip().split('\n')
        if not lines:
            return None
        
        # Tabular array
        if lines[0].startswith('[') and '|' in lines[0]:
            match = re.match(r'\[(\d+)\s*\|\s*(.+)\]', lines[0].strip())
            if match:
                count = int(match.group(1))
                fields = match.group(2).split()
                result = []
                for line in lines[1:count+1]:
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
        
        try:
            return json.loads(toon_str)
        except json.JSONDecodeError:
            return None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float = 1.0
    punctuated_word: str = ""

@dataclass
class Utterance:
    id: int
    speaker: int
    start: float
    end: float
    transcript: str
    words: list[Word] = field(default_factory=list)

@dataclass
class Scene:
    id: int
    start: float
    end: float
    start_word_idx: int
    end_word_idx: int
    speaker: int
    transcript: str
    scene_type: str = ""
    description: str = ""
    viral_score: int = 0

@dataclass
class Clip:
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
    output_file: str = ""

@dataclass
class VideoAnalysis:
    video_id: str
    video_title: str
    duration: float
    scenes: list[Scene]
    clips: list[Clip]
    transcript_path: str
    video_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """Intelligent caching for analysis results"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, transcript_path: str, config: PipelineConfig) -> str:
        """Generate cache key based on transcript content and config"""
        with open(transcript_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()).hexdigest()[:12]
        
        config_str = f"{config.min_clip_duration}_{config.max_clip_duration}_{config.max_clips}_{config.model}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{content_hash}_{config_hash}"
    
    def get(self, transcript_path: str, config: PipelineConfig) -> Optional[dict]:
        """Get cached analysis if exists"""
        cache_key = self._get_cache_key(transcript_path, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None
    
    def set(self, transcript_path: str, config: PipelineConfig, data: dict):
        """Save analysis to cache"""
        cache_key = self._get_cache_key(transcript_path, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear(self):
        """Clear all cache"""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)


# ============================================================================
# Scene Detection
# ============================================================================

class SceneDetector:
    """Local scene detection without LLM"""
    
    def __init__(self, min_pause: float = 1.5, min_duration: float = 5.0):
        self.min_pause = min_pause
        self.min_duration = min_duration
    
    def detect(self, utterances: list[Utterance], words: list[Word]) -> list[Scene]:
        if not utterances:
            return []
        
        scenes = []
        scene_start_time = utterances[0].start
        current_speaker = utterances[0].speaker
        current_text = []
        
        for i, utt in enumerate(utterances):
            gap = utt.start - utterances[i-1].end if i > 0 else 0
            is_new_scene = (utt.speaker != current_speaker) or (gap > self.min_pause)
            
            if is_new_scene and i > 0:
                scene_end = utterances[i-1].end
                if scene_end - scene_start_time >= self.min_duration:
                    start_idx, end_idx = self._find_word_indices(words, scene_start_time, scene_end)
                    scenes.append(Scene(
                        id=len(scenes),
                        start=scene_start_time,
                        end=scene_end,
                        start_word_idx=start_idx,
                        end_word_idx=end_idx,
                        speaker=current_speaker,
                        transcript=' '.join(current_text)
                    ))
                
                scene_start_time = utt.start
                current_speaker = utt.speaker
                current_text = []
            
            current_text.append(utt.transcript)
        
        # Final scene
        if utterances:
            last_end = utterances[-1].end
            if last_end - scene_start_time >= self.min_duration:
                start_idx, end_idx = self._find_word_indices(words, scene_start_time, last_end)
                scenes.append(Scene(
                    id=len(scenes),
                    start=scene_start_time,
                    end=last_end,
                    start_word_idx=start_idx,
                    end_word_idx=end_idx,
                    speaker=current_speaker,
                    transcript=' '.join(current_text)
                ))
        
        return scenes
    
    def _find_word_indices(self, words: list[Word], start: float, end: float) -> tuple[int, int]:
        start_idx = next((i for i, w in enumerate(words) if w.start >= start - 0.1), 0)
        end_idx = next((i for i in range(len(words)-1, -1, -1) if words[i].end <= end + 0.1), len(words)-1)
        return start_idx, end_idx


# ============================================================================
# Word Boundary Aligner
# ============================================================================

class WordBoundaryAligner:
    """Aligns clip boundaries to word/sentence edges"""
    
    def __init__(self, words: list[Word]):
        self.words = words
        self._build_sentences()
    
    def _build_sentences(self):
        self.sentences = []
        current_start = 0
        
        for i, word in enumerate(self.words):
            text = word.punctuated_word or word.word
            if text and text[-1] in '.!?':
                self.sentences.append({
                    'start_idx': current_start,
                    'end_idx': i,
                    'start': self.words[current_start].start,
                    'end': word.end
                })
                current_start = i + 1
        
        if current_start < len(self.words):
            self.sentences.append({
                'start_idx': current_start,
                'end_idx': len(self.words) - 1,
                'start': self.words[current_start].start,
                'end': self.words[-1].end
            })
    
    def align(self, start: float, end: float) -> tuple[float, float, str, str]:
        """Align to sentence boundaries"""
        # Find containing sentences
        start_sent = next((s for s in self.sentences if s['start'] <= start <= s['end']), None)
        end_sent = next((s for s in reversed(self.sentences) if s['start'] <= end <= s['end']), None)
        
        if not start_sent:
            start_sent = next((s for s in self.sentences if s['start'] > start), self.sentences[0] if self.sentences else None)
        if not end_sent:
            end_sent = next((s for s in reversed(self.sentences) if s['end'] < end), self.sentences[-1] if self.sentences else None)
        
        if start_sent and end_sent:
            return (
                start_sent['start'],
                end_sent['end'],
                self.words[start_sent['start_idx']].word,
                self.words[end_sent['end_idx']].word
            )
        
        # Fallback to word alignment
        start_idx = next((i for i, w in enumerate(self.words) if w.start >= start - 0.1), 0)
        end_idx = next((i for i in range(len(self.words)-1, -1, -1) if self.words[i].end <= end + 0.1), len(self.words)-1)
        
        return (
            self.words[start_idx].start,
            self.words[end_idx].end,
            self.words[start_idx].word,
            self.words[end_idx].word
        )


# ============================================================================
# Multi-Agent Analyzer
# ============================================================================

class MultiAgentAnalyzer:
    """4-phase multi-agent video analyzer"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.scene_detector = SceneDetector()
    
    def analyze(self, transcript_path: str, video_id: str = "") -> VideoAnalysis:
        """Run complete analysis pipeline"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        words = self._parse_words(data)
        utterances = self._parse_utterances(data)
        duration = words[-1].end if words else 0
        
        if self.config.debug:
            Console.info(f"Loaded {len(words)} words, {len(utterances)} utterances")
        
        # Phase 1: Scene detection
        scenes = self.scene_detector.detect(utterances, words)
        if self.config.debug:
            Console.info(f"Detected {len(scenes)} scenes")
        
        # Phase 2: Classification
        scenes = self._classify_scenes(scenes)
        
        # Filter eligible scenes
        excluded = {'intro', 'outro', 'teaser', 'transition'}
        eligible = [s for s in scenes if s.scene_type not in excluded]
        
        # Phase 3: Clip selection
        clips = self._select_clips(eligible)
        
        # Phase 4: Supervision
        clips = self._supervise_clips(clips, scenes)
        
        # Phase 5: Boundary alignment
        aligner = WordBoundaryAligner(words)
        clips = self._align_clips(clips, aligner)
        
        return VideoAnalysis(
            video_id=video_id or Path(transcript_path).stem.replace('_transcript', ''),
            video_title=data.get('metadata', {}).get('title', ''),
            duration=duration,
            scenes=scenes,
            clips=clips,
            transcript_path=transcript_path,
            video_path=""
        )
    
    def _parse_words(self, data: dict) -> list[Word]:
        word_data = data.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('words', [])
        # Fallback for flattened structure if word_data is empty
        if not word_data:
            word_data = data.get('words', [])
        
        return [Word(
            word=str(w.get('word', '')),
            start=float(w.get('start', 0)),
            end=float(w.get('end', 0)),
            confidence=float(w.get('confidence', 1.0)),
            punctuated_word=str(w.get('punctuated_word', w.get('word', '')))
        ) for w in word_data]
    
    def _parse_utterances(self, data: dict) -> list[Utterance]:
        utt_data = data.get('results', {}).get('utterances', [])
        # Fallback for flattened or missing utterances
        if not utt_data:
             utt_data = data.get('utterances', [])
             
        return [Utterance(
            id=i,
            speaker=int(u.get('speaker', 0)),
            start=float(u.get('start', 0)),
            end=float(u.get('end', 0)),
            transcript=str(u.get('transcript', '')),
            words=[Word(
                word=str(w.get('word', '')),
                start=float(w.get('start', 0)),
                end=float(w.get('end', 0)),
                punctuated_word=str(w.get('punctuated_word', ''))
            ) for w in u.get('words', [])]
        ) for i, u in enumerate(utt_data)]
    
    def _classify_scenes(self, scenes: list[Scene]) -> list[Scene]:
        if not scenes:
            return scenes
        
        toon_data = [{'id': s.id, 'start': round(s.start, 1), 'end': round(s.end, 1), 
                      'speaker': s.speaker, 'text': s.transcript[:150]} for s in scenes]
        
        prompt = f"""Classifica queste scene video.

SCENE:
{ToonV3.encode(toon_data)}

Rispondi in formato TOON tabulare con campi: id type viral_score description
- type: intro|outro|interview|monologue|highlight|transition|teaser
- viral_score: 1-10
- description: max 40 caratteri

Solo la tabella TOON, nient'altro."""

        response = self._call_llm(prompt)
        
        try:
            classifications = ToonV3.decode(response)
            if classifications:
                for cls in classifications:
                    scene = next((s for s in scenes if s.id == cls.get('id')), None)
                    if scene:
                        scene.scene_type = cls.get('type', 'unknown')
                        scene.viral_score = int(cls.get('viral_score', 5))
                        scene.description = cls.get('description', '')
        except Exception as e:
            if self.config.debug:
                Console.warning(f"Classification parse error: {e}")
        
        return scenes
    
    def _select_clips(self, scenes: list[Scene]) -> list[Clip]:
        if not scenes:
            return []
        
        toon_data = [{'id': s.id, 'start': round(s.start, 1), 'end': round(s.end, 1),
                      'type': s.scene_type, 'viral': s.viral_score, 'desc': s.description[:50]} for s in scenes]
        
        prompt = f"""Seleziona le migliori clip per Shorts/TikTok.

SCENE:
{ToonV3.encode(toon_data)}

Requisiti:
- Durata: {self.config.min_clip_duration}-{self.config.max_clip_duration}s
- Max {self.config.max_clips} clip
- Clip auto-contenute, non tagliare a metà

Rispondi in TOON tabulare: id scene_id start end title viral_score

Solo la tabella."""

        response = self._call_llm(prompt)
        
        clips = []
        try:
            clip_data = ToonV3.decode(response)
            if clip_data:
                for c in clip_data:
                    clips.append(Clip(
                        id=c.get('id', len(clips)),
                        scene_id=c.get('scene_id', 0),
                        start=float(c.get('start', 0)),
                        end=float(c.get('end', 0)),
                        duration=float(c.get('end', 0)) - float(c.get('start', 0)),
                        start_word='', end_word='',
                        title=c.get('title', ''),
                        description='',
                        viral_score=int(c.get('viral_score', 5))
                    ))
        except Exception as e:
            if self.config.debug:
                Console.warning(f"Clip selection parse error: {e}")
        
        return clips
    
    def _supervise_clips(self, clips: list[Clip], scenes: list[Scene]) -> list[Clip]:
        if not clips:
            return clips
        
        clips_toon = [{'id': c.id, 'scene_id': c.scene_id, 'start': round(c.start, 1),
                       'end': round(c.end, 1), 'duration': round(c.duration, 1)} for c in clips]
        scenes_toon = [{'id': s.id, 'start': round(s.start, 1), 'end': round(s.end, 1),
                        'type': s.scene_type} for s in scenes]
        
        prompt = f"""Valida queste clip.

CLIP:
{ToonV3.encode(clips_toon)}

SCENE:
{ToonV3.encode(scenes_toon)}

Verifica:
1. Clip dentro una singola scena
2. Durata {self.config.min_clip_duration}-{self.config.max_clip_duration}s
3. No intro/outro

Rispondi TOON: id approved adjusted_start adjusted_end issue

Solo tabella."""

        response = self._call_llm(prompt)
        
        approved = []
        try:
            validations = ToonV3.decode(response)
            if validations:
                for v in validations:
                    if v.get('approved', True):
                        clip = next((c for c in clips if c.id == v.get('id')), None)
                        if clip:
                            clip.start = float(v.get('adjusted_start', clip.start))
                            clip.end = float(v.get('adjusted_end', clip.end))
                            clip.duration = clip.end - clip.start
                            issue = v.get('issue', '')
                            if issue and issue != 'none':
                                clip.quality_notes.append(issue)
                            approved.append(clip)
        except Exception as e:
            if self.config.debug:
                Console.warning(f"Supervision parse error: {e}")
            approved = clips
        
        return approved
    
    def _align_clips(self, clips: list[Clip], aligner: WordBoundaryAligner) -> list[Clip]:
        aligned = []
        for clip in clips:
            start, end, start_word, end_word = aligner.align(clip.start, clip.end)
            clip.start = start
            clip.end = end
            clip.duration = end - start
            clip.start_word = start_word
            clip.end_word = end_word
            
            if self.config.min_clip_duration <= clip.duration <= self.config.max_clip_duration:
                aligned.append(clip)
        
        return aligned
    
    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Sei un esperto analista video. Rispondi sempre in formato TOON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.config.debug:
                Console.error(f"LLM error: {e}")
            return ""


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generate HTML and Markdown reports"""
    
    @staticmethod
    def generate_html(analysis: VideoAnalysis, output_path: Path):
        """Generate HTML report"""
        html = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Report - {analysis.video_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               background: #0f0f0f; color: #fff; padding: 2rem; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #ff0050; margin-bottom: 0.5rem; }}
        h2 {{ color: #00f2ea; margin: 2rem 0 1rem; border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        .meta {{ color: #888; margin-bottom: 2rem; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .stat {{ background: #1a1a1a; padding: 1rem; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #00f2ea; }}
        .stat-label {{ color: #888; font-size: 0.85rem; }}
        .scene, .clip {{ background: #1a1a1a; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
        .scene-header, .clip-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
        .badge-intro {{ background: #333; }}
        .badge-outro {{ background: #333; }}
        .badge-interview {{ background: #00f2ea; color: #000; }}
        .badge-highlight {{ background: #ff0050; }}
        .badge-monologue {{ background: #7c3aed; }}
        .score {{ display: flex; align-items: center; gap: 0.5rem; }}
        .score-bar {{ width: 100px; height: 8px; background: #333; border-radius: 4px; overflow: hidden; }}
        .score-fill {{ height: 100%; background: linear-gradient(90deg, #00f2ea, #ff0050); }}
        .transcript {{ background: #0a0a0a; padding: 1rem; border-radius: 4px; font-size: 0.9rem; color: #ccc; 
                       max-height: 100px; overflow-y: auto; margin-top: 1rem; }}
        .time {{ color: #888; font-family: monospace; }}
        .clip-title {{ font-size: 1.25rem; font-weight: 600; }}
        .words {{ color: #00f2ea; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #888; font-weight: 500; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #333; color: #666; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Video Analysis Report</h1>
        <p class="meta">Video ID: {analysis.video_id} | Generated: {analysis.created_at[:19]}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(analysis.scenes)}</div>
                <div class="stat-label">Scene rilevate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(analysis.clips)}</div>
                <div class="stat-label">Clip generate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{int(analysis.duration // 60)}:{int(analysis.duration % 60):02d}</div>
                <div class="stat-label">Durata video</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(c.duration for c in analysis.clips):.0f}s</div>
                <div class="stat-label">Contenuto clip</div>
            </div>
        </div>
        
        <h2>🎬 Clip Generate</h2>
        {"".join(ReportGenerator._clip_html(c) for c in analysis.clips) or "<p>Nessuna clip generata.</p>"}
        
        <h2>📋 Tutte le Scene</h2>
        <table>
            <tr><th>#</th><th>Tipo</th><th>Tempo</th><th>Durata</th><th>Score</th><th>Descrizione</th></tr>
            {"".join(ReportGenerator._scene_row(s) for s in analysis.scenes)}
        </table>
        
        <div class="footer">
            Generated by AI YouTube Shorts Generator v4.0
        </div>
    </div>
</body>
</html>"""
        
        output_path.write_text(html, encoding='utf-8')
    
    @staticmethod
    def _clip_html(clip: Clip) -> str:
        return f"""
        <div class="clip">
            <div class="clip-header">
                <span class="clip-title">{clip.title}</span>
                <div class="score">
                    <span>{clip.viral_score}/10</span>
                    <div class="score-bar"><div class="score-fill" style="width: {clip.viral_score * 10}%"></div></div>
                </div>
            </div>
            <p><span class="time">{int(clip.start // 60)}:{clip.start % 60:05.2f} → {int(clip.end // 60)}:{clip.end % 60:05.2f}</span> ({clip.duration:.1f}s)</p>
            <p class="words">"{clip.start_word}" → "{clip.end_word}"</p>
        </div>"""
    
    @staticmethod
    def _scene_row(scene: Scene) -> str:
        duration = scene.end - scene.start
        time_str = f"{int(scene.start // 60)}:{int(scene.start % 60):02d}"
        return f"""<tr>
            <td>{scene.id}</td>
            <td><span class="badge badge-{scene.scene_type}">{scene.scene_type}</span></td>
            <td class="time">{time_str}</td>
            <td>{duration:.0f}s</td>
            <td>{scene.viral_score}/10</td>
            <td>{scene.description}</td>
        </tr>"""
    
    @staticmethod
    def generate_markdown(analysis: VideoAnalysis, output_path: Path):
        """Generate Markdown report"""
        clips_md = "\n".join([
            f"### Clip {c.id + 1}: {c.title}\n"
            f"- **Tempo:** {c.start:.2f}s → {c.end:.2f}s ({c.duration:.1f}s)\n"
            f"- **Parole:** \"{c.start_word}\" → \"{c.end_word}\"\n"
            f"- **Viral Score:** {'⭐' * c.viral_score}{'☆' * (10 - c.viral_score)} {c.viral_score}/10\n"
            for c in analysis.clips
        ]) or "Nessuna clip generata."
        
        scenes_md = "| # | Tipo | Tempo | Durata | Score | Descrizione |\n|---|------|-------|--------|-------|-------------|\n"
        scenes_md += "\n".join([
            f"| {s.id} | {s.scene_type} | {int(s.start // 60)}:{int(s.start % 60):02d} | {s.end - s.start:.0f}s | {s.viral_score}/10 | {s.description} |"
            for s in analysis.scenes
        ])
        
        md = f"""# 📊 Video Analysis Report

**Video ID:** {analysis.video_id}  
**Generated:** {analysis.created_at[:19]}  
**Duration:** {int(analysis.duration // 60)}:{int(analysis.duration % 60):02d}

---

## 📈 Summary

| Metric | Value |
|--------|-------|
| Scene rilevate | {len(analysis.scenes)} |
| Clip generate | {len(analysis.clips)} |
| Durata contenuto | {sum(c.duration for c in analysis.clips):.0f}s |

---

## 🎬 Clip Generate

{clips_md}

---

## 📋 Tutte le Scene

{scenes_md}

---

*Generated by AI YouTube Shorts Generator v4.0*
"""
        output_path.write_text(md, encoding='utf-8')


# ============================================================================
# Dry Run Interface
# ============================================================================

class DryRunInterface:
    """Interactive dry-run preview"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def run(self, analysis: VideoAnalysis, words: list[Word]) -> Optional[list[Clip]]:
        """Run interactive dry-run"""
        Console.header("ANALISI VIDEO - DRY RUN")
        
        # Display scenes
        Console.section("SCENE RILEVATE", "📊")
        self._print_scenes_table(analysis.scenes)
        
        # Display clips
        Console.section(f"CLIP PROPOSTE ({len(analysis.clips)})", "🎬")
        for i, clip in enumerate(analysis.clips):
            self._print_clip(clip, i)
        
        # Summary
        self._print_summary(analysis)
        
        # Prompt
        action = self._prompt_action()
        
        if action == 's':
            Console.success("Procedo con il taglio...")
            return analysis.clips
        elif action == 'm':
            return self._edit_mode(analysis, words)
        else:
            Console.warning("Operazione annullata.")
            return None
    
    def _print_scenes_table(self, scenes: list[Scene]):
        excluded = {'intro', 'outro', 'teaser'}
        print(f" {'#':>2}  │ {'Tipo':<11} │ {'Tempo':<13} │ {'Durata':>7} │ {'Score':>5} │ Descrizione")
        print(f"{'─' * 4}┼{'─' * 13}┼{'─' * 15}┼{'─' * 9}┼{'─' * 7}┼{'─' * 20}")
        
        for s in scenes:
            time_str = f"{int(s.start // 60)}:{int(s.start % 60):02d}-{int(s.end // 60)}:{int(s.end % 60):02d}"
            duration = f"{s.end - s.start:.0f}s"
            is_excluded = s.scene_type in excluded
            color = Console.c('dim') if is_excluded else (Console.c('green') if s.viral_score >= 8 else '')
            emoji = {'intro': '🎬', 'outro': '👋', 'interview': '🎤', 'highlight': '✨', 
                     'monologue': '🗣️', 'transition': '🔄', 'teaser': '👀'}.get(s.scene_type, '❓')
            print(f"{color} {s.id:>2}  │ {emoji} {s.scene_type:<8} │ {time_str:<13} │ {duration:>7} │ {s.viral_score:>2}/10 │ {s.description[:20]}{Console.c('reset')}")
    
    def _print_clip(self, clip: Clip, index: int):
        print(f"\n {Console.c('bold')}CLIP {index + 1}{Console.c('reset')} {'─' * 60}")
        print(f" │ Titolo:  {clip.title}")
        print(f" │ Tempo:   {int(clip.start // 60)}:{clip.start % 60:05.2f} → {int(clip.end // 60)}:{clip.end % 60:05.2f}  ({clip.duration:.1f}s)")
        print(f" │ Parole:  \"{clip.start_word}\" → \"{clip.end_word}\"")
        print(f" │ Viral:   {'⭐' * clip.viral_score}{'☆' * (10 - clip.viral_score)} {clip.viral_score}/10")
        print(f" └{'─' * 65}")
    
    def _print_summary(self, analysis: VideoAnalysis):
        total_duration = sum(c.duration for c in analysis.clips)
        eligible = len([s for s in analysis.scenes if s.scene_type not in {'intro', 'outro', 'teaser'}])
        
        print(f"\n{'═' * 70}")
        print(f" Scene: {len(analysis.scenes)} totali, {eligible} eligibili")
        print(f" Clip:  {len(analysis.clips)} generate, {total_duration:.0f}s contenuto")
        print(f"{'═' * 70}")
    
    def _prompt_action(self) -> str:
        print(f"\n{Console.c('bold')}Vuoi procedere?{Console.c('reset')}")
        print(f"  {Console.c('green')}s{Console.c('reset')} = Taglia  |  {Console.c('yellow')}m{Console.c('reset')} = Modifica  |  {Console.c('red')}n{Console.c('reset')} = Annulla")
        try:
            return input(f"\n{Console.c('cyan')}>{Console.c('reset')} ").strip().lower() or 'n'
        except (KeyboardInterrupt, EOFError):
            return 'n'
    
    def _edit_mode(self, analysis: VideoAnalysis, words: list[Word]) -> Optional[list[Clip]]:
        clips = list(analysis.clips)
        aligner = WordBoundaryAligner(words)
        
        print(f"\n{Console.c('bold')}Modalità modifica{Console.c('reset')}")
        print("Comandi: lista | anteprima N | escludi N | aggiungi N | procedi | annulla")
        
        while True:
            try:
                cmd = input(f"\n{Console.c('cyan')}>{Console.c('reset')} ").strip().lower().split()
            except (KeyboardInterrupt, EOFError):
                return None
            
            if not cmd:
                continue
            
            action = cmd[0]
            
            if action in ('procedi', 'p', 'ok'):
                return clips
            elif action in ('annulla', 'q'):
                return None
            elif action in ('lista', 'l'):
                for i, c in enumerate(clips):
                    print(f"  {i+1}. {c.title} ({c.duration:.1f}s)")
            elif action in ('escludi', 'e', 'r') and len(cmd) > 1:
                try:
                    idx = int(cmd[1]) - 1
                    if 0 <= idx < len(clips):
                        removed = clips.pop(idx)
                        Console.success(f"Rimossa: {removed.title}")
                except (ValueError, IndexError):
                    Console.error("Indice non valido")
            elif action in ('anteprima', 'a') and len(cmd) > 1:
                try:
                    idx = int(cmd[1]) - 1
                    if 0 <= idx < len(clips):
                        self._print_clip(clips[idx], idx)
                except (ValueError, IndexError):
                    Console.error("Indice non valido")
            elif action in ('aggiungi', 'add') and len(cmd) > 1:
                try:
                    scene_id = int(cmd[1])
                    scene = next((s for s in analysis.scenes if s.id == scene_id), None)
                    if scene:
                        title = input("  Titolo: ").strip() or f"Clip da scena {scene_id}"
                        start, end, sw, ew = aligner.align(scene.start, scene.end)
                        clips.append(Clip(
                            id=len(clips), scene_id=scene_id,
                            start=start, end=end, duration=end-start,
                            start_word=sw, end_word=ew,
                            title=title, description='', viral_score=scene.viral_score
                        ))
                        Console.success(f"Aggiunta: {title}")
                except (ValueError, StopIteration):
                    Console.error("Scena non trovata")
            else:
                Console.warning("Comando non riconosciuto")
        
        return clips


# ============================================================================
# Video Clipper (Hardware Accelerated)
# ============================================================================

class VideoClipper:
    """Cut video clips using FFmpeg with Hardware Acceleration detection"""
    
    _settings_cache = None
    
    @classmethod
    def _check_encoder(cls, encoder_name: str) -> bool:
        """Check if specific encoder is available"""
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return encoder_name in result.stdout
        except FileNotFoundError:
            return False

    @classmethod
    def _get_settings(cls) -> dict:
        """Determine best encoding settings"""
        if cls._settings_cache:
            return cls._settings_cache
            
        system = platform.system()
        settings = {
            "hwaccel": "",
            "codec": "-c:v libx264",
            "quality": "-crf 23 -preset medium",
            "audio": "-c:a copy"
        }
        
        # 1. NVIDIA (Windows/Linux)
        if cls._check_encoder("h264_nvenc"):
            Console.info(f"Hardware Acceleration: NVIDIA NVENC detected ({system})")
            settings.update({
                "hwaccel": "-hwaccel cuda",
                "codec": "-c:v h264_nvenc",
                "quality": "-rc constqp -qp 23 -preset p4",
                "audio": "-c:a copy"
            })
            
        # 2. Apple Silicon / Intel Mac (macOS)
        elif system == "Darwin" and cls._check_encoder("h264_videotoolbox"):
            Console.info("Hardware Acceleration: Apple VideoToolbox detected")
            settings.update({
                "hwaccel": "-hwaccel videotoolbox",
                "codec": "-c:v h264_videotoolbox",
                "quality": "-q:v 60", # ~CRF 23 for VideoToolbox
                "audio": "-c:a copy"
            })
            
        else:
            Console.info("No Hardware Acceleration detected. Using CPU (x264).")
            
        cls._settings_cache = settings
        return settings
    
    @classmethod
    def cut(cls, video_path: str, clip: Clip, output_path: str, vertical: bool = True) -> bool:
        """Cut a single clip from video"""
        settings = cls._get_settings()
        duration = clip.end - clip.start
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-vsync', '0']
        
        # HW Accel input options
        if settings['hwaccel']:
            cmd.extend(settings['hwaccel'].split())
            
        cmd.extend(['-ss', str(clip.start)])
        cmd.extend(['-i', video_path])
        cmd.extend(['-t', str(duration)])
        
        # Codec & Quality
        cmd.extend(settings['codec'].split())
        cmd.extend(settings['quality'].split())
        cmd.extend(settings['audio'].split())
        
        if vertical:
            # Cinematic Crop: Zoom 15% -> Center Crop -> Pad to 9:16
            # Note: Software filters are used for consistency across platforms
            vf_chain = "scale=1920*1.15:1080*1.15,crop=1080:ih,pad=1080:1920:(ow-iw)/2:(1920-ih)/2:black"
            cmd.extend(['-filter_complex', f"[0:v]{vf_chain}[v]", '-map', '[v]', '-map', '0:a'])
        else:
            # Horizontal (Pass-through map)
            cmd.extend(['-map', '0:v', '-map', '0:a'])
            
        cmd.extend(['-movflags', '+faststart', output_path])
        
        try:
            # Suppress excessive ffmpeg output unless error
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                Console.error(f"FFmpeg Error: {result.stderr[:200]}")
            return result.returncode == 0
        except Exception as e:
            Console.error(f"Execution Error: {e}")
            return False


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """Process multiple videos"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = CacheManager(config.cache_dir)
        self.analyzer = MultiAgentAnalyzer(config)
    
    def process_batch(self, video_sources: list[str]) -> list[VideoAnalysis]:
        """Process multiple videos"""
        Console.header("BATCH PROCESSING")
        Console.info(f"Processing {len(video_sources)} videos...")
        
        results = []
        
        for i, source in enumerate(video_sources):
            Console.section(f"Video {i+1}/{len(video_sources)}", "🎬")
            
            try:
                # Determine paths
                video_id = self._extract_video_id(source)
                transcript_path = self.config.data_dir / "transcripts" / f"{video_id}_transcript.json"
                
                if not transcript_path.exists():
                    Console.warning(f"Transcript not found: {transcript_path}")
                    continue
                
                # Check cache
                if self.config.use_cache:
                    cached = self.cache.get(str(transcript_path), self.config)
                    if cached:
                        Console.success("Using cached analysis")
                        analysis = self._dict_to_analysis(cached)
                        results.append(analysis)
                        continue
                
                # Analyze
                Console.info("Analyzing...")
                analysis = self.analyzer.analyze(str(transcript_path), video_id)
                Console.success(f"Found {len(analysis.clips)} clips")
                
                # Cache
                if self.config.use_cache:
                    self.cache.set(str(transcript_path), self.config, asdict(analysis))
                
                results.append(analysis)
                
            except Exception as e:
                Console.error(f"Error processing {source}: {e}")
        
        # Summary
        Console.header("BATCH COMPLETE")
        total_clips = sum(len(a.clips) for a in results)
        Console.info(f"Processed: {len(results)}/{len(video_sources)} videos")
        Console.info(f"Total clips: {total_clips}")
        
        return results
    
    def _extract_video_id(self, source: str) -> str:
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11})',
            r'([0-9A-Za-z_-]{11})_transcript\.json',
        ]
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return match.group(1)
        return Path(source).stem.replace('_transcript', '')
    
    def _dict_to_analysis(self, data: dict) -> VideoAnalysis:
        return VideoAnalysis(
            video_id=data.get('video_id', ''),
            video_title=data.get('video_title', ''),
            duration=data.get('duration', 0),
            scenes=[Scene(**s) for s in data.get('scenes', [])],
            clips=[Clip(**c) for c in data.get('clips', [])],
            transcript_path=data.get('transcript_path', ''),
            video_path=data.get('video_path', ''),
            created_at=data.get('created_at', '')
        )


# ============================================================================
# Main Pipeline
# ============================================================================

class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.cache = CacheManager(self.config.cache_dir)
        self.analyzer = MultiAgentAnalyzer(self.config)
    
    def run(self, transcript_path: str, video_path: str = None) -> Optional[VideoAnalysis]:
        """Run complete pipeline for single video"""
        transcript_path = Path(transcript_path)
        
        if not transcript_path.exists():
            Console.error(f"Transcript not found: {transcript_path}")
            return None
        
        video_id = transcript_path.stem.replace('_transcript', '')
        
        # Check cache
        if self.config.use_cache:
            cached = self.cache.get(str(transcript_path), self.config)
            if cached:
                Console.info("Using cached analysis")
                analysis = self._dict_to_analysis(cached)
            else:
                analysis = None
        else:
            analysis = None
        
        # Analyze if needed
        if not analysis:
            Console.info("Analyzing video...")
            analysis = self.analyzer.analyze(str(transcript_path), video_id)
            
            if self.config.use_cache:
                self.cache.set(str(transcript_path), self.config, asdict(analysis))
        
        # Load words for dry-run
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = self.analyzer._parse_words(data)
        
        # Dry-run interface
        if self.config.dry_run or not video_path:
            interface = DryRunInterface(self.config)
            approved_clips = interface.run(analysis, words)
            
            if approved_clips is None:
                return None
            
            analysis.clips = approved_clips
        
        # Set video path
        if video_path:
            analysis.video_path = video_path
        else:
            # Try to find video
            possible_paths = [
                self.config.data_dir / "videos" / f"{video_id}.mp4",
                self.config.data_dir / "videos" / f"{video_id}.webm",
            ]
            for p in possible_paths:
                if p.exists():
                    analysis.video_path = str(p)
                    break
        
        # Cut clips if video available
        if analysis.video_path and Path(analysis.video_path).exists() and analysis.clips:
            Console.section("Taglio clip", "✂️")
            
            output_dir = self.config.output_dir / video_id
            output_dir.mkdir(exist_ok=True)
            
            for i, clip in enumerate(analysis.clips):
                output_file = output_dir / f"clip_{i+1:02d}.mp4"
                Console.info(f"Clip {i+1}: {clip.title}")
                
                success = VideoClipper.cut(analysis.video_path, clip, str(output_file))
                
                if success:
                    clip.output_file = str(output_file)
                    Console.success(f"Saved: {output_file.name}")
                else:
                    Console.error(f"Failed to cut clip {i+1}")
        
        # Generate reports
        if self.config.generate_report:
            report_dir = self.config.output_dir / video_id
            report_dir.mkdir(exist_ok=True)
            
            ReportGenerator.generate_html(analysis, report_dir / "report.html")
            ReportGenerator.generate_markdown(analysis, report_dir / "report.md")
            Console.success(f"Reports saved to {report_dir}")
        
        return analysis
    
    def _dict_to_analysis(self, data: dict) -> VideoAnalysis:
        scenes = [Scene(**{k: v for k, v in s.items() if k != 'quality_notes'}) 
                  for s in data.get('scenes', [])]
        clips = [Clip(**c) for c in data.get('clips', [])]
        
        return VideoAnalysis(
            video_id=data.get('video_id', ''),
            video_title=data.get('video_title', ''),
            duration=data.get('duration', 0),
            scenes=scenes,
            clips=clips,
            transcript_path=data.get('transcript_path', ''),
            video_path=data.get('video_path', ''),
            created_at=data.get('created_at', '')
        )


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AI YouTube Shorts Generator v4.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py -t transcript.json
  python pipeline.py -t transcript.json -v video.mp4
  python pipeline.py -t transcript.json --dry-run
  python pipeline.py --batch transcripts/*.json
  python pipeline.py --clear-cache
        """
    )
    
    # Input
    parser.add_argument('-t', '--transcript', type=str, help='Transcript JSON file')
    parser.add_argument('-v', '--video', type=str, help='Video file')
    parser.add_argument('--batch', nargs='+', help='Batch process multiple transcripts')
    
    # Settings
    parser.add_argument('--min-duration', type=float, default=15.0, help='Min clip duration (default: 15)')
    parser.add_argument('--max-duration', type=float, default=60.0, help='Max clip duration (default: 60)')
    parser.add_argument('--max-clips', type=int, default=5, help='Max clips per video (default: 5)')
    parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001', help='LLM model')
    
    # Options
    parser.add_argument('--dry-run', '-d', action='store_true', help='Preview without cutting')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache and exit')
    parser.add_argument('--debug', action='store_true', help='Debug output')
    
    # Output
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check colors
    if not os.isatty(1):
        Console.disable_colors()
    
    # Build config
    config = PipelineConfig(
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        max_clips=args.max_clips,
        model=args.model,
        output_dir=Path(args.output),
        use_cache=not args.no_cache,
        generate_report=not args.no_report,
        dry_run=args.dry_run,
        debug=args.debug
    )
    
    # Clear cache
    if args.clear_cache:
        cache = CacheManager(config.cache_dir)
        cache.clear()
        Console.success("Cache cleared")
        return 0
    
    # Batch processing
    if args.batch:
        processor = BatchProcessor(config)
        results = processor.process_batch(args.batch)
        return 0 if results else 1
    
    # Single video
    if args.transcript:
        pipeline = Pipeline(config)
        result = pipeline.run(args.transcript, args.video)
        
        if result:
            Console.header("COMPLETE")
            Console.success(f"Generated {len(result.clips)} clips")
            return 0
        return 1
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())
