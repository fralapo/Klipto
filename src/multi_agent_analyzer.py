"""
Multi-Agent Clip Analyzer
Implements a 3-phase pipeline for accurate clip extraction.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    ANALYSIS_DIR,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    TARGET_CLIP_DURATION,
    DEFAULT_LLM_MODEL,
)
from transcriber import format_timestamp
import toon


@dataclass
class Scene:
    """A logical scene/segment in the video."""
    id: int
    start_time: float
    end_time: float
    speaker_ids: List[int]
    summary: str
    scene_type: str  # interview, monologue, transition, intro, outro, etc.
    utterance_indices: List[int] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ScoredScene(Scene):
    """Scene with virality scoring."""
    virality_score: int = 0
    hook_text: str = ""
    emotional_triggers: List[str] = field(default_factory=list)
    why_viral: str = ""
    is_viable: bool = True  # False if too short, outro, etc.


@dataclass 
class RefinedClip:
    """Final clip with word-aligned boundaries."""
    scene_id: int
    start_time: float
    end_time: float
    refined_start: float
    refined_end: float
    first_word_idx: int
    last_word_idx: int
    first_words: str
    last_words: str
    hook_text: str
    virality_score: int
    why_viral: str
    duration: float
    
    def to_dict(self) -> dict:
        return {
            "clip_number": self.scene_id,
            "start_time": self.refined_start,
            "end_time": self.refined_end,
            "duration": self.duration,
            "first_words": self.first_words,
            "last_words": self.last_words,
            "hook_text": self.hook_text,
            "virality_score": self.virality_score,
            "why_viral": self.why_viral,
            "word_alignment": {
                "first_word_idx": self.first_word_idx,
                "last_word_idx": self.last_word_idx,
            }
        }


class MultiAgentAnalyzer:
    """
    Multi-agent system for precise clip extraction.
    
    Phase 1 (Scene Detector): Segments video into logical scenes
    Phase 2 (Virality Scorer): Scores each scene for viral potential  
    Phase 3 (Clip Refiner): Aligns boundaries to word-level timestamps
    """
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
    
    # Different models for different tasks (can be customized)
    DEFAULT_MODELS = {
        "segmenter": DEFAULT_LLM_MODEL,      # Good at structure
        "scorer": DEFAULT_LLM_MODEL,          # Good at analysis
        "refiner": DEFAULT_LLM_MODEL,         # Good at precision
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[Dict[str, str]] = None,
        app_name: str = "Klipto",
        app_url: str = "https://github.com/klipto",
        debug: bool = False
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.models = models or self.DEFAULT_MODELS.copy()
        self.app_name = app_name
        self.app_url = app_url
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        self.client = OpenAI(
            base_url=self.OPENROUTER_API_URL,
            api_key=self.api_key,
        )
        
        # Track costs
        self.total_cost = 0.0
        self.phase_costs = {}
    
    def analyze(
        self,
        transcript: dict,
        video_title: str,
        max_clips: int = 5,
        language: str = "it",
        target_platform: str = "all",
        min_scene_duration: float = 10.0,
        exclude_intro_outro: bool = True,
    ) -> dict:
        """
        Run full multi-agent analysis pipeline.
        
        Args:
            transcript: Deepgram transcript with words and utterances
            video_title: Video title for context
            max_clips: Maximum clips to extract
            language: Output language
            target_platform: Target platform
            min_scene_duration: Minimum scene duration to consider
            exclude_intro_outro: Auto-exclude intro/outro segments
        
        Returns:
            Analysis result with refined clips
        """
        print(f"\n{'='*70}")
        print("🤖 MULTI-AGENT ANALYSIS PIPELINE")
        print(f"{'='*70}")
        
        words = transcript.get("words", [])
        utterances = transcript.get("utterances", [])
        duration = transcript.get("duration", 0)
        
        if not utterances:
            raise ValueError("Transcript must have utterances for scene detection")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Scene Segmentation
        # ═══════════════════════════════════════════════════════════════
        print(f"\n📍 PHASE 1: Scene Segmentation")
        print(f"   Utterances: {len(utterances)}, Duration: {format_timestamp(duration)}")
        
        scenes = self._phase1_segment_scenes(
            utterances=utterances,
            video_title=video_title,
            total_duration=duration,
            language=language,
            min_duration=min_scene_duration,
        )
        
        print(f"   ✓ Found {len(scenes)} scenes")
        for s in scenes:
            print(f"     [{s.id}] {format_timestamp(s.start_time)}-{format_timestamp(s.end_time)} "
                  f"({s.duration:.1f}s) [{s.scene_type}] {s.summary[:40]}...")
        
        # Filter out intro/outro if requested
        if exclude_intro_outro:
            viable_scenes = [
                s for s in scenes 
                if s.scene_type not in ("intro", "outro", "transition", "credits")
                and s.duration >= min_scene_duration
            ]
            print(f"   ✓ {len(viable_scenes)} viable scenes after filtering")
        else:
            viable_scenes = [s for s in scenes if s.duration >= min_scene_duration]
        
        if not viable_scenes:
            return {
                "error": "No viable scenes found",
                "scenes_detected": len(scenes),
                "clips": []
            }
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Virality Scoring
        # ═══════════════════════════════════════════════════════════════
        print(f"\n📍 PHASE 2: Virality Scoring")
        
        scored_scenes = self._phase2_score_scenes(
            scenes=viable_scenes,
            utterances=utterances,
            video_title=video_title,
            language=language,
            target_platform=target_platform,
        )
        
        # Sort by score and take top N
        scored_scenes.sort(key=lambda x: x.virality_score, reverse=True)
        top_scenes = scored_scenes[:max_clips]
        
        print(f"   ✓ Top {len(top_scenes)} scenes by virality:")
        for s in top_scenes:
            print(f"     [{s.id}] Score: {s.virality_score}/10 - {s.hook_text[:50]}...")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: Boundary Refinement
        # ═══════════════════════════════════════════════════════════════
        print(f"\n📍 PHASE 3: Boundary Refinement")
        
        refined_clips = self._phase3_refine_boundaries(
            scenes=top_scenes,
            words=words,
            utterances=utterances,
            language=language,
        )
        
        print(f"   ✓ Refined {len(refined_clips)} clips with word-aligned boundaries")
        
        # ═══════════════════════════════════════════════════════════════
        # Build final result
        # ═══════════════════════════════════════════════════════════════
        
        result = {
            "video_title": video_title,
            "video_duration": duration,
            "target_platform": target_platform,
            "pipeline": "multi-agent-v1",
            "phases": {
                "segmentation": {
                    "scenes_found": len(scenes),
                    "viable_scenes": len(viable_scenes),
                },
                "scoring": {
                    "scenes_scored": len(scored_scenes),
                },
                "refinement": {
                    "clips_refined": len(refined_clips),
                }
            },
            "clips_found": len(refined_clips),
            "clips": [c.to_dict() for c in refined_clips],
            "models_used": self.models,
            "language": language,
            "costs": self.phase_costs,
            "total_cost": self.total_cost,
        }
        
        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"   Clips: {len(refined_clips)}")
        print(f"   Total Cost: ${self.total_cost:.6f}")
        print(f"{'='*70}")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Scene Segmentation
    # ═══════════════════════════════════════════════════════════════════════
    
    def _phase1_segment_scenes(
        self,
        utterances: List[dict],
        video_title: str,
        total_duration: float,
        language: str,
        min_duration: float,
    ) -> List[Scene]:
        """
        Segment video into logical scenes based on:
        - Speaker changes
        - Topic shifts
        - Temporal gaps
        - Conversation boundaries
        """
        
        # Prepare utterance data for LLM
        utt_data = []
        for i, u in enumerate(utterances):
            utt_data.append({
                "idx": i,
                "start": round(u["start"], 2),
                "end": round(u["end"], 2),
                "speaker": u.get("speaker", 0),
                "text": u["text"][:200],  # Truncate long utterances
            })
        
        system_prompt = self._get_segmenter_prompt(language)
        
        user_prompt = f"""Analyze this video transcript and identify distinct SCENES.

VIDEO: {video_title}
DURATION: {format_timestamp(total_duration)} ({total_duration:.1f}s)
MIN SCENE DURATION: {min_duration}s

UTTERANCES (idx, start, end, speaker, text):
```
{self._format_utterances_compact(utt_data)}
```

Identify logical scene boundaries. Each scene should be:
- A complete conversation/interview segment
- A distinct topic or activity
- Minimum {min_duration} seconds long

Output as TOON:
```toon
scenes[N]{{id,start_idx,end_idx,start_time,end_time,scene_type,summary}}:
  1,0,5,0.0,45.2,interview,"First person interview about cosplay"
  2,6,12,46.0,120.5,interview,"Second person interview about manga"
  ...
```

scene_type must be one of: intro, interview, monologue, discussion, transition, outro, credits, other
"""
        
        response = self._call_llm(
            model=self.models["segmenter"],
            system=system_prompt,
            user=user_prompt,
            phase="segmentation"
        )
        
        scenes = self._parse_scenes_response(response, utterances)
        return scenes
    
    def _get_segmenter_prompt(self, language: str) -> str:
        if language == "it":
            return """Sei un esperto di analisi video. Il tuo compito è segmentare una trascrizione in SCENE LOGICHE distinte.

REGOLE DI SEGMENTAZIONE:
1. Ogni INTERVISTA è una scena separata (anche se breve)
2. INTRO e OUTRO sono scene separate (spesso contengono saluti, CTA, musica)
3. I CAMBI DI SPEAKER spesso indicano nuove scene
4. Le PAUSE LUNGHE (>3s) possono indicare transizioni
5. I CAMBI DI ARGOMENTO definiscono nuove scene

TIPI DI SCENA:
- intro: Presentazione iniziale, saluti
- interview: Domande/risposte con altra persona
- monologue: Speaker singolo che parla alla camera
- discussion: Conversazione tra più persone
- transition: Brevi momenti tra scene
- outro: Conclusione, CTA, saluti finali
- credits: Crediti, ringraziamenti
- other: Altro

Rispondi SOLO con il blocco TOON richiesto."""
        else:
            return """You are a video analysis expert. Your task is to segment a transcript into distinct LOGICAL SCENES.

SEGMENTATION RULES:
1. Each INTERVIEW is a separate scene (even if brief)
2. INTRO and OUTRO are separate scenes (often contain greetings, CTA, music)
3. SPEAKER CHANGES often indicate new scenes
4. LONG PAUSES (>3s) can indicate transitions
5. TOPIC CHANGES define new scenes

SCENE TYPES:
- intro: Opening presentation, greetings
- interview: Q&A with another person
- monologue: Single speaker talking to camera
- discussion: Conversation between multiple people
- transition: Brief moments between scenes
- outro: Conclusion, CTA, final greetings
- credits: Credits, thanks
- other: Other

Respond ONLY with the requested TOON block."""
    
    def _format_utterances_compact(self, utt_data: List[dict]) -> str:
        """Format utterances compactly for LLM."""
        lines = []
        for u in utt_data:
            lines.append(f"{u['idx']}\t{u['start']}\t{u['end']}\tS{u['speaker']}\t{u['text']}")
        return "\n".join(lines)
    
    def _parse_scenes_response(self, response: str, utterances: List[dict]) -> List[Scene]:
        """Parse scene segmentation response."""
        scenes = []
        
        try:
            # Try TOON first
            raw_scenes = toon.decode(response, strict=False)
            
            for item in raw_scenes:
                scene = Scene(
                    id=int(item.get("id", len(scenes) + 1)),
                    start_time=float(item.get("start_time", 0)),
                    end_time=float(item.get("end_time", 0)),
                    speaker_ids=[],
                    summary=str(item.get("summary", "")),
                    scene_type=str(item.get("scene_type", "other")),
                    utterance_indices=list(range(
                        int(item.get("start_idx", 0)),
                        int(item.get("end_idx", 0)) + 1
                    ))
                )
                
                # Extract speakers from utterances
                for idx in scene.utterance_indices:
                    if 0 <= idx < len(utterances):
                        spk = utterances[idx].get("speaker")
                        if spk is not None and spk not in scene.speaker_ids:
                            scene.speaker_ids.append(spk)
                
                scenes.append(scene)
                
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ TOON parse failed: {e}")
            # Fallback: treat entire video as one scene
            scenes = [Scene(
                id=1,
                start_time=0,
                end_time=utterances[-1]["end"] if utterances else 0,
                speaker_ids=[],
                summary="Full video",
                scene_type="other",
                utterance_indices=list(range(len(utterances)))
            )]
        
        return scenes
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Virality Scoring
    # ═══════════════════════════════════════════════════════════════════════
    
    def _phase2_score_scenes(
        self,
        scenes: List[Scene],
        utterances: List[dict],
        video_title: str,
        language: str,
        target_platform: str,
    ) -> List[ScoredScene]:
        """Score each scene for viral potential."""
        
        # Build scene texts
        scene_texts = []
        for scene in scenes:
            text_parts = []
            for idx in scene.utterance_indices:
                if 0 <= idx < len(utterances):
                    text_parts.append(utterances[idx]["text"])
            scene_texts.append({
                "id": scene.id,
                "start": scene.start_time,
                "end": scene.end_time,
                "duration": scene.duration,
                "type": scene.scene_type,
                "text": " ".join(text_parts)[:500],  # Limit text length
            })
        
        system_prompt = self._get_scorer_prompt(language, target_platform)
        
        user_prompt = f"""Score these video scenes for VIRAL POTENTIAL.

VIDEO: {video_title}
PLATFORM: {target_platform}

SCENES TO SCORE:
```
{json.dumps(scene_texts, ensure_ascii=False, indent=2)}
```

For each scene, evaluate:
1. HOOK STRENGTH (first 3 seconds grabbing power)
2. EMOTIONAL IMPACT (triggers engagement)
3. SHAREABILITY (would people share this?)
4. COMPLETENESS (does it tell a complete micro-story?)

Output as TOON:
```toon
scores[N]{{id,score,hook_text,triggers,why_viral}}:
  1,8,"Opening question hook","humor,surprise","Strong opening, funny punchline"
  2,5,"Weak start","","No clear hook, needs better opening"
  ...
```

score: 1-10 (10 = extremely viral)
hook_text: The grabbing text from first 3 seconds
triggers: comma-separated emotional triggers
why_viral: Brief explanation
"""
        
        response = self._call_llm(
            model=self.models["scorer"],
            system=system_prompt,
            user=user_prompt,
            phase="scoring"
        )
        
        return self._parse_scoring_response(response, scenes)
    
    def _get_scorer_prompt(self, language: str, platform: str) -> str:
        platform_tips = {
            "tiktok": "Focus on pattern interrupts, trending sounds potential, quick payoffs",
            "youtube": "Focus on watch time, strong hooks, satisfying conclusions",
            "instagram": "Focus on visual moments, quotable content, aesthetic appeal",
            "all": "Balance all platform requirements"
        }
        
        tip = platform_tips.get(platform, platform_tips["all"])
        
        if language == "it":
            return f"""Sei un esperto di contenuti virali. Valuta ogni scena per potenziale virale.

CRITERI DI VALUTAZIONE:
1. HOOK (primi 3 secondi): Cattura l'attenzione immediatamente?
2. EMOZIONE: Provoca reazione emotiva forte?
3. CONDIVISIBILITÀ: Le persone vorrebbero condividerlo?
4. COMPLETEZZA: Racconta una storia completa in sé?

PIATTAFORMA TARGET: {platform}
TIP: {tip}

SCORING:
- 9-10: Virale garantito, hook perfetto, emozione forte
- 7-8: Alto potenziale, buon hook, contenuto engaging
- 5-6: Medio potenziale, hook debole o contenuto incompleto
- 1-4: Basso potenziale, non adatto come clip

Rispondi SOLO con il blocco TOON richiesto."""
        else:
            return f"""You are a viral content expert. Evaluate each scene for viral potential.

EVALUATION CRITERIA:
1. HOOK (first 3 seconds): Does it grab attention immediately?
2. EMOTION: Does it provoke strong emotional reaction?
3. SHAREABILITY: Would people want to share this?
4. COMPLETENESS: Does it tell a complete story on its own?

TARGET PLATFORM: {platform}
TIP: {tip}

SCORING:
- 9-10: Guaranteed viral, perfect hook, strong emotion
- 7-8: High potential, good hook, engaging content
- 5-6: Medium potential, weak hook or incomplete content
- 1-4: Low potential, not suitable as clip

Respond ONLY with the requested TOON block."""
    
    def _parse_scoring_response(
        self, 
        response: str, 
        scenes: List[Scene]
    ) -> List[ScoredScene]:
        """Parse scoring response and merge with scenes."""
        scored = []
        
        try:
            raw_scores = toon.decode(response, strict=False)
            score_map = {int(s.get("id", 0)): s for s in raw_scores}
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Score parse failed: {e}")
            score_map = {}
        
        for scene in scenes:
            score_data = score_map.get(scene.id, {})
            
            triggers = score_data.get("triggers", "")
            if isinstance(triggers, str):
                triggers = [t.strip() for t in triggers.split(",") if t.strip()]
            
            scored_scene = ScoredScene(
                id=scene.id,
                start_time=scene.start_time,
                end_time=scene.end_time,
                speaker_ids=scene.speaker_ids,
                summary=scene.summary,
                scene_type=scene.scene_type,
                utterance_indices=scene.utterance_indices,
                virality_score=int(score_data.get("score", 5)),
                hook_text=str(score_data.get("hook_text", "")),
                emotional_triggers=triggers,
                why_viral=str(score_data.get("why_viral", "")),
                is_viable=scene.scene_type not in ("intro", "outro", "credits", "transition")
            )
            scored.append(scored_scene)
        
        return scored
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Boundary Refinement
    # ═══════════════════════════════════════════════════════════════════════
    
    def _phase3_refine_boundaries(
        self,
        scenes: List[ScoredScene],
        words: List[dict],
        utterances: List[dict],
        language: str,
    ) -> List[RefinedClip]:
        """
        Refine clip boundaries to align with word-level timestamps.
        Ensures no words are cut mid-pronunciation.
        """
        refined_clips = []
        
        for scene in scenes:
            # Find words within scene boundaries
            scene_words = [
                (i, w) for i, w in enumerate(words)
                if w["start"] >= scene.start_time - 0.5 
                and w["end"] <= scene.end_time + 0.5
            ]
            
            if not scene_words:
                continue
            
            # Find optimal start: first word that starts a sentence
            start_idx, start_word = self._find_sentence_start(
                scene_words, 
                scene.start_time,
                words
            )
            
            # Find optimal end: last word that ends a sentence
            end_idx, end_word = self._find_sentence_end(
                scene_words,
                scene.end_time, 
                words
            )
            
            if start_idx >= end_idx:
                continue
            
            # Calculate refined boundaries with padding
            refined_start = max(0, start_word["start"] - 0.2)
            refined_end = end_word["end"] + 0.3
            
            # Validate duration
            duration = refined_end - refined_start
            if duration < MIN_CLIP_DURATION:
                # Try to expand
                refined_start, refined_end, start_idx, end_idx = self._expand_to_minimum(
                    start_idx, end_idx, words, MIN_CLIP_DURATION
                )
                duration = refined_end - refined_start
            
            if duration > MAX_CLIP_DURATION:
                # Truncate to max (keeping complete sentences)
                refined_end, end_idx = self._truncate_to_maximum(
                    start_idx, words, MAX_CLIP_DURATION, refined_start
                )
                duration = refined_end - refined_start
            
            if duration < MIN_CLIP_DURATION:
                continue
            
            # Get first and last words text
            first_words = " ".join(
                w.get("punctuated_word", w["word"]) 
                for w in words[start_idx:min(start_idx+8, end_idx)]
            )
            last_words = " ".join(
                w.get("punctuated_word", w["word"]) 
                for w in words[max(start_idx, end_idx-8):end_idx+1]
            )
            
            clip = RefinedClip(
                scene_id=scene.id,
                start_time=scene.start_time,
                end_time=scene.end_time,
                refined_start=round(refined_start, 3),
                refined_end=round(refined_end, 3),
                first_word_idx=start_idx,
                last_word_idx=end_idx,
                first_words=first_words,
                last_words=last_words,
                hook_text=scene.hook_text,
                virality_score=scene.virality_score,
                why_viral=scene.why_viral,
                duration=round(duration, 2),
            )
            refined_clips.append(clip)
        
        return refined_clips
    
    def _find_sentence_start(
        self, 
        scene_words: List[tuple],
        target_time: float,
        all_words: List[dict]
    ) -> tuple:
        """Find the best sentence start near target time."""
        
        # Look for words that likely start sentences
        sentence_starters = []
        
        for idx, word in scene_words:
            text = word.get("punctuated_word", word["word"])
            
            # Check if previous word ended with sentence-ending punctuation
            if idx > 0:
                prev_text = all_words[idx-1].get("punctuated_word", "")
                if prev_text.endswith(('.', '!', '?', '...')):
                    sentence_starters.append((idx, word, abs(word["start"] - target_time)))
            
            # First word is always a potential starter
            if idx == scene_words[0][0]:
                sentence_starters.append((idx, word, abs(word["start"] - target_time)))
            
            # Words after long pauses (>0.5s)
            if idx > 0:
                gap = word["start"] - all_words[idx-1]["end"]
                if gap > 0.5:
                    sentence_starters.append((idx, word, abs(word["start"] - target_time)))
        
        if sentence_starters:
            # Sort by distance to target time
            sentence_starters.sort(key=lambda x: x[2])
            best = sentence_starters[0]
            return best[0], best[1]
        
        # Fallback: just use first word
        return scene_words[0]
    
    def _find_sentence_end(
        self,
        scene_words: List[tuple],
        target_time: float,
        all_words: List[dict]
    ) -> tuple:
        """Find the best sentence end near target time."""
        
        sentence_enders = []
        
        for idx, word in scene_words:
            text = word.get("punctuated_word", word["word"])
            
            # Words ending with sentence punctuation
            if text.endswith(('.', '!', '?', '...')):
                sentence_enders.append((idx, word, abs(word["end"] - target_time)))
            
            # Last word is always a potential ender
            if idx == scene_words[-1][0]:
                sentence_enders.append((idx, word, abs(word["end"] - target_time)))
            
            # Words before long pauses
            if idx < len(all_words) - 1:
                gap = all_words[idx+1]["start"] - word["end"]
                if gap > 0.5:
                    sentence_enders.append((idx, word, abs(word["end"] - target_time)))
        
        if sentence_enders:
            sentence_enders.sort(key=lambda x: x[2])
            best = sentence_enders[0]
            return best[0], best[1]
        
        return scene_words[-1]
    
    def _expand_to_minimum(
        self,
        start_idx: int,
        end_idx: int,
        words: List[dict],
        min_duration: float
    ) -> tuple:
        """Expand boundaries to meet minimum duration."""
        
        current_start = words[start_idx]["start"]
        current_end = words[end_idx]["end"]
        duration = current_end - current_start
        
        while duration < min_duration:
            # Try expanding end first
            if end_idx < len(words) - 1:
                end_idx += 1
                current_end = words[end_idx]["end"]
            elif start_idx > 0:
                start_idx -= 1
                current_start = words[start_idx]["start"]
            else:
                break
            
            duration = current_end - current_start
        
        return (
            max(0, current_start - 0.2),
            current_end + 0.3,
            start_idx,
            end_idx
        )
    
    def _truncate_to_maximum(
        self,
        start_idx: int,
        words: List[dict],
        max_duration: float,
        start_time: float
    ) -> tuple:
        """Truncate to maximum duration while keeping complete sentences."""
        
        target_end = start_time + max_duration
        
        # Find last sentence-ending word before target
        best_end_idx = start_idx
        
        for i in range(start_idx, len(words)):
            word = words[i]
            
            if word["end"] > target_end:
                break
            
            text = word.get("punctuated_word", word["word"])
            if text.endswith(('.', '!', '?', '...')):
                best_end_idx = i
        
        # If no sentence end found, just use last word before target
        if best_end_idx == start_idx:
            for i in range(start_idx, len(words)):
                if words[i]["end"] > target_end:
                    best_end_idx = max(start_idx, i - 1)
                    break
        
        return words[best_end_idx]["end"] + 0.3, best_end_idx
    
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
        """Call LLM and track costs."""
        
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
                # Rough cost estimate (varies by model)
                cost = tokens * 0.0000003  # ~$0.30 per 1M tokens average
                self.phase_costs[phase] = self.phase_costs.get(phase, 0) + cost
                self.total_cost += cost
                
                if self.debug:
                    print(f"   [{phase}] {tokens} tokens, ~${cost:.6f}")
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"LLM call failed ({phase}): {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_analysis(self, analysis: dict, video_id: str) -> str:
        """Save analysis to file."""
        output_path = ANALYSIS_DIR / f"{video_id}_multiagent_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Integration with existing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def analyze_with_multiagent(
    transcript: dict,
    video_title: str,
    max_clips: int = 5,
    language: str = "it",
    target_platform: str = "all",
    debug: bool = False,
) -> dict:
    """
    Convenience function to run multi-agent analysis.
    Drop-in replacement for ClipAnalyzer.analyze_for_clips()
    """
    analyzer = MultiAgentAnalyzer(debug=debug)
    return analyzer.analyze(
        transcript=transcript,
        video_title=video_title,
        max_clips=max_clips,
        language=language,
        target_platform=target_platform,
    )
