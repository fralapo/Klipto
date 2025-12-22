# dry_run.py
"""
Dry-Run System for Enhanced Multi-Agent Video Analyzer
- Interactive preview of scenes and clips before cutting
- Manual modification of clip selection
- Transcript preview for each clip
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path

# Import from enhanced_analyzer module
from enhanced_analyzer import (
    EnhancedMultiAgentAnalyzer,
    WordBoundaryAligner,
    Word,
    Scene,
    Clip,
    ToonV3
)


# ============================================================================
# Console Colors & Formatting
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.RESET = ''


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.s"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def format_time_short(seconds: float) -> str:
    """Format seconds as M:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def viral_stars(score: int) -> str:
    """Convert viral score to star display"""
    filled = '⭐' * score
    empty = '☆' * (10 - score)
    return filled + empty


def type_emoji(scene_type: str) -> str:
    """Get emoji for scene type"""
    emojis = {
        'intro': '🎬',
        'outro': '👋',
        'interview': '🎤',
        'monologue': '🗣️',
        'highlight': '✨',
        'transition': '🔄',
        'teaser': '👀',
        'unknown': '❓'
    }
    return emojis.get(scene_type, '❓')


# ============================================================================
# Dry Run Display
# ============================================================================

class DryRunDisplay:
    """Handles all terminal display for dry-run mode"""
    
    def __init__(self, width: int = 70):
        self.width = width
        
        # Check if terminal supports colors
        # In some environments (like standard Windows CMD without ANSI support enabled), 
        # this might need more robust checking, but for now we assume standard behavior.
        try:
            if not os.isatty(sys.stdout.fileno()):
                Colors.disable()
        except Exception:
            pass # Fallback
    
    def header(self, text: str):
        """Print main header"""
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(self.width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * self.width}{Colors.RESET}")
        print()
    
    def section(self, text: str, emoji: str = ""):
        """Print section header"""
        if emoji:
            text = f"{emoji} {text}"
        print(f"\n{Colors.BOLD}{Colors.YELLOW}{text}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * self.width}{Colors.RESET}")
    
    def print_scenes_table(self, scenes: List[Scene], excluded_types: List[str] = None):
        """Print scenes in table format"""
        if excluded_types is None:
            excluded_types = ['intro', 'outro', 'teaser']
        
        # Header
        print(f" {'#':>2}  │ {'Tipo':<10} │ {'Tempo':<15} │ {'Durata':>7} │ {'Score':>5} │ Descrizione")
        print(f"{'─' * 4}┼{'─' * 12}┼{'─' * 17}┼{'─' * 9}┼{'─' * 7}┼{'─' * 20}")
        
        excluded_ids = []
        
        for scene in scenes:
            time_range = f"{format_time_short(scene.start)} - {format_time_short(scene.end)}"
            duration = f"{scene.end - scene.start:.0f}s"
            score = f"{scene.viral_score}/10"
            emoji = type_emoji(scene.scene_type)
            desc = scene.description[:25] if scene.description else ""
            
            is_excluded = scene.scene_type in excluded_types
            
            if is_excluded:
                excluded_ids.append(scene.id)
                color = Colors.DIM
            elif scene.viral_score >= 8:
                color = Colors.GREEN
            elif scene.viral_score >= 5:
                color = Colors.RESET
            else:
                color = Colors.DIM
            
            print(f"{color} {scene.id:>2}  │ {emoji} {scene.scene_type:<8} │ {time_range:<15} │ {duration:>7} │ {score:>5} │ {desc}{Colors.RESET}")
        
        if excluded_ids:
            excluded_str = ", ".join(f"#{id}" for id in excluded_ids)
            print(f"\n{Colors.YELLOW}⚠️  ESCLUSE: {excluded_str} (intro/outro/teaser){Colors.RESET}")
    
    def print_clip_card(self, clip: Clip, index: int, transcript: str = None):
        """Print single clip as card"""
        status_icon = "✅" if not clip.quality_notes else "⚠️"
        status_text = "APPROVATA" if not clip.quality_notes else f"NOTA: {clip.quality_notes[0]}"
        
        print(f"\n {Colors.BOLD}CLIP {index + 1}{Colors.RESET} {'─' * (self.width - 10)}")
        print(f" │ {Colors.CYAN}Titolo:{Colors.RESET}     {clip.title}")
        print(f" │ {Colors.CYAN}Scena:{Colors.RESET}      #{clip.scene_id}")
        print(f" │ {Colors.CYAN}Tempo:{Colors.RESET}      {format_time(clip.start)} → {format_time(clip.end)}  ({clip.duration:.1f}s)")
        print(f" │ {Colors.CYAN}Parole:{Colors.RESET}     \"{clip.start_word}\" → \"{clip.end_word}\"")
        print(f" │ {Colors.CYAN}Viral:{Colors.RESET}      {viral_stars(clip.viral_score)} {clip.viral_score}/10")
        print(f" │ {Colors.CYAN}Stato:{Colors.RESET}      {status_icon} {status_text}")
        
        if transcript:
            print(f" │")
            print(f" │ {Colors.DIM}Trascrizione:{Colors.RESET}")
            # Word wrap transcript
            words = transcript.split()
            line = " │   "
            for word in words:
                if len(line) + len(word) + 1 > self.width:
                    print(f"{Colors.DIM}{line}{Colors.RESET}")
                    line = " │   "
                line += word + " "
            if line.strip() != "│":
                print(f"{Colors.DIM}{line}{Colors.RESET}")
        
        print(f" └{'─' * (self.width - 2)}")
    
    def print_clips(self, clips: List[Clip]):
        """Print all clips"""
        for i, clip in enumerate(clips):
            self.print_clip_card(clip, i)
    
    def print_summary(self, scenes: List[Scene], clips: List[Clip], eligible_count: int, cost_estimate: float = 0.002):
        """Print analysis summary"""
        total_duration = sum(c.duration for c in clips)
        
        print(f"\n{Colors.BOLD}{'═' * self.width}{Colors.RESET}")
        print(f"{Colors.BOLD}{'RIEPILOGO'.center(self.width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * self.width}{Colors.RESET}")
        print(f" Scene totali:     {len(scenes)}")
        print(f" Scene eligibili:  {eligible_count} (escluse intro/outro/teaser)")
        print(f" Clip generate:    {len(clips)}")
        print(f" Durata totale:    {total_duration:.1f}s di contenuto")
        print(f"\n Costo LLM stimato: ~${cost_estimate:.4f}")
        print(f"{'═' * self.width}")
    
    def prompt_action(self) -> str:
        """Prompt user for action"""
        print(f"\n{Colors.BOLD}Vuoi procedere con il taglio?{Colors.RESET}")
        print(f"  {Colors.GREEN}s{Colors.RESET} = Sì, taglia tutte le clip")
        print(f"  {Colors.RED}n{Colors.RESET} = No, annulla")
        print(f"  {Colors.YELLOW}m{Colors.RESET} = Modifica selezione")
        print()
        
        try:
            response = input(f"{Colors.BOLD}Scelta [s/N/m]: {Colors.RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return 'n'
        
        return response if response in ['s', 'n', 'm', 'modifica'] else 'n'


# ============================================================================
# Interactive Editor
# ============================================================================

class ClipEditor:
    """Interactive editor for clip modification"""
    
    def __init__(self, scenes: List[Scene], clips: List[Clip], words: List[Word], display: DryRunDisplay):
        self.scenes = scenes
        self.clips = clips.copy()
        self.words = words
        self.display = display
        self.aligner = WordBoundaryAligner(words)
    
    def get_transcript_for_clip(self, clip: Clip) -> str:
        """Extract transcript text for a clip's time range"""
        text_words = []
        for word in self.words:
            if word.start >= clip.start and word.end <= clip.end:
                text_words.append(word.punctuated_word or word.word)
        return ' '.join(text_words)
    
    def get_transcript_for_scene(self, scene: Scene) -> str:
        """Get full transcript for a scene"""
        return scene.transcript
    
    def run(self) -> List[Clip]:
        """Run interactive editor loop"""
        self._print_help()
        
        while True:
            try:
                cmd = input(f"\n{Colors.CYAN}>{Colors.RESET} ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nAnnullato.")
                return []
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0]
            
            if action in ['procedi', 'p', 'ok']:
                return self.clips
            
            elif action in ['annulla', 'quit', 'q']:
                return []
            
            elif action in ['help', 'h', '?']:
                self._print_help()
            
            elif action in ['lista', 'l', 'list']:
                self._list_clips()
            
            elif action in ['scene', 'scenes']:
                self._list_scenes()
            
            elif action in ['anteprima', 'preview', 'a'] and len(parts) > 1:
                self._preview_clip(parts[1])
            
            elif action in ['escludi', 'rimuovi', 'e', 'r'] and len(parts) > 1:
                self._remove_clip(parts[1])
            
            elif action in ['aggiungi', 'add'] and len(parts) > 1:
                self._add_clip_from_scene(parts[1])
            
            elif action in ['modifica', 'edit'] and len(parts) > 1:
                self._edit_clip(parts[1])
            
            elif action in ['scena', 'scene'] and len(parts) > 1:
                self._preview_scene(parts[1])
            
            else:
                print(f"{Colors.RED}Comando non riconosciuto. Digita 'help' per aiuto.{Colors.RESET}")
    
    def _print_help(self):
        """Print help message"""
        print(f"\n{Colors.BOLD}Modalità modifica. Comandi disponibili:{Colors.RESET}")
        print(f"  {Colors.CYAN}lista{Colors.RESET}          - Mostra clip correnti")
        print(f"  {Colors.CYAN}scene{Colors.RESET}          - Mostra tutte le scene")
        print(f"  {Colors.CYAN}anteprima N{Colors.RESET}    - Mostra trascrizione clip N")
        print(f"  {Colors.CYAN}scena N{Colors.RESET}        - Mostra trascrizione scena N")
        print(f"  {Colors.CYAN}escludi N{Colors.RESET}      - Rimuove clip N")
        print(f"  {Colors.CYAN}aggiungi N{Colors.RESET}     - Crea clip da scena N")
        print(f"  {Colors.CYAN}modifica N{Colors.RESET}     - Modifica tempi clip N")
        print(f"  {Colors.CYAN}procedi{Colors.RESET}        - Taglia le clip confermate")
        print(f"  {Colors.CYAN}annulla{Colors.RESET}        - Esci senza tagliare")
    
    def _list_clips(self):
        """List current clips"""
        if not self.clips:
            print(f"{Colors.YELLOW}Nessuna clip selezionata.{Colors.RESET}")
            return
        
        print(f"\n{Colors.BOLD}Clip correnti ({len(self.clips)}):{Colors.RESET}")
        for i, clip in enumerate(self.clips):
            duration = f"{clip.duration:.1f}s"
            try:
                # Handle cases where score might be missing or None
                score = clip.viral_score if clip.viral_score is not None else 0
            except:
                score = 0
            print(f"  {i+1}. [{format_time_short(clip.start)}-{format_time_short(clip.end)}] {duration:>6} │ {clip.title} (score: {score})")
    
    def _list_scenes(self):
        """List all scenes"""
        self.display.section("TUTTE LE SCENE")
        self.display.print_scenes_table(self.scenes, excluded_types=[])
    
    def _preview_clip(self, clip_num: str):
        """Preview a clip's transcript"""
        try:
            idx = int(clip_num) - 1
            if 0 <= idx < len(self.clips):
                clip = self.clips[idx]
                transcript = self.get_transcript_for_clip(clip)
                self.display.print_clip_card(clip, idx, transcript)
            else:
                print(f"{Colors.RED}Clip {clip_num} non trovata.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Numero non valido.{Colors.RESET}")
    
    def _preview_scene(self, scene_num: str):
        """Preview a scene's transcript"""
        try:
            idx = int(scene_num)
            scene = next((s for s in self.scenes if s.id == idx), None)
            if scene:
                print(f"\n{Colors.BOLD}SCENA {scene.id} - {scene.scene_type}{Colors.RESET}")
                print(f"Tempo: {format_time(scene.start)} → {format_time(scene.end)} ({scene.end - scene.start:.1f}s)")
                print(f"Viral score: {scene.viral_score}/10")
                print(f"\n{Colors.DIM}Trascrizione:{Colors.RESET}")
                print(f"{scene.transcript}")
            else:
                print(f"{Colors.RED}Scena {scene_num} non trovata.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Numero non valido.{Colors.RESET}")
    
    def _remove_clip(self, clip_num: str):
        """Remove a clip"""
        try:
            idx = int(clip_num) - 1
            if 0 <= idx < len(self.clips):
                removed = self.clips.pop(idx)
                print(f"{Colors.GREEN}Rimossa clip: {removed.title}{Colors.RESET}")
                # Re-number clips
                for i, clip in enumerate(self.clips):
                    clip.id = i
            else:
                print(f"{Colors.RED}Clip {clip_num} non trovata.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Numero non valido.{Colors.RESET}")
    
    def _add_clip_from_scene(self, scene_num: str):
        """Add a clip from a scene"""
        try:
            idx = int(scene_num)
            scene = next((s for s in self.scenes if s.id == idx), None)
            
            if not scene:
                print(f"{Colors.RED}Scena {scene_num} non trovata.{Colors.RESET}")
                return
            
            print(f"\n{Colors.BOLD}Crea clip da scena {scene.id}{Colors.RESET}")
            print(f"Scena: {format_time(scene.start)} → {format_time(scene.end)} ({scene.end - scene.start:.1f}s)")
            print(f"\nInserisci i tempi (invio per usare tutta la scena):")
            
            start_input = input(f"  Start [{format_time(scene.start)}]: ").strip()
            if start_input:
                start = self._parse_time(start_input)
                if start is None:
                    return
            else:
                start = scene.start
            
            end_input = input(f"  End [{format_time(scene.end)}]: ").strip()
            if end_input:
                end = self._parse_time(end_input)
                if end is None:
                    return
            else:
                end = scene.end
            
            title = input("  Titolo: ").strip() or f"Clip da scena {scene.id}"
            
            # Align to word boundaries
            aligned_start, aligned_end, start_word, end_word = self.aligner.align_to_sentences(start, end)
            
            new_clip = Clip(
                id=len(self.clips),
                scene_id=scene.id,
                start=aligned_start,
                end=aligned_end,
                duration=aligned_end - aligned_start,
                start_word=start_word,
                end_word=end_word,
                title=title,
                description="",
                viral_score=scene.viral_score
            )
            
            self.clips.append(new_clip)
            print(f"{Colors.GREEN}Aggiunta clip: {title} ({new_clip.duration:.1f}s){Colors.RESET}")
            
        except ValueError:
            print(f"{Colors.RED}Numero non valido.{Colors.RESET}")
    
    def _edit_clip(self, clip_num: str):
        """Edit a clip's timing"""
        try:
            idx = int(clip_num) - 1
            if not (0 <= idx < len(self.clips)):
                print(f"{Colors.RED}Clip {clip_num} non trovata.{Colors.RESET}")
                return
            
            clip = self.clips[idx]
            print(f"\n{Colors.BOLD}Modifica clip {idx + 1}: {clip.title}{Colors.RESET}")
            print(f"Tempo attuale: {format_time(clip.start)} → {format_time(clip.end)}")
            
            start_input = input(f"  Nuovo start [{format_time(clip.start)}]: ").strip()
            if start_input:
                start = self._parse_time(start_input)
                if start is None:
                    return
            else:
                start = clip.start
            
            end_input = input(f"  Nuovo end [{format_time(clip.end)}]: ").strip()
            if end_input:
                end = self._parse_time(end_input)
                if end is None:
                    return
            else:
                end = clip.end
            
            # Align to word boundaries
            aligned_start, aligned_end, start_word, end_word = self.aligner.align_to_sentences(start, end)
            
            clip.start = aligned_start
            clip.end = aligned_end
            clip.duration = aligned_end - aligned_start
            clip.start_word = start_word
            clip.end_word = end_word
            
            print(f"{Colors.GREEN}Clip modificata: {format_time(clip.start)} → {format_time(clip.end)} ({clip.duration:.1f}s){Colors.RESET}")
            
        except ValueError:
            print(f"{Colors.RED}Numero non valido.{Colors.RESET}")
    
    def _parse_time(self, time_str: str) -> Optional[float]:
        """Parse time string (M:SS or seconds) to float"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    mins = int(parts[0])
                    secs = float(parts[1])
                    return mins * 60 + secs
            else:
                return float(time_str)
        except ValueError:
            print(f"{Colors.RED}Formato tempo non valido. Usa M:SS o secondi.{Colors.RESET}")
            return None


# ============================================================================
# Dry Run Manager
# ============================================================================

class DryRunManager:
    """Manages the dry-run workflow"""
    
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
        self.analyzer = EnhancedMultiAgentAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            min_clip_duration=min_clip_duration,
            max_clip_duration=max_clip_duration,
            max_clips=max_clips,
            debug=debug
        )
        self.display = DryRunDisplay()
        self.debug = debug
    
    def run(self, transcript_path: str) -> Optional[List[dict]]:
        """
        Run dry-run analysis and return approved clips.
        
        Returns:
            List of clip dictionaries if approved, None if cancelled.
        """
        # Load transcript for word data
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Use updated methods from shared instance logic
        # Note: self.analyzer has _parse_words, _parse_utterances etc.
        # We need to ensure we call them correctly.
        
        words = self.analyzer._parse_words(transcript_data)
        utterances = self.analyzer._parse_utterances(transcript_data)
        
        if not words and not utterances:
             print(f"{Colors.RED}Errore: Impossibile parsare la trascrizione o trascrizione vuota.{Colors.RESET}")
             return None

        # Run analysis
        self.display.header("ANALISI VIDEO - DRY RUN")
        
        print(f"{Colors.DIM}Analisi in corso...{Colors.RESET}")
        
        # Phase 1: Scene detection
        scenes = self.analyzer.scene_detector.detect(utterances, words)
        print(f"  ✓ Rilevate {len(scenes)} scene")
        
        # Phase 2: Classification
        scenes = self.analyzer._phase2_classify_scenes(scenes)
        print(f"  ✓ Scene classificate")
        
        # Filter eligible
        excluded_types = ['intro', 'outro', 'teaser']
        eligible_scenes = [s for s in scenes if s.scene_type not in excluded_types]
        print(f"  ✓ {len(eligible_scenes)} scene eligibili")
        
        # Phase 3: Clip selection
        clips = self.analyzer._phase3_select_clips(eligible_scenes)
        print(f"  ✓ Selezionate {len(clips)} clip")
        
        # Phase 4: Supervision
        clips = self.analyzer._phase4_supervise(clips, scenes)
        print(f"  ✓ Supervisione completata")
        
        # Phase 5: Word alignment
        aligner = WordBoundaryAligner(words)
        clips = self.analyzer._phase5_align_boundaries(clips, aligner)
        print(f"  ✓ Allineamento parole completato")
        
        # Display results
        self.display.section("SCENE RILEVATE", "📊")
        self.display.print_scenes_table(scenes, excluded_types)
        
        self.display.section(f"CLIP PROPOSTE ({len(clips)} di max {self.analyzer.max_clips})", "🎬")
        self.display.print_clips(clips)
        
        self.display.print_summary(scenes, clips, len(eligible_scenes))
        
        # Prompt for action
        action = self.display.prompt_action()
        
        if action == 's':
            print(f"\n{Colors.GREEN}Procedo con il taglio...{Colors.RESET}")
            return [asdict(c) for c in clips]
        
        elif action in ['m', 'modifica']:
            editor = ClipEditor(scenes, clips, words, self.display)
            modified_clips = editor.run()
            
            if modified_clips:
                print(f"\n{Colors.GREEN}Procedo con il taglio di {len(modified_clips)} clip...{Colors.RESET}")
                return [asdict(c) for c in modified_clips]
            else:
                print(f"\n{Colors.YELLOW}Operazione annullata.{Colors.RESET}")
                return None
        
        else:
            print(f"\n{Colors.YELLOW}Operazione annullata.{Colors.RESET}")
            return None
    
    def save_analysis(self, scenes: List[Scene], clips: List[Clip], output_path: str):
        """Save analysis results to JSON"""
        results = {
            'scenes': [asdict(s) for s in scenes],
            'clips': [asdict(c) for c in clips],
            'metadata': {
                'total_scenes': len(scenes),
                'final_clips': len(clips)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"{Colors.DIM}Analisi salvata in {output_path}{Colors.RESET}")


# ============================================================================
# Main Entry Point
# ============================================================================

def dry_run_analyze(
    transcript_path: str,
    api_key: str = None,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    max_clips: int = 5,
    debug: bool = False
) -> Optional[List[dict]]:
    """
    Run dry-run analysis with interactive preview.
    
    Args:
        transcript_path: Path to Deepgram transcript JSON
        api_key: OpenRouter API key (or from env OPENROUTER_API_KEY)
        min_duration: Minimum clip duration
        max_duration: Maximum clip duration
        max_clips: Maximum number of clips
        debug: Enable debug output
    
    Returns:
        List of approved clip dictionaries, or None if cancelled
    """
    if api_key is None:
        api_key = os.environ.get('OPENROUTER_API_KEY', '')
    
    if not api_key:
        print(f"{Colors.RED}Errore: API key non trovata. Imposta OPENROUTER_API_KEY.{Colors.RESET}")
        return None
    
    manager = DryRunManager(
        api_key=api_key,
        min_clip_duration=min_duration,
        max_clip_duration=max_duration,
        max_clips=max_clips,
        debug=debug
    )
    
    return manager.run(transcript_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dry_run.py <transcript.json> [--debug]")
        sys.exit(1)
    
    transcript = sys.argv[1]
    debug = '--debug' in sys.argv
    
    clips = dry_run_analyze(transcript, debug=debug)
    
    if clips:
        print(f"\n{Colors.GREEN}Clip approvate: {len(clips)}{Colors.RESET}")
        for clip in clips:
            print(f"  - {clip['title']}: {clip['start']:.2f}s - {clip['end']:.2f}s")
    else:
        print(f"\n{Colors.YELLOW}Nessuna clip generata.{Colors.RESET}")
