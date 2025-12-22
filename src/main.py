"""
AI YouTube Shorts Generator - Main Pipeline
Orchestrates video download, transcription, analysis, and precision cutting.

Usage:
    python main.py process <source> [options]
    python main.py analyze <source> [options]
    python main.py cut <video> --analysis <json> [options]
    python main.py transcribe <source> [options]
    python main.py validate <transcript> --video <video> [options]
    python main.py models [options]
    python main.py cache [options]
"""

import argparse
import json
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from config import BASE_DIR

from config import (
    DOWNLOADS_DIR,
    TRANSCRIPTS_DIR,
    ANALYSIS_DIR,
    CLIPS_DIR,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    TARGET_CLIP_DURATION,
    OPENROUTER_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEEPGRAM_API_KEY,
)
from deepgram_cache import DeepgramModelCache
from deepgram_selector import DeepgramModelSelector
from downloader import VideoDownloader, VideoSource
from transcriber import Transcriber, generate_srt
from analyzer import ClipAnalyzer
from clipper import VideoClipper, ClipResult
from multi_agent_analyzer import analyze_with_multiagent
from semantic_analyzer import (
    analyze_video_enhanced,
    EnhancedMultiAgentAnalyzer,
    estimate_token_savings,
)
from enhanced_analyzer import analyze_video as analyze_video_v4
from dry_run import dry_run_analyze

# Optional precision imports
try:
    from precision import (
        TranscriptionParser,
        VideoWordCutter,
        WordTimestamp,
        AudioAnalyzer,
        VADValidator,
        BoundaryRefiner,
    )
    PRECISION_AVAILABLE = True
except ImportError:
    PRECISION_AVAILABLE = False


@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    success: bool
    video_id: str
    source: str
    source_type: str
    # Paths
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    analysis_path: Optional[Path] = None
    # Results
    duration: float = 0
    clips_found: int = 0
    clips_cut: int = 0
    clips: list = None
    # Timing
    download_time: float = 0
    transcribe_time: float = 0
    analyze_time: float = 0
    cut_time: float = 0
    total_time: float = 0
    # Costs
    transcription_cost: float = 0
    analysis_cost: float = 0
    total_cost: float = 0
    # Validation
    validation_report: Optional[dict] = None
    # Errors
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.clips is None:
            self.clips = []


class ShortsPipeline:
    """
    Complete pipeline for generating YouTube Shorts from long videos.
    
    Workflow:
    1. Download video (YouTube, Twitch, or local file)
    2. Extract audio for transcription
    3. Transcribe with Deepgram (word-level timestamps)
    4. Analyze with LLM to find viral clips
    5. Validate timestamps with audio analysis (optional)
    6. Cut clips with FFmpeg (precision mode)
    """
    
    def __init__(
        self,
        # Model settings
        llm_model: str = "deepseek/deepseek-chat",
        transcription_model: str = "nova-2",
        language: str = "it",
        # Precision settings
        enable_precision: bool = True,
        validate_timestamps: bool = True,
        # Clip settings
        min_duration: int = MIN_CLIP_DURATION,
        max_duration: int = MAX_CLIP_DURATION,
        target_duration: int = TARGET_CLIP_DURATION,
        # Cut settings
        cut_method: str = "hybrid",
        crop_vertical: bool = True,
        # Output settings
        output_dir: Optional[Path] = None,
    ):
        self.llm_model = llm_model
        self.transcription_model = transcription_model
        self.language = language
        self.enable_precision = enable_precision and PRECISION_AVAILABLE
        self.validate_timestamps = validate_timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.cut_method = cut_method
        self.crop_vertical = crop_vertical
        self.output_dir = output_dir or CLIPS_DIR
        
        # Initialize components
        self.downloader = VideoDownloader()
        
        # Validate Model/Language compatibility
        # We need to check if the chosen model supports the language.
        try:
            cache = DeepgramModelCache(DEEPGRAM_API_KEY)
            if not cache.is_combination_supported(transcription_model, language):
                print(f"⚠️  WARNING: Model '{transcription_model}' might not support language '{language}'.")
                # We could fallback, but since is_combination_supported returns True if model not found,
                # this only triggers if model IS found but language is NOT in its list.
                # In that case, fallback is wise.
                fallback = "nova-2"
                print(f"   Falling back to '{fallback}' (General Purpose).")
                transcription_model = fallback
        except Exception as e:
            print(f"⚠️  Validation skipped due to cache error: {e}")

        self.transcriber = Transcriber(
            model=transcription_model,
            language=language,
        )
        self.analyzer = ClipAnalyzer(model=llm_model)
        self.clipper = VideoClipper(
            output_dir=self.output_dir,
            enable_precision=self.enable_precision,
        )
        
        # Ensure directories exist
        for dir_path in [DOWNLOADS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR, CLIPS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process(
        self,
        source: str,
        max_clips: int = 5,
        download_only: bool = False,
        transcribe_only: bool = False,
        analyze_only: bool = False,
        skip_download: bool = False,
        skip_transcription: bool = False,
        skip_analysis: bool = False,
        skip_cutting: bool = False,
        force: bool = False,
        use_cache: bool = True,
        target_platform: Optional[str] = None,
        use_multiagent: bool = True,
        analyzer_type: str = "enhanced",
        dry_run: bool = False,
        interactive: bool = True,
        debug: bool = False,
    ) -> PipelineResult:
        """
        Run complete pipeline on a video source.
        
        Args:
            source: Video URL or local file path
            max_clips: Maximum number of clips to generate
            download_only: If True, only download the video and exit.
            transcribe_only: If True, download and transcribe, then exit.
            analyze_only: If True, download, transcribe, and analyze, then exit.
            use_cache: If True, use cached results for download, transcription, and analysis if available.
            target_platform: Target platform (tiktok, youtube_shorts, instagram_reels)
            use_multiagent: If True, use the multi-agent analysis system.
        
        Returns:
            PipelineResult with all outputs and metrics
        """
        start_time = time.time()
        
        # Detect source type
        source_type = self.downloader.detect_source(source)
        
        result = PipelineResult(
            success=False,
            video_id="",
            source=source,
            source_type=source_type.value,
        )
        
        video_title = source # Default to source if title not found
        
        try:
            # Step 1: Download/Get Video
            print(f"\n{'='*60}")
            print("📥 Step 1: Getting Video")
            print(f"{'='*60}")
            
            step_start = time.time()
            
            if source_type == VideoSource.LOCAL:
                video_path = Path(source)
                if not video_path.exists():
                    raise FileNotFoundError(f"Video not found: {source}")
                result.video_id = video_path.stem
                video_title = video_path.stem
                print(f"✓ Using local file: {video_path.name}")
            else:
                # Get video info first
                info = self.downloader.get_info(source)
                result.video_id = info.id
                result.duration = info.duration
                video_title = info.title
                
                print(f"  Title: {info.title}")
                print(f"  Duration: {info.duration:.1f}s ({info.duration/60:.1f} min)")
                
                if use_cache and not force and not skip_download:
                    # Check cache
                    cache_key = self.downloader._get_cache_key(source)
                    if cache_key in self.downloader.cache:
                        video_path = Path(self.downloader.cache[cache_key]["path"])
                        if video_path.exists():
                            print(f"✓ Using cached video: {video_path.name}")
                        else:
                            video_path = self.downloader.download(source, force=force)
                    else:
                        video_path = self.downloader.download(source, force=force)
                else:
                    # If skip_download is True, we try to find it in cache, otherwise error?
                    # The original intention of skip_download usually implies "assume it's there or use cache".
                    # But if we rely on downloader.download(..., force=False), it effectively skips if present.
                    # force=True forces re-download.
                    video_path = self.downloader.download(source, force=force)
                
                print(f"✓ Video ready: {video_path.name}")
            
            result.video_path = video_path
            result.download_time = time.time() - step_start
            
            if download_only:
                print("\nDownload only mode: Exiting after download.")
                result.success = True
                result.total_time = time.time() - start_time
                return result

            # Extract audio for transcription
            audio_path = TRANSCRIPTS_DIR / f"{result.video_id}_audio.wav"
            if not audio_path.exists() or not use_cache:
                print("  Extracting audio...")
                audio_path = self.downloader.extract_audio_for_transcription(
                    video_path, audio_path
                )
            result.audio_path = audio_path
            
            # Step 2: Transcription
            print(f"\n{'='*60}")
            print("🎙️ Step 2: Transcription (Deepgram)")
            print(f"{'='*60}")
            
            step_start = time.time()
            transcript_path = TRANSCRIPTS_DIR / f"{result.video_id}_transcript.json"
            
            if (use_cache or skip_transcription) and transcript_path.exists() and not force:
                print(f"✓ Using cached transcript: {transcript_path.name}")
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = json.load(f)
            else:
                print(f"  Model: {self.transcription_model}")
                
                transcript = self.transcriber.transcribe(
                    audio_path=audio_path,
                    model=self.transcription_model,
                    language=self.language,
                    diarize=True,
                    punctuate=True,
                    paragraphs=True,
                    utterances=True,
                    smart_format=True,
                )
                
                # Save transcript
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(transcript, f, ensure_ascii=False, indent=2)
                
                # Generate SRT
                srt_path = TRANSCRIPTS_DIR / f"{result.video_id}.srt"
                srt_content = generate_srt(transcript)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                
                print(f"✓ Transcription complete")
                print(f"  Words: {len(transcript.get('words', []))}")
                print(f"  Utterances: {len(transcript.get('utterances', []))}")
                print(f"  Duration: {transcript.get('duration', 0):.1f}s")
            
            result.transcript_path = transcript_path
            result.transcribe_time = time.time() - step_start
            result.duration = transcript.get("duration", result.duration)
            
            # Estimate transcription cost (~$0.0043/min for Nova-2)
            result.transcription_cost = (result.duration / 60) * 0.0043
            
            # Step 2.5: Validate timestamps (optional)
            if self.validate_timestamps and self.enable_precision:
                print(f"\n{'='*60}")
                print("🔍 Step 2.5: Timestamp Validation")
                print(f"{'='*60}")
                
                try:
                    words = TranscriptionParser.parse(transcript_path, "deepgram")
                    
                    # Initialize validators
                    audio_analyzer = None
                    vad_validator = None
                    
                    try:
                        audio_analyzer = AudioAnalyzer(audio_path)
                        refiner = BoundaryRefiner(audio_analyzer)
                        print("  ✓ Audio analyzer initialized")
                    except Exception as e:
                        print(f"  ⚠ Audio analyzer not available: {e}")
                    
                    try:
                        vad_validator = VADValidator(audio_path)
                        print("  ✓ VAD validator initialized")
                    except Exception as e:
                        print(f"  ⚠ VAD validator not available: {e}")
                    
                    # Validate words
                    if vad_validator:
                        words = vad_validator.validate_words(words, min_speech_ratio=0.3)
                        valid_count = sum(1 for w in words if w.is_valid)
                        print(f"  VAD validation: {valid_count}/{len(words)} words valid")
                    
                    # Refine boundaries
                    if audio_analyzer:
                        for word in words:
                            refiner.refine_word(word)
                        
                        avg_diff = sum(
                            (w.start_diff_ms or 0) + (w.end_diff_ms or 0) 
                            for w in words
                        ) / (len(words) * 2) if words else 0
                        print(f"  Boundary refinement: avg diff {avg_diff:.1f}ms")
                    
                    result.validation_report = {
                        "total_words": len(words),
                        "valid_words": sum(1 for w in words if w.is_valid),
                        "avg_boundary_diff_ms": avg_diff if audio_analyzer else None,
                    }
                    
                except Exception as e:
                    print(f"  ⚠ Validation skipped: {e}")
            
            # Step 3: AI Analysis
            print(f"\n{'='*60}")
            print("🧠 Step 3: AI Analysis (Viral Clip Detection)")
            print(f"{'='*60}")
            
            step_start = time.time()
            analysis_path = ANALYSIS_DIR / f"{result.video_id}_analysis.json"
            
            if (skip_analysis or use_cache) and analysis_path.exists() and not force:
                print(f"✓ Using cached analysis: {analysis_path.name}")
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
            else:
                print(f"  Model: {self.llm_model}")
                print(f"  Target: {max_clips} clips")
                if target_platform:
                    print(f"  Platform: {target_platform}")
                
                if analyzer_type == "enhanced":
                    # Enhanced Multi-Agent Analysis (TOON)
                    print(f"  Analyzer: Enhanced Multi-Agent v3.1 (TOON)")
                    
                    if debug:
                        savings = estimate_token_savings(transcript)
                        print(f"  Token savings estimate:")
                        for field, data in savings.items():
                            print(f"    {field}: {data['savings_percent']:.1f}% savings "
                                  f"({data['json_tokens']} -> {data['toon_tokens']} tokens)")
                    
                    analysis = analyze_video_enhanced(
                        transcript=transcript,
                        video_title=video_title or result.source,
                        max_clips=max_clips,
                        language=self.language,
                        target_platform=target_platform or "all",
                        debug=debug,
                    )
                    
                elif analyzer_type == "multiagent":
                    # V4.0 Enhanced Multi-Agent (Strict TOON v3.0)
                    print(f"  Analyzer: Enhanced Multi-Agent v4.0 (TOON v3.0)")
                    
                    if dry_run or interactive:
                         # Interactive Dry Run
                         clips = dry_run_analyze(
                             transcript_path=str(result.transcript_path),
                             min_duration=self.min_duration,
                             max_duration=self.max_duration,
                             max_clips=max_clips,
                             debug=debug
                         )
                         
                         if clips is None:
                             print("\n⚠️ Operazione annullata dall'utente.")
                             result.success = True
                             return result
                             
                         # Wrap in analysis dict format
                         analysis = {
                             "clips": [
                                 {
                                     "start_time": c["start"],
                                     "end_time": c["end"],
                                     "virality_score": c["viral_score"],
                                     "hook_text": c.get("title", ""),
                                     "description": c.get("description", "")
                                 } for c in clips
                             ],
                             "metadata": {"method": "dry-run-v4"}
                         }
                    else:
                        analysis = analyze_video_v4(
                            transcript_path=str(result.transcript_path),
                            min_duration=self.min_duration,
                            max_duration=self.max_duration,
                            max_clips=max_clips,
                            debug=debug
                        )
                    
                else:
                    # Simple Analysis
                    print(f"  Analyzer: Simple")
                    # Estimate cost
                    cost_estimate = self.analyzer.estimate_cost(transcript)
                    print(f"  Estimated cost: ${cost_estimate.get('total_cost', 0):.4f}")
                    
                    analysis = self.analyzer.analyze_for_clips(
                        transcript,
                        video_title=result.source,
                        max_clips=max_clips,
                        target_platform=target_platform or "all",
                        language=self.language
                    )

                # Save analysis
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2)
                
                print(f"✓ Analysis complete")
                print(f"  Clips found: {len(analysis.get('clips', []))}")
                
                result.analysis_cost = analysis.get("costs", {}).get("total", 0)
            
            result.analysis_path = analysis_path
            result.clips_found = len(analysis.get("clips", []))
            result.analyze_time = time.time() - step_start
            
            # Display found clips
            print("\n📋 Clips identified:")
            for i, clip in enumerate(analysis.get("clips", [])[:max_clips], 1):
                duration = clip["end_time"] - clip["start_time"]
                score = clip.get("virality_score", "N/A")
                print(f"  {i}. [{clip['start_time']:.1f}s - {clip['end_time']:.1f}s] "
                      f"({duration:.1f}s) - Score: {score}/10")
                print(f"     Hook: {clip.get('hook_text', 'N/A')[:50]}...")
            
            # Step 4: Cut Clips
            if not skip_cutting and result.clips_found > 0:
                print(f"\n{'='*60}")
                print("✂️ Step 4: Cutting Clips (FFmpeg)")
                print(f"{'='*60}")
                
                step_start = time.time()
                
                print(f"  Method: {self.cut_method}")
                print(f"  Crop vertical: {self.crop_vertical}")
                print(f"  Precision mode: {self.enable_precision}")
                
                clip_results = self.clipper.cut_clips_from_analysis(
                    video_path=video_path,
                    analysis=analysis,
                    audio_path=audio_path if self.enable_precision else None,
                    method=self.cut_method,
                    validate=self.validate_timestamps,
                    crop_vertical=self.crop_vertical,
                )
                
                result.cut_time = time.time() - step_start
                
                # Process results
                successful_clips = []
                for i, clip_result in enumerate(clip_results, 1):
                    if clip_result.success:
                        successful_clips.append({
                            "index": i,
                            "path": str(clip_result.path),
                            "duration": clip_result.duration,
                            "boundary_quality": clip_result.boundary_quality,
                            "start_time": clip_result.start_time,
                            "end_time": clip_result.end_time,
                            "refined_start": clip_result.refined_start,
                            "refined_end": clip_result.refined_end,
                        })
                        print(f"  ✓ Clip {i}: {clip_result.path.name} "
                              f"({clip_result.duration:.1f}s, {clip_result.boundary_quality})")
                    else:
                        print(f"  ✗ Clip {i} failed: {clip_result.error[:50]}...")
                
                result.clips = successful_clips
                result.clips_cut = len(successful_clips)
                
                print(f"\n✓ Cut complete: {result.clips_cut}/{result.clips_found} clips")
            
            # Final summary
            result.total_time = time.time() - start_time
            result.total_cost = result.transcription_cost + result.analysis_cost
            result.success = True
            
            print(f"\n{'='*60}")
            print("📊 Pipeline Complete")
            print(f"{'='*60}")
            print(f"  Video: {result.video_id}")
            print(f"  Duration: {result.duration:.1f}s ({result.duration/60:.1f} min)")
            print(f"  Clips: {result.clips_cut}/{result.clips_found} cut successfully")
            print(f"\n  ⏱️ Timing:")
            print(f"    Download: {result.download_time:.1f}s")
            print(f"    Transcribe: {result.transcribe_time:.1f}s")
            print(f"    Analyze: {result.analyze_time:.1f}s")
            print(f"    Cut: {result.cut_time:.1f}s")
            print(f"    Total: {result.total_time:.1f}s")
            print(f"\n  💰 Cost:")
            print(f"    Transcription: ${result.transcription_cost:.4f}")
            print(f"    Analysis: ${result.analysis_cost:.4f}")
            print(f"    Total: ${result.total_cost:.4f}")
            
            # Save pipeline result
            result_path = ANALYSIS_DIR / f"{result.video_id}_pipeline_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2, default=str)
            
            return result
            
        except Exception as e:
            result.error = str(e)
            result.total_time = time.time() - start_time
            
            if "ERRORE POLICY" in str(e):
                 # Messaggio già formattato, non serve prefisso o traceback
                 print(str(e))
            else:
                print(f"\n❌ Pipeline failed: {e}")
                import traceback
                traceback.print_exc()
            
            return result
    
    def analyze_only(
        self,
        source: str,
        max_clips: int = 5,
        target_platform: Optional[str] = None,
    ) -> dict:
        """
        Only analyze a video without cutting.
        Returns analysis results.
        """
        result = self.process(
            source=source,
            max_clips=max_clips,
            target_platform=target_platform,
            skip_cutting=True,
        )
        
        if result.analysis_path and result.analysis_path.exists():
            with open(result.analysis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return {"error": result.error or "Analysis failed"}
    
    def cut_from_analysis(
        self,
        video_source: str,
        analysis_path: Path,
        method: str = "hybrid",
        crop_vertical: bool = True,
    ) -> list[ClipResult]:
        """
        Cut clips from an existing analysis file.
        """
        # Load analysis
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        
        # Get video
        source_type = self.downloader.detect_source(video_source)
        
        if source_type == VideoSource.LOCAL:
            video_path = Path(video_source)
        else:
            video_path = self.downloader.download(video_source)
        
        # Extract audio if precision enabled
        audio_path = None
        if self.enable_precision:
            audio_path = video_path.with_suffix(".wav")
            if not audio_path.exists():
                audio_path = self.downloader.extract_audio_for_transcription(
                    video_path, audio_path
                )
        
        # Cut clips
        return self.clipper.cut_clips_from_analysis(
            video_path=video_path,
            analysis=analysis,
            audio_path=audio_path,
            method=method,
            validate=self.validate_timestamps,
            crop_vertical=crop_vertical,
        )
    
    def transcribe_only(
        self,
        source: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict:
        """
        Only transcribe a video.
        Returns transcription result.
        """
        # Get video
        source_type = self.downloader.detect_source(source)
        
        if source_type == VideoSource.LOCAL:
            video_path = Path(source)
            video_id = video_path.stem
        else:
            info = self.downloader.get_info(source)
            video_id = info.id
            video_path = self.downloader.download(source)
        
        # Extract audio
        audio_path = TRANSCRIPTS_DIR / f"{video_id}_audio.wav"
        if not audio_path.exists():
            audio_path = self.downloader.extract_audio_for_transcription(
                video_path, audio_path
            )
        
        # Transcribe
        transcript = self.transcriber.transcribe(
            audio_path=audio_path,
            model=model or self.transcription_model,
            language=language or self.language,
            diarize=True,
            punctuate=True,
            paragraphs=True,
            utterances=True,
            smart_format=True,
        )
        
        # Save
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}_transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        # Generate SRT
        srt_path = TRANSCRIPTS_DIR / f"{video_id}.srt"
        srt_content = generate_srt(transcript)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        return {
            "transcript_path": str(transcript_path),
            "srt_path": str(srt_path),
            "duration": transcript.get("duration", 0),
            "words": len(transcript.get("words", [])),
            "utterances": len(transcript.get("utterances", [])),
        }
    
    def validate_transcript(
        self,
        transcript_path: Path,
        video_path: Path,
    ) -> dict:
        """
        Validate a transcript against video audio.
        """
        if not PRECISION_AVAILABLE:
            return {"error": "Precision module not available"}
        
        # Extract audio
        audio_path = video_path.with_suffix(".wav")
        if not audio_path.exists():
            self.downloader.extract_audio_for_transcription(video_path, audio_path)
        
        # Parse transcript
        words = TranscriptionParser.parse(transcript_path, "auto")
        
        # Initialize validators
        report = {
            "total_words": len(words),
            "vad_validation": None,
            "boundary_analysis": None,
        }
        
        try:
            vad = VADValidator(audio_path)
            words = vad.validate_words(words)
            
            valid_count = sum(1 for w in words if w.is_valid)
            invalid_words = [
                {"word": w.word, "start": w.start, "speech_ratio": w.speech_ratio}
                for w in words if not w.is_valid
            ][:20]
            
            report["vad_validation"] = {
                "valid_words": valid_count,
                "invalid_words": len(words) - valid_count,
                "validation_rate": valid_count / len(words) * 100,
                "sample_invalid": invalid_words,
            }
        except Exception as e:
            report["vad_validation"] = {"error": str(e)}
        
        try:
            analyzer = AudioAnalyzer(audio_path)
            refiner = BoundaryRefiner(analyzer)
            
            diffs = []
            for word in words[:100]:  # Sample first 100 words
                refiner.refine_word(word)
                if word.start_diff_ms is not None:
                    diffs.append((word.start_diff_ms + word.end_diff_ms) / 2)
            
            report["boundary_analysis"] = {
                "sampled_words": len(diffs),
                "avg_diff_ms": sum(diffs) / len(diffs) if diffs else 0,
                "max_diff_ms": max(diffs) if diffs else 0,
                "min_diff_ms": min(diffs) if diffs else 0,
            }
        except Exception as e:
            report["boundary_analysis"] = {"error": str(e)}
        
        return report



def update_env_default(key: str, value: str):
    """Helper to update .env file."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        print(f"⚠️  File .env non trovato in {env_path}")
        return
    
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex replace or append
        pattern = f"^{key}=.*$"
        replacement = f"{key}={value}"
        
        if re.search(pattern, content, re.MULTILINE):
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            new_content = content + f"\n{replacement}"
        
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✅ Aggiornato {key} in .env")
    except Exception as e:
        print(f"❌ Errore aggiornamento .env: {e}")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a YouTube video
  python main.py process "https://youtube.com/watch?v=..." --max-clips 5
  
  # Process a local video
  python main.py process ./video.mp4 --platform tiktok
  
  # Only analyze without cutting
  python main.py analyze "https://youtube.com/watch?v=..."
  
  # Cut from existing analysis
  python main.py cut ./video.mp4 --analysis ./analysis.json
  
  # Only transcribe
  python main.py transcribe "https://youtube.com/watch?v=..." --language en
  
  # Validate transcript
  python main.py validate ./transcript.json --video ./video.mp4
  
  # Manage LLM models
  python main.py models --list
  python main.py models --set "anthropic/claude-3-haiku"
  
  # Manage Deepgram transcriber models
  python main.py transcriber --list
  python main.py transcriber --set "nova-2-ea"
  
  # Manage cache
  python main.py cache --info
  python main.py cache --clear
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Full pipeline")
    process_parser.add_argument("source", help="Video URL or local file")
    process_parser.add_argument("--max-clips", type=int, default=5, help="Max clips to generate")
    process_parser.add_argument("--platform", choices=["tiktok", "youtube_shorts", "instagram_reels"],
                                help="Target platform")
    process_parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model")
    process_parser.add_argument("--transcription-model", default=DEFAULT_TRANSCRIPTION_MODEL, help="Deepgram model")
    process_parser.add_argument("--language", default="it", help="Language code")
    process_parser.add_argument("--method", choices=["fast", "accurate", "hybrid"],
                                default="hybrid", help="Cutting method")
    process_parser.add_argument("--no-crop", action="store_true", help="Don't crop to vertical")
    process_parser.add_argument("--no-precision", action="store_true", help="Disable precision mode")
    process_parser.add_argument("--no-validation", action="store_true", help="Skip timestamp validation")
    process_parser.add_argument("--skip-download", action="store_true", help="Use cached video")
    process_parser.add_argument("--skip-transcription", action="store_true", help="Use cached transcript")
    process_parser.add_argument("--skip-analysis", action="store_true", help="Use cached analysis")
    process_parser.add_argument("--force", action="store_true", help="Force re-processing")
    process_parser.add_argument("--analyzer", choices=["simple", "enhanced", "multiagent"], default="multiagent",
                                help="Analyzer type: simple, enhanced (v3.1), or multiagent (v4.0)")
    process_parser.add_argument("--dry-run", action="store_true", help="Interactive dry-run mode (only for multiagent)")
    process_parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode for dry-run")
    process_parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Only analyze video")
    analyze_parser.add_argument("source", help="Video URL or local file")
    analyze_parser.add_argument("--max-clips", type=int, default=5, help="Max clips to find")
    analyze_parser.add_argument("--platform", help="Target platform")
    analyze_parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model")
    analyze_parser.add_argument("--language", default="it", help="Language code")
    
    # Cut command
    cut_parser = subparsers.add_parser("cut", help="Cut from existing analysis")
    cut_parser.add_argument("video", help="Video file or URL")
    cut_parser.add_argument("--analysis", required=True, help="Analysis JSON file")
    cut_parser.add_argument("--method", choices=["fast", "accurate", "hybrid"],
                            default="hybrid", help="Cutting method")
    cut_parser.add_argument("--no-crop", action="store_true", help="Don't crop to vertical")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Only transcribe")
    transcribe_parser.add_argument("source", help="Video URL or local file")
    transcribe_parser.add_argument("--model", default=DEFAULT_TRANSCRIPTION_MODEL, help="Deepgram model")
    transcribe_parser.add_argument("--language", default="it", help="Language code")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate transcript")
    validate_parser.add_argument("transcript", help="Transcript JSON file")
    validate_parser.add_argument("--video", required=True, help="Video file")
    
    # --- Transcriber Management ---
    transcriber_parser = subparsers.add_parser("transcriber", help="Gestisci modelli di trascrizione (Deepgram)")
    transcriber_parser.add_argument("--list", action="store_true", help="Lista modelli disponibili")
    transcriber_parser.add_argument("--set", type=str, help="Imposta modello di default")
    transcriber_parser.add_argument("--interactive", action="store_true", help="Modalità interattiva")

    # --- Models Management ---
    models_parser = subparsers.add_parser("models", help="Manage LLM models")
    models_parser.add_argument("--list", action="store_true", help="List available models")
    models_parser.add_argument("--interactive", action="store_true", help="Interactive selection mode")
    models_parser.add_argument("--set", help="Set default model")
    models_parser.add_argument("--credits", action="store_true", help="Check API credits")
    
    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage download cache")
    cache_parser.add_argument("--info", action="store_true", help="Show cache info")
    cache_parser.add_argument("--list", action="store_true", help="List cached files")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--older-than", type=int, help="Clear files older than N days")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "process":
        pipeline = ShortsPipeline(
            llm_model=args.model,
            transcription_model=args.transcription_model,
            language=args.language,
            enable_precision=not args.no_precision,
            validate_timestamps=not args.no_validation,
            cut_method=args.method,
            crop_vertical=not args.no_crop,
        )
        
        try:
            result = pipeline.process(
                source=args.source,
                max_clips=args.max_clips,
                target_platform=args.platform,
                skip_download=args.skip_download,
                skip_transcription=args.skip_transcription,
                skip_analysis=args.skip_analysis,
                force=args.force,
                dry_run=args.dry_run,
                interactive=not args.no_interactive and args.dry_run,
            )
            return 0 if result.success else 1
            
        except Exception as e:
            # Handle user-friendly errors without traceback
            if "ERRORE POLICY" in str(e):
                print(str(e))
                return 1
            raise
    
    elif args.command == "analyze":
        pipeline = ShortsPipeline(
            llm_model=args.model,
            language=args.language,
        )
        
        try:
            result = pipeline.analyze_only(
                source=args.source,
                max_clips=args.max_clips,
                target_platform=args.platform,
            )
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if "error" not in result else 1

        except Exception as e:
            if "ERRORE POLICY" in str(e):
                print(str(e))
                return 1
            raise
    
    elif args.command == "cut":
        pipeline = ShortsPipeline(
            cut_method=args.method,
            crop_vertical=not args.no_crop,
        )
        
        results = pipeline.cut_from_analysis(
            video_source=args.video,
            analysis_path=Path(args.analysis),
            method=args.method,
            crop_vertical=not args.no_crop,
        )
        
        success_count = sum(1 for r in results if r.success)
        print(f"Cut {success_count}/{len(results)} clips successfully")
        return 0 if success_count > 0 else 1
    
    elif args.command == "transcribe":
        pipeline = ShortsPipeline(
            transcription_model=args.model,
            language=args.language,
        )
        
        result = pipeline.transcribe_only(
            source=args.source,
            model=args.model,
            language=args.language,
        )
        
        print(json.dumps(result, indent=2))
        return 0
    
    elif args.command == "validate":
        if not PRECISION_AVAILABLE:
            print("Error: Precision module not available. Install librosa and webrtcvad.")
            return 1
        
        pipeline = ShortsPipeline()
        result = pipeline.validate_transcript(
            transcript_path=Path(args.transcript),
            video_path=Path(args.video),
        )
        
        print(json.dumps(result, indent=2))
        return 0
    
    elif args.command == "models":
        analyzer = ClipAnalyzer()

        # Check API Key using new method
        if not analyzer.is_configured():
            print("\n⚠️  OpenRouter API Key non configurata!")
            print("   Per vedere la lista completa e i prezzi aggiornati, configura la chiave.")
            print("\n📋 Modelli consigliati (Fallback):")
            print(f"{'ID Modello':<45} {'Prezzo In/Out ($/M)'}")
            print("-" * 70)
            for m in analyzer.get_recommended_models():
                print(f"{m.id:<45} ${m.input_price}/{m.output_price}")
            
            if args.interactive:
                print("\n🔧 Configurazione Rapida:")
                print("   1. Ottieni chiave da: https://openrouter.ai/keys")
                key = input("   2. Incolla qui la chiave (o premi Invio per uscire): ").strip()
                if key and key.startswith("sk-or-"):
                    update_env_default("OPENROUTER_API_KEY", key)
                    analyzer = ClipAnalyzer() # Re-init
                    # Continue to list...
                else:
                    return 1
            else:
                 print("\nUsa --interactive per configurare ora.")
                 return 1
            
        if args.list or args.interactive:
            try:
                print("\n🔄 Recupero lista modelli aggiornata da OpenRouter...")
                models = analyzer.get_available_models(refresh=True)
            except Exception as e:
                print(f"❌ Errore recupero modelli: {e}")
                return 1
            
            if not models:
                print("❌ Nessun modello recuperato.")
                return 1

            # Cache models
            all_models = models
            recommended_ids = [m.id for m in analyzer.get_recommended_models()]
            
            # Application state
            view_mode = "recommended" # recommended, top, all, search
            filter_text = ""
            
            while True:
                # 1. Determine models to display based on mode
                display_list = []
                title = ""
                
                if view_mode == "search":
                    display_list = [m for m in all_models if filter_text.lower() in m.id.lower()]
                    title = f"🔍 Risultati ricerca: '{filter_text}'"
                
                elif view_mode == "all":
                    display_list = all_models
                    title = "≡ Lista Completa (Tutti i modelli)"
                    
                elif view_mode == "free":
                    # Filter strictly by ":free" suffix as per OpenRouter convention
                    display_list = [m for m in all_models if m.id.endswith(":free")]
                    title = "🆓 Modelli Gratuiti (:free)"
                    
                else: # canonical 'recommended' replaced by 'newest'
                    # Sort desc by created date (newest first)
                    # Use a stable sort key, treating empty dates as very old
                    def date_key(m):
                         return m.created or 0
                    
                    newest_models = sorted(all_models, key=date_key, reverse=True)
                    display_list = newest_models[:20]
                    title = "✨ Nuovi Arrivi & Trending (Ultimi 20)"

                # 2. Render UI
                # Artificial clear screen by printing newlines
                print("\n" * 5)
                print(f"╔{'═'*80}╗")
                print(f"║ {title:<78} ║")
                print(f"╚{'═'*80}╝")
                print(f" 👉 Modello Attuale: {analyzer.model}")
                print(f"{'#':<4} {'ID Modello':<45} {'Context':<10} {'Input $/M':<12} {'Output $/M':<12}")
                print("-" * 82)
                
                # Limit display if list is huge (unless in 'all' mode, maybe paginate?)
                # Actually user asked for 'full list', but printing 100 lines is annoying.
                # Let's show up to 50, then say "... and X more".
                LIMIT = 50
                showing_count = len(display_list)
                
                for idx, model in enumerate(display_list[:LIMIT], 1):
                    ctx_str = f"{model.context_length // 1000}k" if model.context_length else "?"
                    
                    # Highlight current
                    is_current = model.id == analyzer.model
                    prefix = "👉" if is_current else "  "
                    # idx column
                    idx_str = f"{idx}"
                    
                    # Smart formatting for prices
                    def fmt_p(p):
                        if p < 0.01: return f"${p:.4f}".rstrip('0').rstrip('.')
                        return f"${p:.2f}"
                    
                    row = (f"{prefix} {idx_str:<3} {model.id:<45} {ctx_str:<10} "
                           f"{fmt_p(model.input_price):<12} {fmt_p(model.output_price):<12}")
                    print(row)
                
                if showing_count > LIMIT:
                    print(f"\n... e altri {showing_count - LIMIT} modelli (usa il filtro per raffinare).")
                elif showing_count == 0:
                    print("\n   (Nessun modello trovato)")

                if not args.interactive:
                    break

                # 3. Interactive Menu
                print("-" * 82)
                print(" OPZIONI:")
                print(" [testo]  🔍 Cerca modello per nome (es. 'gpt', 'claude')")
                print(" [numero] ✅ Seleziona modello dalla lista sopra")
                print(" 'all'    ≡  Vedi lista completa")
                print(" 'new'    ✨ Vedi nuovi arrivi (Default)")
                print(" 'free'   🆓 Vedi SOLO gratuiti")
                print(" 'q'      ❌ Esci")
                
                raw_choice = input("\n👉 Comando: ").strip()
                choice = raw_choice.lower()
                
                if choice in ['q', 'exit', 'quit', 'b']:
                    break
                
                if choice == 'all':
                    view_mode = "all"
                    continue
                elif choice == 'free':
                    view_mode = "free"
                    continue
                elif choice == 'new' or choice == 'rec' or choice == 'recommended':
                    view_mode = "recommended" # keep internal name but logic is new
                    continue
                # Removed 'top' option handling
                
                if not choice:
                    continue

                if choice.isdigit():
                    # SELECTION
                    idx = int(choice)
                    if 1 <= idx <= len(display_list):
                        selected = display_list[idx-1]
                        update_env_default("DEFAULT_LLM_MODEL", selected.id)
                        analyzer.set_model(selected.id)
                        print(f"\n✅ Modello impostato: {selected.id}")
                        time.sleep(1.5)
                        # Don't break, allow seeing the change (highlighted)
                        # Refetch cache? No need.
                    else:
                        print(f"❌ Numero {idx} non valido per questa lista.")
                        time.sleep(1.5)
                else:
                    # SEARCH / FILTER
                    filter_text = raw_choice # Keep case for display if needed, but search lower
                    view_mode = "search"
                    continue
        
        elif args.set:
            update_env_default("DEFAULT_LLM_MODEL", args.set)
            analyzer.set_model(args.set)
        
        elif args.credits:
            credits = analyzer.check_credits()
            if credits and "error" not in credits:
                print(f"\n💰 OpenRouter Credits:")
                print(f"   Totale Crediti: ${credits.get('total_credits', 0):.2f}")
                print(f"   Usati:          ${credits.get('used', 0):.2f}")
                print(f"   Rimanenti:      ${credits.get('remaining', 0):.2f}")
                if credits.get('label'):
                    print(f"   Label:          {credits.get('label')}")
            else:
                print(f"\n❌ Errore/Info: {credits.get('error', 'Sconosciuto')}")
        
        else:
            print(f"Modello attuale: {analyzer.model}")
        
        return 0
    
    elif args.command == "cache":
        downloader = VideoDownloader()
        
        if args.info:
            info = downloader.get_cache_info()
            print(f"\n📦 Download Cache:")
            print(f"  Files: {info['file_count']}")
            print(f"  Size: {info['total_size_mb']:.1f} MB")
            print(f"  Entries: {info['cache_entries']}")
        
        elif args.list:
            print("\n📦 Cached Videos:\n")
            for key, info in downloader.cache.items():
                path = Path(info['path'])
                exists = "✓" if path.exists() else "✗"
                print(f"  {exists} {info.get('title', key)[:50]}")
                print(f"    Path: {path.name}")
                print(f"    Duration: {info.get('duration', 0):.1f}s")
                print()
        
        elif args.clear:
            count = downloader.clear_cache(older_than_days=args.older_than)
            print(f"✓ Cleared {count} cached files")
        
        else:
            info = downloader.get_cache_info()
            print(f"Cache: {info['file_count']} files, {info['total_size_mb']:.1f} MB")
        
        return 0
    
    elif args.command == "transcriber":
        cache = DeepgramModelCache(DEEPGRAM_API_KEY)
        
        if args.list or args.interactive:
            while True:
                # Load models (tries api first usually, but cache behavior is smart)
                data = cache.get_models()
                
                # Use the new Smart Selector with Language-First flow
                selector = DeepgramModelSelector(data)
                
                # If just listing, we might want a different view, but for now interactive is primary
                if args.interactive:
                    print("\n🚀 Avvio configurazione guidata (Language-First)...")
                    sel_model, sel_lang, sel_mode = selector.interactive_select_model()
                    
                    if not sel_model:
                        print("❌ Selezione annullata.")
                        break
                        
                    print(f"\n✅ Configurazione scelta:")
                    print(f"   Modello: {sel_model}")
                    print(f"   Lingua:  {sel_lang}")
                    print(f"   Mode:    {sel_mode}")
                    
                    # Update .env
                    update_env_default("DEFAULT_TRANSCRIPTION_MODEL", sel_model)
                    # We should also arguably update the default language if the CLI supported it, 
                    # but Klipto usually takes language from the video args.
                    # For now we only persist the model ID.
                    print(f"   💾 Modello salvato in .env!")
                    break
                
                else:
                    # Non-interactive list (legacy view or simple dump?)
                    # Since the user specifically wanted the interactive flow, let's auto-switch to it 
                    # OR just print a summary.
                    print("⚠️  Per usare il selettore avanzato, usa l'opzione --interactive (o -i)")
                    break
                        
        elif args.set:
             # Logic simplifies: assume user knows what they are doing or use cache check
             update_env_default("DEFAULT_TRANSCRIPTION_MODEL", args.set)
             print(f"✅ Modello trascrizione impostato: {args.set}")

        else:
             print(f"Modello trascrizione attuale: {DEFAULT_TRANSCRIPTION_MODEL}")
        
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
