"""
Video clipper with precision cutting support.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config import CLIPS_DIR

# Import precision module if available
try:
    from precision import (
        PrecisionCutter,
        CutSegment,
        TranscriptionParser,
        AudioAnalyzer,
        BoundaryRefiner,
        format_timestamp_ffmpeg
    )
    PRECISION_AVAILABLE = True
except ImportError:
    PRECISION_AVAILABLE = False


@dataclass
class ClipResult:
    """Result of clip extraction."""
    path: Path
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    # Precision metrics
    boundary_quality: Optional[str] = None
    speech_coverage: Optional[float] = None
    refined_start: Optional[float] = None
    refined_end: Optional[float] = None


class VideoClipper:
    """FFmpeg-based video clipper with precision support."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_precision: bool = True
    ):
        self.output_dir = output_dir or CLIPS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_precision = enable_precision and PRECISION_AVAILABLE
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream
        
        # Parse fps
        fps = 30
        if video_stream:
            fps_str = video_stream.get("r_frame_rate", "30/1")
            try:
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = int(num) / int(den)
                else:
                    fps = float(fps_str)
            except (ValueError, ZeroDivisionError):
                fps = 30
        
        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "video": {
                "width": video_stream.get("width") if video_stream else None,
                "height": video_stream.get("height") if video_stream else None,
                "fps": fps,
                "codec": video_stream.get("codec_name") if video_stream else None,
            },
            "audio": {
                "sample_rate": int(audio_stream.get("sample_rate", 44100)) if audio_stream else 44100,
                "channels": audio_stream.get("channels") if audio_stream else 2,
                "codec": audio_stream.get("codec_name") if audio_stream else None,
            },
        }
    
    def _get_hardware_encoder_settings(self) -> dict:
        """
        Detects available hardware acceleration and returns optimal FFmpeg settings.
        Supports: NVIDIA (NVENC), macOS (VideoToolbox), CPU (x264/Software fallback).
        """
        try:
            # Check available encoders
            result = subprocess.run(
                ["ffmpeg", "-v", "quiet", "-encoders"], 
                capture_output=True, 
                text=True
            )
            output = result.stdout
            
            # 1. NVIDIA (NVENC)
            if "h264_nvenc" in output:
                print("  ⚡ Hardware detected: NVIDIA GPU (NVENC)")
                return {
                    "input_flags": ["-hwaccel", "cuda"],
                    "codec": "h264_nvenc",
                    "encoder_flags": ["-rc", "constqp", "-qp", "23", "-preset", "p6", "-tune", "hq"],
                    "audio_codec": "aac"
                }

            # 2. macOS (Apple Silicon / Intel VideoToolbox)
            if "h264_videotoolbox" in output:
                print("  🍎 Hardware detected: macOS (VideoToolbox)")
                return {
                    "input_flags": ["-hwaccel", "videotoolbox"],
                    "codec": "h264_videotoolbox",
                    "encoder_flags": ["-q:v", "60"],  # ~CRF 23 equivalent
                    "audio_codec": "aac"
                }

        except Exception as e:
            print(f"  ⚠️ Warning: Hardware detection failed ({e}). Falling back to CPU.")
            
        # 3. CPU Fallback
        print("  🐌 No hardware acceleration found. Using CPU.")
        return {
            "input_flags": [],
            "codec": "libx264",
            "encoder_flags": ["-crf", "23", "-preset", "medium"],
            "audio_codec": "aac"
        }

    def cut_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        output_name: Optional[str] = None,
        # Method
        method: str = "hybrid",  # fast, accurate, hybrid
        # Precision options
        audio_path: Optional[Path] = None,
        validate_boundaries: bool = True,
        # Output settings
        output_format: str = "mp4",
        # Crop options
        crop_vertical: bool = False,
        # Filters
        video_filters: Optional[list[str]] = None,
        audio_filters: Optional[list[str]] = None,
    ) -> ClipResult:
        """
        Cut a clip with hardware acceleration and cinematic cropping.
        """
        video_path = Path(video_path)
        duration = end_time - start_time
        
        if output_name is None:
            output_name = f"{video_path.stem}_{start_time:.1f}-{end_time:.1f}"
        
        output_path = self.output_dir / f"{output_name}.{output_format}"
        
        # Precision validation
        refined_start = start_time
        refined_end = end_time
        boundary_quality = "unvalidated"
        speech_coverage = None
        
        if self.enable_precision and validate_boundaries and audio_path:
            try:
                analyzer = AudioAnalyzer(audio_path)
                refiner = BoundaryRefiner(analyzer)
                
                segment = CutSegment(start=start_time, end=end_time)
                segment = refiner.refine_segment(segment)
                
                refined_start = segment.refined_start or start_time
                refined_end = segment.refined_end or end_time
                boundary_quality = segment.boundary_quality
                
                duration = refined_end - refined_start
            except Exception as e:
                print(f"Precision validation failed: {e}")
        
        # Get optimal hardware settings
        hw_settings = self._get_hardware_encoder_settings()
        
        # Build filter chain
        vfilters = video_filters.copy() if video_filters else []
        afilters = audio_filters.copy() if audio_filters else []
        
        # Vertical crop ("Cinematic Zoom")
        if crop_vertical:
            # Logic: Zoom 15% -> Crop Center 1080 -> Pad Vertical 1920
            # Filters run on CPU for compatibility
            # Pre-calc: 1920*1.15 = 2208, 1080*1.15 = 1242
            vfilters.append(
                "scale=2208:1242,"
                "crop=1080:ih,"
                "pad=1080:1920:(ow-iw)/2:(1920-ih)/2:black,"
                "setsar=1[v]"
            )
            
            # Force re-encoding for filter changes
            if method == "fast":
                method = "hybrid"
        
        # Build command based on method
        cmd = ["ffmpeg", "-y"]
        
        # Input optimization
        if hw_settings["input_flags"]:
            cmd.extend(hw_settings["input_flags"])
            
        if method == "fast" and not vfilters and not afilters:
            cmd.extend([
                "-ss", str(refined_start),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path)
            ])
        
        else:
            # Calculate offsets for hybrid/accurate
            if method == "hybrid":
                keyframe_seek = max(0, refined_start - 5)
                offset = refined_start - keyframe_seek
                cmd.extend(["-ss", str(keyframe_seek)])
                cmd.extend(["-i", str(video_path)])
                cmd.extend(["-ss", str(offset)])
            else: # accurate
                cmd.extend(["-i", str(video_path)])
                cmd.extend(["-ss", str(refined_start)])

            cmd.extend(["-t", str(duration)])
            
            # Video Encoding Settings
            cmd.extend(["-c:v", hw_settings["codec"]])
            cmd.extend(hw_settings["encoder_flags"])

            # Audio Encoding Settings
            if afilters:
                # Must re-encode if filters are present
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            else:
                # Copy if possible (per optimization request)
                cmd.extend(["-c:a", hw_settings["audio_codec"]])
                
            cmd.extend(["-avoid_negative_ts", "make_zero"])
            cmd.extend(["-movflags", "+faststart"]) # Web optimization
            
            if vfilters:
                cmd.extend(["-filter_complex", ",".join(vfilters)])
                cmd.extend(["-map", "[v]"]) # Map filtered video
                cmd.extend(["-map", "0:a"]) # Map original audio
            
            if afilters:
                cmd.extend(["-af", ",".join(afilters)])
            
            cmd.append(str(output_path))
            
        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        success = result.returncode == 0 and output_path.exists()
        
        return ClipResult(
            path=output_path,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error=result.stderr if not success else None,
            boundary_quality=boundary_quality,
            speech_coverage=speech_coverage,
            refined_start=refined_start if refined_start != start_time else None,
            refined_end=refined_end if refined_end != end_time else None,
        )
    
    def cut_clips_from_analysis(
        self,
        video_path: Path,
        analysis: dict,
        audio_path: Optional[Path] = None,
        method: str = "hybrid",
        validate: bool = True,
        crop_vertical: bool = True,
        **kwargs
    ) -> list[ClipResult]:
        """Cut multiple clips from analysis results."""
        results = []
        clips = analysis.get("clips", [])
        
        for i, clip in enumerate(clips):
            output_name = f"{Path(video_path).stem}_clip{i+1:02d}"
            
            result = self.cut_clip(
                video_path=video_path,
                start_time=clip["start_time"],
                end_time=clip["end_time"],
                output_name=output_name,
                method=method,
                audio_path=audio_path,
                validate_boundaries=validate,
                crop_vertical=crop_vertical,
                **kwargs
            )
            
            # Attach clip metadata
            result.clip_info = clip
            results.append(result)
        
        return results
    
    def verify_clip(self, clip_path: Path) -> dict:
        """Verify clip integrity and sync."""
        info = self.get_video_info(clip_path)
        
        # Check audio sync
        sync_offset = self._check_av_sync(clip_path)
        
        return {
            **info,
            "sync_offset_ms": round(sync_offset * 1000, 2),
            "sync_ok": sync_offset < 0.1,
        }
    
    def _check_av_sync(self, clip_path: Path) -> float:
        """Check A/V sync offset."""
        # Get first video frame PTS
        cmd_v = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            "-read_intervals", "%+#1",
            str(clip_path)
        ]
        
        result_v = subprocess.run(cmd_v, capture_output=True, text=True)
        video_pts = float(result_v.stdout.strip()) if result_v.stdout.strip() else 0
        
        # Get first audio frame PTS
        cmd_a = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            "-read_intervals", "%+#1",
            str(clip_path)
        ]
        
        result_a = subprocess.run(cmd_a, capture_output=True, text=True)
        audio_pts = float(result_a.stdout.strip()) if result_a.stdout.strip() else 0
        
        return abs(video_pts - audio_pts)
