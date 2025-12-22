"""
Video downloader supporting YouTube, Twitch, and local files.
Uses yt-dlp with optimized settings.
"""

import json
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from config import DOWNLOADS_DIR


class VideoSource(Enum):
    """Supported video sources."""
    LOCAL = "local"
    YOUTUBE = "youtube"
    TWITCH = "twitch"
    OTHER = "other"


@dataclass
class VideoInfo:
    """Video metadata."""
    id: str
    title: str
    duration: float
    source: VideoSource
    url: Optional[str]
    local_path: Optional[Path]
    thumbnail: Optional[str] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None
    formats: Optional[list] = None


class VideoDownloader:
    """Multi-source video downloader with caching."""
    
    def __init__(self, download_dir: Optional[Path] = None):
        self.download_dir = download_dir or DOWNLOADS_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.download_dir / ".cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load download cache."""
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save download cache."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _get_cache_key(self, source: str) -> str:
        """Generate cache key for source."""
        return hashlib.md5(source.encode()).hexdigest()
    
    def detect_source(self, source: str) -> VideoSource:
        """Detect video source type."""
        source_lower = source.lower()
        
        if Path(source).exists():
            return VideoSource.LOCAL
        elif "youtube.com" in source_lower or "youtu.be" in source_lower:
            return VideoSource.YOUTUBE
        elif "twitch.tv" in source_lower:
            return VideoSource.TWITCH
        else:
            return VideoSource.OTHER
    
    def get_info(self, source: str) -> VideoInfo:
        """Get video information without downloading."""
        source_type = self.detect_source(source)
        
        if source_type == VideoSource.LOCAL:
            return self._get_local_info(source)
        else:
            return self._get_remote_info(source, source_type)
    
    def _get_local_info(self, path: str) -> VideoInfo:
        """Get info for local file."""
        file_path = Path(path)
        
        # Get duration using ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path)
            ],
            capture_output=True,
            text=True
        )
        
        info = json.loads(result.stdout) if result.returncode == 0 else {}
        duration = float(info.get("format", {}).get("duration", 0))
        
        return VideoInfo(
            id=file_path.stem,
            title=file_path.stem,
            duration=duration,
            source=VideoSource.LOCAL,
            url=None,
            local_path=file_path,
        )
    
    def _get_remote_info(self, url: str, source_type: VideoSource) -> VideoInfo:
        """Get info for remote video."""
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                url
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        return VideoInfo(
            id=data.get("id", ""),
            title=data.get("title", ""),
            duration=data.get("duration", 0),
            source=source_type,
            url=url,
            local_path=None,
            thumbnail=data.get("thumbnail"),
            uploader=data.get("uploader"),
            upload_date=data.get("upload_date"),
            description=data.get("description"),
            formats=data.get("formats"),
        )
    
    def download(
        self,
        source: str,
        max_height: int = 1080,
        prefer_mp4: bool = True,
        extract_audio: bool = False,
        download_subtitles: bool = False,
        subtitle_langs: list[str] = None,
        force: bool = False,
        # Sections
        download_sections: Optional[str] = None,
        # Rate limiting
        limit_rate: Optional[str] = None,
        # SponsorBlock
        sponsorblock_remove: Optional[list[str]] = None,
    ) -> Path:
        """
        Download video with caching support.
        
        Args:
            source: Video URL or local path
            max_height: Maximum video height (1080, 720, 480)
            prefer_mp4: Prefer MP4 format
            extract_audio: Download audio only
            download_subtitles: Download subtitles
            subtitle_langs: Subtitle languages (e.g., ["en", "it"])
            force: Force re-download even if cached
            download_sections: Time range to download (e.g., "*10:00-15:00")
            limit_rate: Download rate limit (e.g., "5M" for 5MB/s)
            sponsorblock_remove: SponsorBlock categories to remove
        
        Returns:
            Path to downloaded file
        """
        source_type = self.detect_source(source)
        
        # Handle local files
        if source_type == VideoSource.LOCAL:
            return Path(source)
        
        # Check cache
        cache_key = self._get_cache_key(source)
        if not force and cache_key in self.cache:
            cached_path = Path(self.cache[cache_key]["path"])
            if cached_path.exists():
                return cached_path
        
        # Build yt-dlp command
        output_template = str(self.download_dir / "%(id)s.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "--output", output_template,
            "--no-playlist",
            "--no-mtime",
            "--write-info-json",
        ]
        
        # Format selection
        if extract_audio:
            cmd.extend([
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
            ])
        else:
            if prefer_mp4:
                format_str = f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[height<={max_height}]"
            else:
                format_str = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
            
            cmd.extend([
                "-f", format_str,
                "--merge-output-format", "mp4",
            ])
        
        # Subtitles
        if download_subtitles:
            cmd.append("--write-subs")
            cmd.append("--write-auto-subs")
            if subtitle_langs:
                cmd.extend(["--sub-langs", ",".join(subtitle_langs)])
            cmd.append("--convert-subs=srt")
        
        # Sections
        if download_sections:
            cmd.extend(["--download-sections", download_sections])
        
        # Rate limiting
        if limit_rate:
            cmd.extend(["--limit-rate", limit_rate])
        
        # SponsorBlock
        if sponsorblock_remove:
            cmd.extend([
                "--sponsorblock-remove", ",".join(sponsorblock_remove)
            ])
        
        cmd.append(source)
        
        # Execute download
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        
        # Find downloaded file
        info = self.get_info(source)
        expected_ext = "mp3" if extract_audio else "mp4"
        downloaded_path = self.download_dir / f"{info.id}.{expected_ext}"
        
        # Try to find the actual file if expected doesn't exist
        if not downloaded_path.exists():
            for ext in ["mp4", "webm", "mkv", "mp3", "m4a"]:
                test_path = self.download_dir / f"{info.id}.{ext}"
                if test_path.exists():
                    downloaded_path = test_path
                    break
        
        if not downloaded_path.exists():
            raise RuntimeError(f"Downloaded file not found for {info.id}")
        
        # Update cache
        self.cache[cache_key] = {
            "path": str(downloaded_path),
            "title": info.title,
            "duration": info.duration,
            "source": source_type.value,
        }
        self._save_cache()
        
        return downloaded_path
    
    def extract_audio_for_transcription(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """
        Extract audio optimized for transcription.
        
        Args:
            video_path: Path to video file
            output_path: Output audio path (default: same name with .wav)
            sample_rate: Audio sample rate (16000 for transcription)
            channels: Number of channels (1 for mono)
        
        Returns:
            Path to extracted audio
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = video_path.with_suffix(".wav")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
        
        return output_path
    
    def list_formats(self, url: str) -> list[dict]:
        """List available formats for a URL."""
        result = subprocess.run(
            [
                "yt-dlp",
                "--list-formats",
                "--dump-json",
                url
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to list formats: {result.stderr}")
        
        data = json.loads(result.stdout)
        return data.get("formats", [])
    
    def list_subtitles(self, url: str) -> dict:
        """List available subtitles for a URL."""
        result = subprocess.run(
            [
                "yt-dlp",
                "--list-subs",
                "--dump-json",
                url
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {}
        
        data = json.loads(result.stdout)
        return {
            "manual": data.get("subtitles", {}),
            "automatic": data.get("automatic_captions", {}),
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear download cache.
        
        Args:
            older_than_days: Only clear files older than this many days
        
        Returns:
            Number of files removed
        """
        import time
        
        removed = 0
        current_time = time.time()
        
        for cache_key, info in list(self.cache.items()):
            path = Path(info["path"])
            
            if older_than_days is not None:
                if path.exists():
                    file_age_days = (current_time - path.stat().st_mtime) / 86400
                    if file_age_days < older_than_days:
                        continue
            
            if path.exists():
                path.unlink()
                removed += 1
            
            # Also remove info json
            info_json = path.with_suffix(".info.json")
            if info_json.exists():
                info_json.unlink()
            
            del self.cache[cache_key]
        
        self._save_cache()
        return removed
    
    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        total_size = 0
        file_count = 0
        
        for info in self.cache.values():
            path = Path(info["path"])
            if path.exists():
                total_size += path.stat().st_size
                file_count += 1
        
        return {
            "file_count": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_entries": len(self.cache),
        }
