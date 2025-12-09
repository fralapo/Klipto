
import cv2
import ffmpeg
import json
from pathlib import Path
from typing import Dict, List, Any

class VideoProcessor:
    """Extracts metadata, frames and audio from videos."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def get_metadata(self) -> Dict[str, Any]:
        """Extracts video metadata using ffmpeg-python."""
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

            # Handle rotation if present
            tags = video_info.get('tags', {})
            rotate = int(tags.get('rotate', 0))

            width = int(video_info['width'])
            height = int(video_info['height'])

            if rotate in [90, 270]:
                width, height = height, width

            return {
                'duration': float(probe['format']['duration']),
                'width': width,
                'height': height,
                'fps': eval(video_info.get('r_frame_rate', '30/1')),
                'codec': video_info['codec_name'],
                'bitrate': int(probe['format']['bit_rate']) if 'bit_rate' in probe['format'] else None
            }
        except ffmpeg.Error as e:
            print(f"Error probing video: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def extract_audio(self, output_path: str) -> str:
        """Extracts audio as WAV/MP3."""
        try:
            (
                ffmpeg
                .input(str(self.video_path))
                .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def extract_frames(self, interval_seconds: float = 2.0) -> List[Dict[str, Any]]:
        """Extracts frames at regular intervals for analysis."""
        frames = []
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0 # Fallback

        frame_interval = int(fps * interval_seconds)
        if frame_interval == 0:
            frame_interval = 1

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                frames.append({
                    'timestamp': count / fps,
                    'frame': frame
                })

            count += 1

        cap.release()
        return frames
