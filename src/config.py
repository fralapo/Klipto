"""
Configuration for Shorts Generator.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DOWNLOADS_DIR = DATA_DIR / "downloads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ANALYSIS_DIR = DATA_DIR / "analysis"
CLIPS_DIR = DATA_DIR / "clips"

# Ensure directories exist
for dir_path in [DOWNLOADS_DIR, TRANSCRIPTS_DIR, ANALYSIS_DIR, CLIPS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Clip duration settings (seconds)
MIN_CLIP_DURATION = 15
MAX_CLIP_DURATION = 90
TARGET_CLIP_DURATION = 45

# Deepgram settings
# Deepgram settings
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("DEFAULT_TRANSCRIPTION_MODEL", "nova-2")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "it")

# LLM settings
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "deepseek/deepseek-chat")

# FFmpeg settings
DEFAULT_CUT_METHOD = "hybrid"  # fast, accurate, hybrid
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"
DEFAULT_AUDIO_BITRATE = "192k"

# Precision settings
ENABLE_PRECISION = True
VALIDATE_TIMESTAMPS = True
VAD_AGGRESSIVENESS = 2  # 0-3
MIN_SPEECH_RATIO = 0.3
BOUNDARY_SEARCH_WINDOW_MS = 50
BOUNDARY_PADDING_MS = 100
