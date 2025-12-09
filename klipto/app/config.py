"""
Klipto Configuration Module
Handles all application settings via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # =========================================
    # Project Info
    # =========================================
    PROJECT_NAME = os.getenv("PROJECT_NAME", "Klipto")
    VERSION = "0.2.0"
    DESCRIPTION = "AI-powered video repurposing tool"
    
    # =========================================
    # Paths
    # =========================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    ROOT_DIR = BASE_DIR.parent  # Project root (contains uploads, outputs, etc.)
    
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(ROOT_DIR / "uploads"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs"))
    TEMP_DIR = os.getenv("TEMP_DIR", str(ROOT_DIR / "temp"))
    
    # =========================================
    # Database (optional, for future use)
    # =========================================
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./klipto.db")
    
    # =========================================
    # Redis & Celery
    # =========================================
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # =========================================
    # API Keys
    # =========================================
    # OpenRouter (recommended - access to multiple models)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # OpenAI Direct
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Anthropic Direct (or via OpenRouter)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Google AI (or via OpenRouter)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # =========================================
    # LLM Configuration
    # =========================================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
    # Options: openrouter, openai, anthropic, google
    
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
    # Recommended models per provider:
    # - openrouter: deepseek/deepseek-chat, anthropic/claude-3.5-sonnet, openai/gpt-4o
    # - openai: gpt-4o, gpt-4o-mini, gpt-4-turbo
    # - anthropic: claude-3-5-sonnet-20241022
    # - google: gemini-2.0-flash-exp
    
    # =========================================
    # Whisper Configuration
    # =========================================
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    # Options: tiny, base, small, medium, large-v3
    # Larger models are more accurate but slower and require more VRAM
    
    USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "true").lower() == "true"
    # Set to false to use a cloud transcription API (future feature)
    
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
    # Options: auto, cpu, cuda
    
    # =========================================
    # Video Processing
    # =========================================
    DEFAULT_NUM_CLIPS = int(os.getenv("DEFAULT_NUM_CLIPS", "3"))
    MIN_CLIP_DURATION = float(os.getenv("MIN_CLIP_DURATION", "15.0"))
    MAX_CLIP_DURATION = float(os.getenv("MAX_CLIP_DURATION", "60.0"))
    
    OUTPUT_RESOLUTION = os.getenv("OUTPUT_RESOLUTION", "720x1280")  # 9:16 vertical
    OUTPUT_BITRATE = os.getenv("OUTPUT_BITRATE", "2M")
    OUTPUT_CODEC = os.getenv("OUTPUT_CODEC", "libx264")
    
    # =========================================
    # YOLO Configuration
    # =========================================
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    # Or YOLO11: yolo11n.pt, yolo11s.pt, yolo11m.pt
    
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
    
    # =========================================
    # Scene Detection
    # =========================================
    SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "30.0"))
    # Lower = more sensitive (detects more scene changes)
    
    # =========================================
    # Server Configuration
    # =========================================
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # =========================================
    # Upload Limits
    # =========================================
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))  # 500MB
    ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)


def print_config():
    """Prints current configuration (for debugging)."""
    print("\n" + "="*50)
    print("KLIPTO CONFIGURATION")
    print("="*50)
    print(f"Project: {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"Whisper Model: {settings.WHISPER_MODEL}")
    print(f"YOLO Model: {settings.YOLO_MODEL}")
    print(f"Redis URL: {settings.REDIS_URL}")
    print(f"Upload Dir: {settings.UPLOAD_DIR}")
    print(f"Output Dir: {settings.OUTPUT_DIR}")
    print("="*50 + "\n")
