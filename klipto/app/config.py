
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "Klipto"
    VERSION = "0.1.0"

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR = os.path.join(os.path.dirname(BASE_DIR), "uploads")
    OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "outputs")
    TEMP_DIR = os.path.join(os.path.dirname(BASE_DIR), "temp")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./video_repurposing.db")

    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Model Config
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "false").lower() == "true"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1") # Fallback default

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
