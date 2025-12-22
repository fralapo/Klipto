"""
Deepgram Model Definitions
Source: Deepgram Documentation (Dec 2025)
"""

DEEPGRAM_MODEL_FAMILIES = {
    "Nova-3": {
        "description": "Latest generation. Highest accuracy, speed, and cost-effectiveness.",
        "models": {
            "nova-3": {"desc": "Standard (Recommended for most cases)", "lang": "all"},
            "nova-3-general": {"desc": "Optimized for everyday audio", "lang": "all"},
            "nova-3-medical": {"desc": "Optimized for medical terminology", "lang": "en"}
        }
    },
    "Nova-2": {
        "description": "Previous generation. Highly specialized domain variants.",
        "models": {
            "nova-2": {"desc": "Standard Nova-2", "lang": "all"},
            "nova-2-general": {"desc": "General purpose", "lang": "all"},
            "nova-2-meeting": {"desc": "Meeting analysis", "lang": "en"},
            "nova-2-phonecall": {"desc": "Phone calls", "lang": "en"},
            "nova-2-voicemail": {"desc": "Voicemail", "lang": "en"},
            "nova-2-finance": {"desc": "Financial terminology", "lang": "en"},
            "nova-2-conversationalai": {"desc": "Human-bot interactions", "lang": "en"},
            "nova-2-video": {"desc": "Video audio", "lang": "en"},
            "nova-2-medical": {"desc": "Medical terminology", "lang": "en"},
            "nova-2-drivethru": {"desc": "Drive-thru orders", "lang": "en"},
            "nova-2-automotive": {"desc": "Automotive context", "lang": "en"},
            "nova-2-atc": {"desc": "Air Traffic Control", "lang": "en"}
        }
    },
    "Nova": {
        "description": "Legacy high-accuracy models.",
        "models": {
            "nova-general": {"desc": "General purpose legacy", "lang": "all"},
            "nova-phonecall": {"desc": "Phone call legacy", "lang": "en"}
        }
    },
    "Enhanced": {
        "description": "High accuracy on rare words.",
        "models": {
            "enhanced-general": {"desc": "General purpose", "lang": "en"},
            "enhanced-meeting": {"desc": "Meeting (Beta)", "lang": "en"},
            "enhanced-phonecall": {"desc": "Phone call", "lang": "en"},
            "enhanced-finance": {"desc": "Finance (Beta)", "lang": "en"}
        }
    },
    "Base": {
        "description": "Cost-effective, standard accuracy.",
        "models": {
            "base-general": {"desc": "General purpose", "lang": "en"},
            "base-meeting": {"desc": "Meetings", "lang": "en"},
            "base-phonecall": {"desc": "Phone calls", "lang": "en"},
            "base-voicemail": {"desc": "Voicemails", "lang": "en"},
            "base-finance": {"desc": "Finance", "lang": "en"},
            "base-conversationalai": {"desc": "Conversational AI", "lang": "en"},
            "base-video": {"desc": "Video", "lang": "en"}
        }
    },
    "Whisper": {
        "description": "OpenAI Whisper via Deepgram Cloud.",
        "models": {
            "whisper": {"desc": "Default Whisper (Medium)", "lang": "all"},
            "whisper-tiny": {"desc": "Tiny (Fastest)", "lang": "all"},
            "whisper-base": {"desc": "Base", "lang": "all"},
            "whisper-small": {"desc": "Small", "lang": "all"},
            "whisper-medium": {"desc": "Medium", "lang": "all"},
            "whisper-large": {"desc": "Large (Highest Accuracy)", "lang": "all"}
        }
    }
}

# Flattened dictionary for backward compatibility (simple description)
DEEPGRAM_MODELS = {}
for family, data in DEEPGRAM_MODEL_FAMILIES.items():
    for model_id, info in data["models"].items():
        DEEPGRAM_MODELS[model_id] = info["desc"]

# Helper to check language support
def is_model_supported(model_id: str, language: str) -> bool:
    """Check if model supports the language."""
    for family, data in DEEPGRAM_MODEL_FAMILIES.items():
        if model_id in data["models"]:
            supported = data["models"][model_id]["lang"]
            if supported == "all": return True
            return language.lower().startswith(supported)
    return True # Default to True if unknown
