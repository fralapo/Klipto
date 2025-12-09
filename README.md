# Klipto 🎬

**Klipto** is an open-source AI-powered video repurposing tool designed to automatically convert long-form horizontal videos into engaging vertical shorts (9:16) for platforms like TikTok, Instagram Reels, and YouTube Shorts.

It leverages a powerful stack of local AI models and open-source libraries to ensure high performance, privacy, and control, minimizing reliance on expensive external APIs.

<p align="center">
  <img src="docs/screenshot-home.png" alt="Klipto Home" width="800"/>
</p>

## ✨ Features

### 🤖 AI-Powered Processing
- **Local Audio Transcription**: Uses `faster-whisper` for lightning-fast, accurate speech-to-text on your own hardware (CPU/GPU)
- **Intelligent Scene Detection**: Uses `PySceneDetect` to automatically identify scene changes and cuts
- **Viral Hook Detection**: Analyzes transcripts using LLMs (DeepSeek, GPT-4, Claude, Gemini) to find engaging "hooks"
- **Smart Auto-Cropping**: Uses `YOLOv8/YOLO11` to detect subjects and dynamically crop to 9:16 while keeping subjects centered
- **Background Processing**: Robust pipeline using **Celery** and **Redis**

### 🎨 Modern Web Interface
- **Drag & Drop Upload**: Intuitive video upload with preview
- **Real-time Progress**: Visual timeline showing each processing stage
- **Clip Gallery**: Beautiful grid with virality scores and fullscreen preview
- **Settings Dashboard**: Configure AI providers and models through the UI
- **Dark Mode UI**: Modern glass-morphism design with smooth animations

## 🛠️ Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI 0.115+ |
| Task Queue | Celery 5.4+ |
| Cache/Broker | Redis 5.2+ |
| Video Processing | FFmpeg, OpenCV 4.10+ |

### AI / ML
| Component | Technology |
|-----------|------------|
| Transcription | faster-whisper 1.1+ (CTranslate2) |
| Object Detection | Ultralytics 8.3+ (YOLOv8/YOLO11) |
| Scene Detection | PySceneDetect 0.6.5+ |
| LLM Integration | OpenAI SDK 1.58+ |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 + TypeScript |
| Build Tool | Vite 5+ |
| Styling | Tailwind CSS 4 |
| Animations | Framer Motion |
| Icons | Lucide React |

### Supported LLM Providers
| Provider | Models |
|----------|--------|
| **OpenRouter** (recommended) | DeepSeek V3, Claude 3.5, GPT-4o, Gemini 2.0, Llama 3.3 |
| **OpenAI** | GPT-4o, GPT-4o-mini, o1-preview |
| **Anthropic** | Claude 3.5 Sonnet |
| **Google** | Gemini 2.0 Flash |

## 📁 Project Structure

```
klipto/
├── klipto/                 # Backend Python package
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── config.py       # Configuration settings
│   │   ├── tasks.py        # Celery tasks
│   │   ├── modules/        # AI processing modules
│   │   │   ├── audio_processor.py    # Whisper transcription
│   │   │   ├── vision_analyzer.py    # YOLO detection
│   │   │   ├── nlp_analyzer.py       # LLM hook detection
│   │   │   ├── clip_selector.py      # Clip scoring
│   │   │   └── video_composer.py     # FFmpeg rendering
│   │   └── utils/          # Utility functions
│   └── .env.example        # Environment template
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom hooks
│   │   ├── lib/            # API client
│   │   └── types/          # TypeScript types
│   └── package.json
├── requirements.txt        # Python dependencies
├── uploads/                # Uploaded videos
├── outputs/                # Generated clips
├── temp/                   # Temporary files
└── yolov8n.pt             # YOLO model weights
```

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **FFmpeg** installed on your system
- **Redis** server running
- **API Key** for your chosen LLM provider

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/fralapo/Klipto.git
cd Klipto
```

### 2. Backend Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt install ffmpeg redis-server

# macOS
brew install ffmpeg redis
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Configuration

Copy the example environment file:
```bash
cp klipto/.env.example klipto/.env
```

Edit `klipto/.env` with your settings:
```ini
# Choose LLM provider: openrouter, openai, anthropic, google
LLM_PROVIDER=openrouter
LLM_MODEL=deepseek/deepseek-chat

# Add your API key
OPENROUTER_API_KEY=sk-or-your-key-here

# Whisper model: tiny, base, small, medium, large-v3
WHISPER_MODEL=base
```

### 5. Start the Application

**Terminal 1 - Redis:**
```bash
redis-server
```

**Terminal 2 - Celery Worker:**
```bash
cd klipto
celery -A app.tasks worker --loglevel=info
```

**Terminal 3 - FastAPI Backend:**
```bash
cd klipto
uvicorn app.main:app --reload --port 8000
```

**Terminal 4 - React Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser 🎉

## 🎯 Usage

1. **Upload**: Drag & drop or click to select a horizontal video
2. **Configure**: Choose number of clips to generate (1-10)
3. **Process**: Click "Generate Viral Clips" and watch real-time progress
4. **Review**: Preview clips with virality scores
5. **Download**: Download your favorite clips

## 🔧 Configuration Reference

### Whisper Models
| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `tiny` | Fastest | Low | ~1GB |
| `base` | Fast | Good | ~1GB |
| `small` | Medium | Better | ~2GB |
| `medium` | Slow | High | ~5GB |
| `large-v3` | Slowest | Best | ~10GB |

### LLM Models (via OpenRouter)
| Model | Best For |
|-------|----------|
| `deepseek/deepseek-chat` | Best value, fast |
| `deepseek/deepseek-reasoner` | Complex reasoning |
| `anthropic/claude-3.5-sonnet` | High quality |
| `openai/gpt-4o` | Versatile |
| `google/gemini-2.0-flash-exp` | Fast, multimodal |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/upload` | POST | Upload video & start processing |
| `/status/{task_id}` | GET | Get task progress & results |
| `/clips/{filename}` | GET | Stream/download a clip |
| `/clips/{filename}` | DELETE | Delete a clip |
| `/tasks` | GET | List all tasks |
| `/settings` | GET/POST | Get/update settings |
| `/health` | GET | Health check |

## 🐳 Docker (Coming Soon)

```bash
docker-compose up -d
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Blazing fast transcription
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8/YOLO11
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Scene detection
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [DeepSeek](https://deepseek.com/) - Affordable & powerful LLMs

---

<p align="center">
  Made with ❤️ for content creators
</p>
