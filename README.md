# Klipto 🎬

**Klipto** is an open-source AI-powered video repurposing tool designed to automatically convert long-form horizontal videos into engaging vertical shorts (9:16) for platforms like TikTok, Instagram Reels, and YouTube Shorts.

It leverages a powerful stack of local AI models and open-source libraries to ensure high performance, privacy, and control, minimizing reliance on expensive external APIs.

![Klipto Screenshot](docs/screenshot.png)

## 🚀 Features

### AI-Powered Processing
- **Local Audio Transcription**: Uses `faster-whisper` for lightning-fast, accurate speech-to-text processing on your own hardware (CPU/GPU)
- **Intelligent Scene Detection**: Utilizes `PySceneDetect` to automatically identify scene changes and cuts in the video
- **Viral Hook Detection**: Analyzes transcripts using LLMs (via OpenRouter/DeepSeek/OpenAI) to find the most engaging "hooks" or highlights
- **Smart Auto-Cropping**: Uses `YOLOv8` (Ultralytics) to detect subjects (people) and dynamically crop horizontal video to vertical (9:16) while keeping the subject centered
- **Background Processing**: Robust pipeline orchestration using **Celery** and **Redis**

### Modern Web Interface
- **Drag & Drop Upload**: Intuitive video upload with preview
- **Real-time Progress Tracking**: Visual timeline showing each processing stage
- **Clip Gallery**: Beautiful grid view of generated clips with virality scores
- **Fullscreen Preview**: Modal player with navigation between clips
- **Settings Dashboard**: Configure AI providers and models through the UI
- **Dark Mode UI**: Modern glass-morphism design with smooth animations

## 🛠️ Tech Stack

### Backend
- **Framework**: Python 3.10+, FastAPI
- **Task Queue**: Celery, Redis
- **Video Processing**: FFmpeg, OpenCV

### AI / ML
- **Transcription**: `faster-whisper` (CTranslate2)
- **Vision/Object Detection**: `ultralytics` (YOLOv8 Nano)
- **Scene Detection**: `PySceneDetect`
- **LLM Integration**: OpenRouter, OpenAI, Anthropic

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS 4
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **HTTP Client**: Axios

## 📁 Project Structure

```
klipto/
├── klipto/                 # Backend Python package
│   └── app/
│       ├── main.py         # FastAPI application
│       ├── config.py       # Configuration settings
│       ├── tasks.py        # Celery tasks
│       ├── modules/        # AI processing modules
│       │   ├── audio_processor.py    # Whisper transcription
│       │   ├── vision_analyzer.py    # YOLO detection
│       │   ├── nlp_analyzer.py       # LLM hook detection
│       │   ├── clip_selector.py      # Clip scoring
│       │   └── video_composer.py     # FFmpeg rendering
│       └── utils/          # Utility functions
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ui/         # Button, Card, Progress, Badge
│   │   │   ├── layout/     # Header, Layout
│   │   │   ├── upload/     # VideoDropzone
│   │   │   ├── processing/ # ProcessingStatus
│   │   │   └── clips/      # ClipCard, ClipGrid
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── lib/            # API client
│   │   └── types/          # TypeScript types
│   └── package.json
├── uploads/                # Uploaded videos
├── outputs/                # Generated clips
├── temp/                   # Temporary files
└── yolov8n.pt             # YOLO model weights
```

## 📋 Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- `ffmpeg` installed on your system
- `redis-server` installed and running
- An API Key for OpenRouter (or OpenAI/Anthropic) for the content analysis step

## ⚡ Installation

### 1. Clone the repository
```bash
git clone https://github.com/fralapo/Klipto.git
cd Klipto
```

### 2. Backend Setup
```bash
# Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python Dependencies
pip install -r requirements.txt

# Ensure you have system dependencies:
# Ubuntu/Debian: sudo apt install ffmpeg redis-server
# macOS: brew install ffmpeg redis
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Configuration
Create a `.env` file in the `klipto/` directory:

```ini
# .env
PROJECT_NAME="Klipto"

# Redis
REDIS_URL=redis://localhost:6379/0

# AI Config - Choose your provider
LLM_PROVIDER=openrouter  # Options: openrouter, openai, anthropic
LLM_MODEL=deepseek/deepseek-r1

# API Keys (add the one for your chosen provider)
OPENROUTER_API_KEY=sk-or-your-key-here
# OPENAI_API_KEY=sk-your-openai-key
# ANTHROPIC_API_KEY=sk-ant-your-key

# Local Whisper Config
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
USE_LOCAL_WHISPER=true
```

Create a `.env` file in the `frontend/` directory:
```ini
VITE_API_URL=http://localhost:8000
```

## 🏃 Usage

### Quick Start (Development)

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

Then open your browser to `http://localhost:3000` 🎉

### Using the Interface

1. **Upload a Video**: Drag & drop or click to select a horizontal video file
2. **Select Clip Count**: Choose how many clips to generate (1-10)
3. **Click "Genera Clip Virali"**: Start the AI processing pipeline
4. **Monitor Progress**: Watch real-time updates as each stage completes
5. **Preview & Download**: View generated clips and download your favorites

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/upload` | POST | Upload video and start processing |
| `/status/{task_id}` | GET | Get task progress and results |
| `/clips/{filename}` | GET | Stream/download a clip |
| `/clips/{filename}` | DELETE | Delete a clip |
| `/tasks` | GET | List all processing tasks |
| `/settings` | GET | Get current settings |
| `/settings` | POST | Update settings |
| `/health` | GET | Health check |

## 🖥️ Screenshots

### Home Page - Upload Interface
Modern drag & drop interface with video preview and clip count selection.

### Processing View
Real-time progress tracking with visual timeline of all AI processing stages.

### Results Gallery
Beautiful grid of generated clips with virality scores, preview on hover, and one-click download.

### Settings Page
Configure LLM providers, models, API keys, and Whisper settings through the UI.

## 🔧 Configuration Options

### LLM Providers
- **OpenRouter**: Access to DeepSeek, Claude, GPT, and many more models
- **OpenAI**: Direct GPT-4 and GPT-3.5 access
- **Anthropic**: Direct Claude access

### Whisper Models
| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| tiny | Fastest | Low | ~1GB |
| base | Fast | Good | ~1GB |
| small | Medium | Better | ~2GB |
| medium | Slow | High | ~5GB |
| large | Slowest | Best | ~10GB |

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

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for blazing fast transcription
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection
- [FFmpeg](https://ffmpeg.org/) for video processing

---

Made with ❤️ by the Klipto team
