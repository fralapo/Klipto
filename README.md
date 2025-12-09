
# Klipto 🎬

**Klipto** is an open-source AI-powered video repurposing tool designed to automatically convert long-form horizontal videos into engaging vertical shorts (9:16) for platforms like TikTok, Instagram Reels, and YouTube Shorts.

It leverages a powerful stack of local AI models and open-source libraries to ensure high performance, privacy, and control, minimizing reliance on expensive external APIs.

## 🚀 Features

*   **Local Audio Transcription**: Uses `faster-whisper` for lightning-fast, accurate speech-to-text processing on your own hardware (CPU/GPU).
*   **Intelligent Scene Detection**: Utilizes `PySceneDetect` to automatically identify scene changes and cuts in the video.
*   **Viral Hook Detection**: Analyzes transcripts using LLMs (via OpenRouter/DeepSeek) to find the most engaging "hooks" or highlights.
*   **Smart Auto-Cropping**: Uses `YOLOv8` (Ultralytics) to detect subjects (people) and dynamically crop horizontal video to vertical (9:16) while keeping the subject centered.
*   **Background Processing**: Robust pipeline orchestration using **Celery** and **Redis**.
*   **API-First Design**: Built with **FastAPI**, ready to be integrated with modern frontends (Shadcn UI + TailwindCSS planned).

## 🛠️ Tech Stack

*   **Backend Framework**: Python 3.10+, FastAPI
*   **Task Queue**: Celery, Redis
*   **AI / ML**:
    *   **Transcription**: `faster-whisper` (CTranslate2)
    *   **Vision/Object Detection**: `ultralytics` (YOLOv8 Nano)
    *   **Scene Detection**: `PySceneDetect`
    *   **LLM Integration**: OpenRouter API (DeepSeek V3/R1)
*   **Video Processing**: `FFmpeg`, `OpenCV`

## 📋 Prerequisites

*   Python 3.10 or higher
*   `ffmpeg` installed on your system
*   `redis-server` installed and running
*   An API Key for OpenRouter (or OpenAI) for the content analysis step.

## ⚡ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/klipto.git
    cd klipto
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    # Ensure you have system dependencies:
    # sudo apt install ffmpeg redis-server
    ```

4.  **Configuration**
    Create a `.env` file in the root directory (or rename `.env.example`):

    ```ini
    # .env
    PROJECT_NAME="Klipto"

    # Redis
    REDIS_URL=redis://localhost:6379/0

    # AI Config
    OPENROUTER_API_KEY=sk-or-your-key-here
    LLM_PROVIDER=openrouter
    LLM_MODEL=deepseek/deepseek-r1

    # Local Whisper Config
    WHISPER_MODEL=base
    USE_LOCAL_WHISPER=true
    ```

## 🏃 Usage

1.  **Start Redis**
    ```bash
    redis-server --daemonize yes
    ```

2.  **Start the Celery Worker**
    ```bash
    # Run from the project root
    celery -A app.tasks worker --loglevel=info
    ```

3.  **Start the FastAPI Server**
    ```bash
    uvicorn app.main:app --reload
    ```

4.  **Access the API**
    Open your browser to `http://localhost:8000/docs` to see the interactive API documentation.

    *   **POST /upload**: Upload a video file to start the pipeline.
    *   **GET /status/{task_id}**: Check the progress of a processing task.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License
