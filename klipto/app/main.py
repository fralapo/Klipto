
import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from typing import Optional, List
from pydantic import BaseModel

from app.config import settings
from app.tasks import process_video_pipeline, app as celery_app

app = FastAPI(title="Klipto API", version="0.1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7001", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory to serve generated clips
app.mount("/clips", StaticFiles(directory=settings.OUTPUT_DIR), name="clips")

class SettingsModel(BaseModel):
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = "deepseek/deepseek-r1"
    num_clips: int = 3

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    num_clips: int = 3
):
    """Uploads a video and starts the processing pipeline."""
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
         raise HTTPException(status_code=400, detail="Invalid file format")

    file_location = os.path.join(settings.UPLOAD_DIR, file.filename)

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # Start Celery Task
    task = process_video_pipeline.delay(
        video_path=file_location,
        options={"num_clips": num_clips}
    )

    return {"task_id": task.id, "message": "Video uploaded and processing started."}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Checks the status of a processing task."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }

    # Extract progress info if available
    if task_result.state == 'PROGRESS':
        # Nest progress info to match frontend interface
        response["progress"] = {
            "progress": task_result.info.get('progress', 0),
            "stage": task_result.info.get('stage', 'Processing')
        }
    else:
        # Optional: provide null or default for consistency
        response["progress"] = None

    return response

# --- Missing Endpoints for Frontend Compatibility ---

@app.get("/settings")
async def get_settings():
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "whisper_model": settings.WHISPER_MODEL,
        "api_key": "****************" # Masked
    }

@app.post("/settings")
async def update_settings(new_settings: SettingsModel):
    # In a real app, save to DB or .env. For now, just mock success.
    # Logic to update config runtime could go here.
    return {
        "message": "Settings updated",
        "config": new_settings.dict()
    }

@app.get("/tasks")
async def get_tasks():
    # Return mock history for now
    return []

@app.get("/")
async def root():
    return {"message": "Welcome to Klipto API"}
