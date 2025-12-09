import os
import shutil
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
from pydantic import BaseModel

from app.config import settings
from app.tasks import process_video_pipeline, app as celery_app

# Pydantic Models
class SettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    whisper_model: Optional[str] = None
    use_local_whisper: Optional[bool] = None

class SettingsResponse(BaseModel):
    llm_provider: str
    llm_model: str
    whisper_model: str
    use_local_whisper: bool
    has_api_key: bool

class TaskHistoryItem(BaseModel):
    id: str
    filename: str
    date: str
    status: str
    clips_count: int
    duration: Optional[str] = None

# App setup
app = FastAPI(
    title="Klipto API",
    version="0.1.0",
    description="AI-powered video repurposing API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (output clips)
if os.path.exists(settings.OUTPUT_DIR):
    app.mount("/clips", StaticFiles(directory=settings.OUTPUT_DIR), name="clips")

# In-memory task history (in production, use a database)
task_history: dict = {}

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Klipto API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    num_clips: int = Query(default=3, ge=1, le=10)
):
    """
    Uploads a video and starts the AI processing pipeline.
    
    - **file**: Video file (mp4, mov, avi, mkv, webm)
    - **num_clips**: Number of clips to generate (1-10)
    """
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported: {', '.join(valid_extensions)}"
        )

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_location = os.path.join(settings.UPLOAD_DIR, safe_filename)

    # Save file
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Start Celery Task
    task = process_video_pipeline.delay(
        video_path=file_location,
        options={"num_clips": num_clips}
    )

    # Store in history
    task_history[task.id] = {
        "id": task.id,
        "filename": file.filename,
        "date": datetime.now().isoformat(),
        "status": "processing",
        "clips_count": 0
    }

    return {
        "task_id": task.id,
        "message": "Video uploaded and processing started.",
        "filename": file.filename
    }

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Checks the status of a processing task.
    
    Returns current progress, status, and results when complete.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None,
        "progress": None,
        "error": None
    }

    # Extract progress info if available
    if task_result.state == 'PROGRESS':
        response["progress"] = task_result.info
    elif task_result.state == 'SUCCESS':
        result = task_result.result
        response["result"] = result
        
        # Update history
        if task_id in task_history:
            task_history[task_id]["status"] = "completed"
            task_history[task_id]["clips_count"] = len(result.get("clips", []))
            
        # Add URLs to clips
        if result and "clips" in result:
            for clip in result["clips"]:
                filename = os.path.basename(clip["path"])
                clip["url"] = f"/clips/{filename}"
                
    elif task_result.state == 'FAILURE':
        response["error"] = str(task_result.result)
        if task_id in task_history:
            task_history[task_id]["status"] = "failed"

    return response

@app.get("/clips/{filename}")
async def get_clip(filename: str):
    """Serves a generated clip file."""
    file_path = os.path.join(settings.OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )

@app.delete("/clips/{filename}")
async def delete_clip(filename: str):
    """Deletes a generated clip."""
    file_path = os.path.join(settings.OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    
    try:
        os.remove(file_path)
        return {"message": "Clip deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@app.get("/tasks", response_model=List[TaskHistoryItem])
async def get_tasks():
    """Returns the list of all tasks."""
    return list(task_history.values())

@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Returns current settings (without sensitive data)."""
    return SettingsResponse(
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        whisper_model=settings.WHISPER_MODEL,
        use_local_whisper=settings.USE_LOCAL_WHISPER,
        has_api_key=bool(settings.OPENAI_API_KEY or settings.OPENROUTER_API_KEY)
    )

@app.post("/settings", response_model=SettingsResponse)
async def update_settings(updates: SettingsUpdate):
    """
    Updates settings.
    
    Note: In production, this should update the .env file or database.
    For now, it updates the in-memory settings object.
    """
    if updates.llm_provider:
        settings.LLM_PROVIDER = updates.llm_provider
    if updates.llm_model:
        settings.LLM_MODEL = updates.llm_model
    if updates.whisper_model:
        settings.WHISPER_MODEL = updates.whisper_model
    if updates.use_local_whisper is not None:
        settings.USE_LOCAL_WHISPER = updates.use_local_whisper
    if updates.api_key:
        if updates.llm_provider == "openrouter":
            settings.OPENROUTER_API_KEY = updates.api_key
        else:
            settings.OPENAI_API_KEY = updates.api_key
    
    return SettingsResponse(
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        whisper_model=settings.WHISPER_MODEL,
        use_local_whisper=settings.USE_LOCAL_WHISPER,
        has_api_key=bool(settings.OPENAI_API_KEY or settings.OPENROUTER_API_KEY)
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": True,
            "uploads_dir": os.path.exists(settings.UPLOAD_DIR),
            "outputs_dir": os.path.exists(settings.OUTPUT_DIR),
        }
    }

# Cleanup endpoint for development
@app.post("/cleanup")
async def cleanup_files():
    """Cleans up temporary and output files. Use with caution."""
    from app.utils.file_utils import clear_directory
    
    try:
        clear_directory(settings.TEMP_DIR)
        clear_directory(settings.OUTPUT_DIR)
        clear_directory(settings.UPLOAD_DIR)
        task_history.clear()
        return {"message": "Cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
