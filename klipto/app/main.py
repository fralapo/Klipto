
import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from typing import Optional

from app.config import settings
from app.tasks import process_video_pipeline, app as celery_app

app = FastAPI(title="Klipto API", version="0.1.0")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    num_clips: int = 3
):
    """Uploads a video and starts the processing pipeline."""
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
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
        response["progress"] = task_result.info

    return response

@app.get("/")
async def root():
    return {"message": "Welcome to Klipto API"}
