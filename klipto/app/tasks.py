
import os
import celery
from celery import Celery
from app.config import settings
from app.modules.video_processor import VideoProcessor
from app.modules.audio_processor import AudioProcessor
from app.modules.nlp_analyzer import NLPAnalyzer
from app.modules.vision_analyzer import VisionAnalyzer
from app.modules.clip_selector import ClipSelector
from app.modules.video_composer import VideoComposer

# Setup Celery
app = Celery('video_repurposing', broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@app.task(bind=True)
def process_video_pipeline(self, video_path: str, options: dict = None):
    """
    Full pipeline:
    1. Extract Audio & Frames
    2. Transcribe Audio
    3. Analyze Content (Hooks)
    4. Detect Faces (Vision)
    5. Select Best Clips
    6. Compose Final Vertical Videos
    """
    if options is None:
        options = {}

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    self.update_state(state='PROGRESS', meta={'stage': 'Analyzing Video', 'progress': 10})

    try:
        # 1. Video Processing
        print(f"Processing video: {video_path}")
        vp = VideoProcessor(video_path)
        metadata = vp.get_metadata()

        audio_path = os.path.join(settings.TEMP_DIR, f"{video_id}.wav")
        vp.extract_audio(audio_path)

        # Extract frames for vision analysis
        frames = vp.extract_frames(interval_seconds=2.0)

        # 2. Transcription
        self.update_state(state='PROGRESS', meta={'stage': 'Transcribing Audio', 'progress': 30})
        ap = AudioProcessor()
        transcript = ap.transcribe(audio_path)

        # 3. NLP Analysis
        self.update_state(state='PROGRESS', meta={'stage': 'Finding Highlights', 'progress': 50})
        nlp = NLPAnalyzer()
        hooks = nlp.identify_hooks(transcript)

        # 4. Vision Analysis
        self.update_state(state='PROGRESS', meta={'stage': 'Detecting Faces', 'progress': 70})
        va = VisionAnalyzer()
        face_data = va.detect_faces_in_frames(frames)

        # 5. Clip Selection
        cs = ClipSelector()
        best_clips = cs.score_clips(hooks, face_data)

        # Limit clips
        num_clips = options.get('num_clips', 3)
        selected_clips = best_clips[:num_clips]

        # 6. Composition
        self.update_state(state='PROGRESS', meta={'stage': 'Rendering Clips', 'progress': 85})
        vc = VideoComposer()

        results = []
        for i, clip in enumerate(selected_clips):
            # Determine crop based on faces in that timeframe
            relevant_faces = [
                f for f in face_data
                if clip['start'] <= f['timestamp'] <= clip['end']
            ]

            # Use middle frame's faces as heuristic or first frame
            faces_snapshot = []
            if relevant_faces:
                mid_index = len(relevant_faces) // 2
                faces_snapshot = relevant_faces[mid_index]['faces']

            crop = va.calculate_optimal_crop(
                (metadata['height'], metadata['width']),
                faces_snapshot
            )

            output_filename = f"{video_id}_clip_{i+1}.mp4"
            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

            vc.create_clip(
                video_path,
                clip['start'],
                clip['end'],
                crop,
                output_path
            )

            results.append({
                "id": i+1,
                "path": output_path,
                "score": clip.get('final_score'),
                "reason": clip.get('reason')
            })

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {"status": "completed", "clips": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
