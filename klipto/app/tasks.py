
import os
import celery
from celery import Celery
import cv2
from app.config import settings
from app.modules.audio_processor import AudioProcessor
from app.modules.vision_analyzer import VideoProcessor, VisionAnalyzer
from app.modules.nlp_analyzer import NLPAnalyzer
from app.modules.clip_selector import ClipSelector
from app.modules.video_composer import VideoComposer

# Setup Celery
app = Celery('video_repurposing', broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@app.task(bind=True)
def process_video_pipeline(self, video_path: str, options: dict = None):
    """
    New Pipeline:
    1. Extract Audio
    2. Transcribe (Faster-Whisper)
    3. Detect Scenes (PySceneDetect)
    4. NLP Analysis (GPT) to find Hooks (mapped to time)
    5. Filter Scenes based on Hooks
    6. For each selected Scene:
       - Analyze representative frame with YOLO
       - Calculate Crop
       - Render Clip
    """
    if options is None:
        options = {}

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    self.update_state(state='PROGRESS', meta={'stage': 'Initializing', 'progress': 0})

    try:
        # 1. Audio Extraction
        self.update_state(state='PROGRESS', meta={'stage': 'Extracting Audio', 'progress': 10})
        # Use simple ffmpeg command or a util, but for now let's reuse VideoProcessor logic?
        # Wait, I deleted the old VideoProcessor which had extract_audio.
        # I need to implement extract_audio helper or put it back in the new VideoProcessor.
        # I will do it inline or add to VideoProcessor.

        # Helper for audio
        import ffmpeg
        audio_path = os.path.join(settings.TEMP_DIR, f"{video_id}.wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)

        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True)
        )

        # 2. Transcription
        self.update_state(state='PROGRESS', meta={'stage': 'Transcribing (Whisper)', 'progress': 20})
        ap = AudioProcessor()
        transcript = ap.transcribe(audio_path)

        # 3. Scene Detection
        self.update_state(state='PROGRESS', meta={'stage': 'Detecting Scenes', 'progress': 40})
        vp = VideoProcessor(video_path)
        scenes = vp.detect_scenes()

        # 4. NLP Analysis (Hooks)
        self.update_state(state='PROGRESS', meta={'stage': 'Analyzing Content', 'progress': 60})
        nlp = NLPAnalyzer()
        hooks = nlp.identify_hooks(transcript)

        # 5. Map Hooks to Scenes (Simple intersection)
        selected_scenes = []
        for hook in hooks:
            hook_start = hook['start']
            hook_end = hook['end']

            # Find scene that contains the hook center or overlaps significantly
            hook_center = (hook_start + hook_end) / 2

            best_scene = None
            for scene in scenes:
                if scene['start'] <= hook_center <= scene['end']:
                    best_scene = scene
                    break

            if best_scene:
                # Add metadata from hook to scene
                scene_copy = best_scene.copy()
                scene_copy['score'] = hook.get('score', 0)
                scene_copy['reason'] = hook.get('reason', '')
                selected_scenes.append(scene_copy)

        # Deduplicate scenes
        unique_scenes = []
        seen_ids = set()
        for s in selected_scenes:
            if s['id'] not in seen_ids:
                unique_scenes.append(s)
                seen_ids.add(s['id'])

        # Limit count
        num_clips = options.get('num_clips', 3)
        final_scenes = unique_scenes[:num_clips]

        # 6. Render
        self.update_state(state='PROGRESS', meta={'stage': 'Rendering Clips', 'progress': 80})
        va = VisionAnalyzer()
        vc = VideoComposer()

        results = []
        cap = cv2.VideoCapture(video_path)

        for i, scene in enumerate(final_scenes):
            # Extract middle frame for crop analysis
            mid_time = (scene['start'] + scene['end']) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()

            crop = {'x': 0, 'y': 0, 'width': 720, 'height': 1280} # Default
            if ret:
                crop = va.analyze_frame_for_crop(frame)

            output_filename = f"{video_id}_clip_{i+1}.mp4"
            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

            vc.create_clip(
                video_path,
                scene['start'],
                scene['end'],
                crop,
                output_path
            )

            results.append({
                "id": i+1,
                "path": output_path,
                "score": scene.get('score'),
                "reason": scene.get('reason'),
                "duration": scene.get('duration')
            })

        cap.release()

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {"status": "completed", "clips": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
