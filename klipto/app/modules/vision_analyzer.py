
import cv2
import numpy as np
from typing import List, Dict, Any
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO

class VideoProcessor:
    """Handles Video I/O and Scene Detection."""

    def __init__(self, video_path: str):
        self.video_path = video_path

    def detect_scenes(self, threshold: float = 30.0) -> List[Dict[str, float]]:
        """Detects scenes using PySceneDetect."""
        print(f"Detecting scenes in {self.video_path}...")
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # Detect scenes
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()

        # If no scenes detected (no cuts), assume whole video is one scene
        if not scene_list:
            duration = video.duration.get_seconds()
            scene_list = [(video.base_timecode, video.duration)]

        # Format output
        scenes = []
        for i, scene in enumerate(scene_list):
            start, end = scene
            scenes.append({
                'id': i,
                'start': start.get_seconds(),
                'end': end.get_seconds(),
                'duration': end.get_seconds() - start.get_seconds()
            })
        return scenes

class VisionAnalyzer:
    """Detects subjects using YOLOv8."""

    def __init__(self):
        # Load YOLOv8 Nano model (pretrained on COCO)
        # Suppress verbose output
        self.model = YOLO("yolov8n.pt")

    def analyze_frame_for_crop(self, frame: np.ndarray, target_aspect: float = 9/16) -> Dict[str, int]:
        """
        Detects person class (id 0) and returns crop coordinates.
        If multiple people, tries to center the group or the largest one.
        """
        results = self.model(frame, verbose=False)
        result = results[0]

        # Filter for 'person' class (class_id == 0)
        person_boxes = []
        for box in result.boxes:
            if int(box.cls) == 0: # 0 is person in COCO
                # box.xyxy is [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy()
                person_boxes.append(xyxy)

        h, w = frame.shape[:2]
        target_w = int(h * target_aspect)

        # Center logic
        if person_boxes:
            # Find center of all people
            x_min = min(b[0] for b in person_boxes)
            x_max = max(b[2] for b in person_boxes)
            center_x = (x_min + x_max) / 2
        else:
            # Fallback to center of frame
            center_x = w / 2

        # Calculate crop x_start
        x_start = int(center_x - (target_w / 2))

        # Boundary checks
        if x_start < 0: x_start = 0
        if x_start + target_w > w: x_start = w - target_w

        return {
            'x': x_start,
            'y': 0,
            'width': target_w,
            'height': h
        }
