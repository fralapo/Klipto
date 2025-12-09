
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any

class VisionAnalyzer:
    """Detects faces to help with auto-cropping."""

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, # 1 for longer range/videos
            min_detection_confidence=0.5
        )

    def detect_faces_in_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs face detection on extracted frames."""
        results = []

        for frame_data in frames:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']

            # MediaPipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            detection_result = self.face_detection.process(rgb_frame)

            faces = []
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.location_data.relative_bounding_box

                    # Convert to absolute pixels
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    faces.append({
                        'bbox': [x, y, width, height],
                        'confidence': detection.score[0]
                    })

            results.append({
                'timestamp': timestamp,
                'faces': faces,
                'num_faces': len(faces)
            })

        return results

    def calculate_optimal_crop(
        self,
        frame_shape: tuple,
        face_detections: List[Dict],
        target_aspect: float = 9/16
    ) -> Dict[str, int]:
        """Calculates 9:16 crop coordinates centering on faces."""
        h, w = frame_shape[:2]
        target_w = int(h * target_aspect)
        # If target width > video width, we might need to pad (letterbox) or just crop max width
        if target_w > w:
             # For simplicity, let's just take full width and crop height
             # But typical requirement is 9:16 vertical video from 16:9 horizontal
             # So we take a 9:16 slice of the 16:9 video.
             # Height is usually the constraining factor for resolution, but we want full height.
             target_w = int(h * target_aspect)

        # If faces found, center on them
        center_x = w // 2

        if face_detections:
            # Collect all face centers
            x_centers = []
            for face in face_detections:
                bbox = face['bbox']
                x_centers.append(bbox[0] + bbox[2] // 2)

            if x_centers:
                center_x = int(np.mean(x_centers))

        # Calculate crop window
        x_start = center_x - (target_w // 2)

        # Clamp to boundaries
        if x_start < 0:
            x_start = 0
        if x_start + target_w > w:
            x_start = w - target_w

        return {
            'x': x_start,
            'y': 0,
            'width': target_w,
            'height': h
        }
