
from typing import List, Dict, Any

class ClipSelector:
    """Selects the best clips based on NLP and Vision scores."""

    def score_clips(
        self,
        hooks: List[Dict],
        face_data: List[Dict]
    ) -> List[Dict]:
        """Merges hooks with face availability data."""
        scored_clips = []

        for hook in hooks:
            start = hook.get('start', 0)
            end = hook.get('end', 0)

            # Check if we have faces in this segment
            # face_data is list of {timestamp, num_faces...}
            # We want to boost score if there are faces (talking head)

            relevant_frames = [
                f for f in face_data
                if start <= f['timestamp'] <= end
            ]

            face_score = 0
            if relevant_frames:
                frames_with_faces = sum(1 for f in relevant_frames if f['num_faces'] > 0)
                face_score = frames_with_faces / len(relevant_frames)

            # Final score mix
            base_score = hook.get('score', 50)
            final_score = base_score + (face_score * 20) # Bonus for faces

            scored_clips.append({
                **hook,
                'final_score': final_score,
                'has_faces': face_score > 0.5
            })

        return sorted(scored_clips, key=lambda x: x['final_score'], reverse=True)
