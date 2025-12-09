
import os
import json
from typing import List, Dict, Any
import openai
from app.config import settings

class NLPAnalyzer:
    """Analyzes transcript to identify interesting clips."""

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.LLM_MODEL
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)

    def identify_hooks(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifies viral hooks using LLM."""

        full_text = transcript.get('text', '')

        prompt = f"""
        Analyze the following video transcript and identify 3 potential viral highlights or "hooks".
        For each highlight, provide:
        - start_time (approximate in seconds)
        - end_time (approximate in seconds)
        - reason (why it is interesting)
        - score (0-100 virality score)

        Transcript:
        {full_text[:4000]} # Limit length

        Return ONLY a JSON array in this format:
        [
            {{"start": 10.5, "end": 25.0, "reason": "Funny joke", "score": 90}},
            ...
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional video editor helping to repurpose long videos into viral shorts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content

            # Extract JSON from response (in case of extra text)
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []

        except Exception as e:
            print(f"NLP Analysis failed: {str(e)}")
            if "Incorrect API key" in str(e) or "dummy" in self.api_key:
                return self._mock_hooks()
            return []

    def _mock_hooks(self):
        """Mock hooks for testing."""
        return [
            {"start": 0.0, "end": 5.0, "reason": "Intro hook", "score": 85},
            {"start": 5.0, "end": 10.0, "reason": "Key insight", "score": 90}
        ]
