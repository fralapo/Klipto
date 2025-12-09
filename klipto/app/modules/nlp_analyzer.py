
import os
import json
from typing import List, Dict, Any
import openai
from app.config import settings

class NLPAnalyzer:
    """Analyzes transcript to identify interesting clips using LLM."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

        if self.provider == "openrouter":
            self.api_key = settings.OPENROUTER_API_KEY
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = settings.LLM_MODEL or "deepseek/deepseek-r1" # Default to R1/V3
        else:
            # Fallback to OpenAI
            self.api_key = settings.OPENAI_API_KEY
            self.base_url = None
            self.model = settings.LLM_MODEL or "gpt-4-turbo"

        if self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None

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
        {full_text[:4000]} # Limit length to fit context

        Return ONLY a JSON array in this format:
        [
            {{"start": 10.5, "end": 25.0, "reason": "Funny joke", "score": 90}},
            ...
        ]
        """

        if not self.client:
             print("No API Key configured. Returning mock hooks.")
             return self._mock_hooks()

        try:
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://klipto.ai", # Optional for OpenRouter
                    "X-Title": "Klipto"
                }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional video editor helping to repurpose long videos into viral shorts. Output strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                extra_headers=extra_headers
            )
            content = response.choices[0].message.content

            # Clean content (remove markdown code blocks if DeepSeek adds them)
            content = content.replace("```json", "").replace("```", "").strip()

            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Attempt direct parse if regex fails
            return json.loads(content)

        except Exception as e:
            print(f"NLP Analysis failed: {str(e)}")
            # Fallback for dev/testing
            return self._mock_hooks()

    def _mock_hooks(self):
        """Mock hooks for testing."""
        return [
            {"start": 0.0, "end": 5.0, "reason": "Intro hook (Mock)", "score": 85},
            {"start": 5.0, "end": 10.0, "reason": "Key insight (Mock)", "score": 90}
        ]
