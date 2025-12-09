"""
NLP Analyzer Module
Analyzes video transcripts to identify viral hooks using LLMs.
Supports multiple providers: OpenRouter, OpenAI, Anthropic, Google.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings


# Latest recommended models per provider (as of December 2024)
DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-chat",  # DeepSeek V3 - fast & capable
    "openai": "gpt-4o",  # GPT-4o - latest multimodal
    "anthropic": "anthropic/claude-3.5-sonnet",  # Via OpenRouter
    "google": "google/gemini-2.0-flash-exp",  # Via OpenRouter
}

# Alternative models for different use cases
ALTERNATIVE_MODELS = {
    "openrouter": [
        "deepseek/deepseek-chat",           # DeepSeek V3 - best value
        "deepseek/deepseek-reasoner",       # DeepSeek R1 - reasoning
        "anthropic/claude-3.5-sonnet",      # Claude 3.5 Sonnet
        "openai/gpt-4o",                    # GPT-4o
        "google/gemini-2.0-flash-exp",      # Gemini 2.0 Flash
        "meta-llama/llama-3.3-70b-instruct", # Llama 3.3 70B
    ],
    "openai": [
        "gpt-4o",                           # Latest GPT-4o
        "gpt-4o-mini",                      # Faster, cheaper
        "gpt-4-turbo",                      # Previous gen
        "o1-preview",                       # Reasoning model
    ],
}


class NLPAnalyzer:
    """
    Analyzes transcript to identify interesting clips using LLM.

    Supports:
    - OpenRouter (access to DeepSeek, Claude, GPT, Gemini, Llama, etc.)
    - OpenAI Direct
    - Anthropic via OpenRouter
    - Google Gemini via OpenRouter
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self._setup_client()

    def _setup_client(self):
        """Setup the OpenAI client based on provider."""

        if self.provider == "openrouter":
            self.api_key = settings.OPENROUTER_API_KEY
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = settings.LLM_MODEL or DEFAULT_MODELS["openrouter"]
            self.extra_headers = {
                "HTTP-Referer": "https://klipto.ai",
                "X-Title": "Klipto"
            }
        elif self.provider == "openai":
            self.api_key = settings.OPENAI_API_KEY
            self.base_url = None  # Use default OpenAI endpoint
            self.model = settings.LLM_MODEL or DEFAULT_MODELS["openai"]
            self.extra_headers = {}
        elif self.provider == "anthropic":
            # Use OpenRouter for Anthropic models
            self.api_key = settings.OPENROUTER_API_KEY or settings.ANTHROPIC_API_KEY
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = settings.LLM_MODEL or DEFAULT_MODELS["anthropic"]
            self.extra_headers = {
                "HTTP-Referer": "https://klipto.ai",
                "X-Title": "Klipto"
            }
        elif self.provider == "google":
            # Use OpenRouter for Google models
            self.api_key = settings.OPENROUTER_API_KEY or settings.GOOGLE_API_KEY
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = settings.LLM_MODEL or DEFAULT_MODELS["google"]
            self.extra_headers = {
                "HTTP-Referer": "https://klipto.ai",
                "X-Title": "Klipto"
            }
        else:
            # Default to OpenRouter
            self.api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
            self.base_url = "https://openrouter.ai/api/v1" if settings.OPENROUTER_API_KEY else None
            self.model = settings.LLM_MODEL or DEFAULT_MODELS.get(self.provider, "gpt-4o")
            self.extra_headers = {}

        # Initialize client
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
            print("⚠️ Warning: No API key configured. NLP analysis will use mock data.")

    def identify_hooks(
        self,
        transcript: Dict[str, Any],
        num_hooks: int = 3,
        min_duration: float = 15.0,
        max_duration: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Identifies viral hooks from transcript using LLM.

        Args:
            transcript: Dict containing 'text' and 'segments'
            num_hooks: Number of hooks to identify
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds

        Returns:
            List of hooks with start, end, reason, and score
        """

        if not self.client:
            print("⚠️ No API Key configured. Returning mock hooks.")
            return self._mock_hooks(num_hooks)

        full_text = transcript.get('text', '')
        segments = transcript.get('segments', [])

        # Create a timestamped version for better accuracy
        timestamped_text = self._create_timestamped_transcript(segments)

        # Truncate if too long (keep ~6000 chars for context window safety)
        if len(timestamped_text) > 6000:
            timestamped_text = timestamped_text[:6000] + "\n[...transcript truncated...]"

        prompt = self._build_prompt(
            timestamped_text,
            num_hooks,
            min_duration,
            max_duration
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
                extra_headers=self.extra_headers if self.extra_headers else None
            )

            content = response.choices[0].message.content
            hooks = self._parse_response(content)

            # Validate and sanitize hooks
            hooks = self._validate_hooks(hooks, min_duration, max_duration)

            print(f"✅ Successfully identified {len(hooks)} hooks using {self.model}")
            return hooks

        except Exception as e:
            print(f"❌ NLP Analysis failed: {str(e)}")
            print(f"   Provider: {self.provider}, Model: {self.model}")
            return self._mock_hooks(num_hooks)

    def _create_timestamped_transcript(self, segments: List[Dict]) -> str:
        """Creates a timestamped version of the transcript."""
        lines = []
        for seg in segments:
            start = seg.get('start', 0)
            text = seg.get('text', '').strip()
            if text:
                timestamp = f"[{self._format_time(start)}]"
                lines.append(f"{timestamp} {text}")
        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        """Formats seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the LLM."""
        return """You are an expert video editor and social media content strategist specializing in creating viral short-form content for TikTok, Instagram Reels, and YouTube Shorts.

Your task is to analyze video transcripts and identify the most engaging moments that would make compelling standalone clips.

Focus on:
- Strong opening hooks that grab attention in the first 3 seconds
- Emotional peaks (humor, surprise, insight, controversy)
- Complete thoughts or stories that work standalone
- Quotable or shareable moments
- Educational "aha" moments

Output ONLY valid JSON. No explanations, no markdown formatting."""

    def _build_prompt(
        self,
        transcript: str,
        num_hooks: int,
        min_duration: float,
        max_duration: float
    ) -> str:
        """Builds the analysis prompt."""
        return f"""Analyze this timestamped video transcript and identify the {num_hooks} best potential viral clips.

REQUIREMENTS:
- Each clip should be {min_duration}-{max_duration} seconds long
- Start times should align with natural sentence beginnings
- End times should complete the thought/story
- Score based on viral potential (0-100)

TRANSCRIPT:
{transcript}

OUTPUT FORMAT - Return ONLY a JSON array:
[
  {{"start": 10.5, "end": 35.0, "reason": "Strong hook with surprising revelation", "score": 92}},
  {{"start": 45.0, "end": 70.0, "reason": "Funny anecdote with punchline", "score": 88}}
]

Return exactly {num_hooks} clips in the JSON array."""

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        """Parses the LLM response to extract hooks."""

        # Clean content (remove markdown code blocks)
        content = content.strip()
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse LLM response as JSON")
            return []

    def _validate_hooks(
        self,
        hooks: List[Dict],
        min_duration: float,
        max_duration: float
    ) -> List[Dict[str, Any]]:
        """Validates and sanitizes hooks."""
        valid_hooks = []

        for hook in hooks:
            try:
                start = float(hook.get('start', 0))
                end = float(hook.get('end', start + min_duration))
                duration = end - start

                # Ensure valid duration
                if duration < min_duration:
                    end = start + min_duration
                elif duration > max_duration:
                    end = start + max_duration

                # Ensure start is not negative
                if start < 0:
                    start = 0
                    end = min_duration

                valid_hooks.append({
                    'start': round(start, 2),
                    'end': round(end, 2),
                    'reason': str(hook.get('reason', 'Interesting moment')),
                    'score': min(100, max(0, int(hook.get('score', 75))))
                })
            except (ValueError, TypeError) as e:
                print(f"⚠️ Skipping invalid hook: {hook} - {e}")
                continue

        # Sort by score descending
        valid_hooks.sort(key=lambda x: x['score'], reverse=True)

        return valid_hooks

    def _mock_hooks(self, num_hooks: int = 3) -> List[Dict[str, Any]]:
        """Returns mock hooks for testing when no API key is available."""
        mock_data = [
            {"start": 0.0, "end": 20.0, "reason": "Opening hook - Introduction (Mock)", "score": 85},
            {"start": 25.0, "end": 50.0, "reason": "Key insight or main point (Mock)", "score": 90},
            {"start": 55.0, "end": 80.0, "reason": "Interesting story or example (Mock)", "score": 82},
            {"start": 85.0, "end": 110.0, "reason": "Actionable advice (Mock)", "score": 78},
            {"start": 115.0, "end": 140.0, "reason": "Surprising revelation (Mock)", "score": 75},
        ]
        return mock_data[:num_hooks]

    def get_available_models(self) -> List[str]:
        """Returns list of available models for the current provider."""
        return ALTERNATIVE_MODELS.get(self.provider, ALTERNATIVE_MODELS["openrouter"])
