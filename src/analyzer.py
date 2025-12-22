import json
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY,
    ANALYSIS_DIR,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    TARGET_CLIP_DURATION,
    DEFAULT_LLM_MODEL,
)
from transcriber import format_transcript_for_analysis, format_timestamp
import toon


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    id: str
    name: str
    context_length: int
    input_price: float  # $ per million tokens
    output_price: float  # $ per million tokens
    description: str = ""
    created: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def short_info(self) -> str:
        return f"{self.id} | {self.context_length//1000}K ctx | ${self.input_price}/M in, ${self.output_price}/M out"


class ClipAnalyzer:
    """
    Analizzatore di trascrizioni per identificare clip virali.
    
    Utilizza tecniche avanzate di identificazione basate su:
    - Hook nei primi 3 secondi (71% utenti decide qui)
    - Emotional triggers (3x probabilità viralità)
    - Pattern interrupts ogni 2-3 secondi
    - Completion rate optimization (target >80%)
    """
    
    DEFAULT_MODEL = DEFAULT_LLM_MODEL
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
    
    # Recommended models with fallback info
    RECOMMENDED_MODELS = [
        ModelInfo("deepseek/deepseek-chat", "DeepSeek Chat", 64000, 0.14, 0.28),
        ModelInfo("anthropic/claude-3-haiku", "Claude 3 Haiku", 200000, 0.25, 1.25),
        ModelInfo("openai/gpt-4o-mini", "GPT-4o Mini", 128000, 0.15, 0.60),
        ModelInfo("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet", 200000, 3.00, 15.00),
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        app_name: str = "Klipto",
        app_url: str = "https://github.com/klipto"
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self.app_name = app_name
        self.app_url = app_url
        self._models_cache: Optional[List[ModelInfo]] = None
        
        # Initialize client only if key is present
        if self.is_configured():
            self.client = OpenAI(
                base_url=self.OPENROUTER_API_URL,
                api_key=self.api_key,
            )
        else:
            self.client = None

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key.strip())
    
    def set_model(self, model_id: str):
        """Cambia il modello da usare."""
        self.model = model_id
        print(f"Modello impostato: {model_id}")
    
    def get_available_models(self, refresh: bool = False) -> List[ModelInfo]:
        """Ottiene lista modelli disponibili da OpenRouter."""
        if not self.is_configured():
            raise ValueError("OpenRouter API key non configurata")
            
        if self._models_cache and not refresh:
            return self._models_cache
        
        try:
            response = httpx.get(
                f"{self.OPENROUTER_API_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json().get("data", [])
            models = []
            
            for m in data:
                try:
                    # Skip non-chat models if ID is empty
                    if not m.get("id"):
                        continue
                        
                    pricing = m.get("pricing", {})
                    input_price = float(pricing.get("prompt", "0")) * 1_000_000
                    output_price = float(pricing.get("completion", "0")) * 1_000_000
                    
                    models.append(ModelInfo(
                        id=m.get("id", ""),
                        name=m.get("name", m.get("id", "")),
                        context_length=m.get("context_length", 0),
                        input_price=round(input_price, 4),
                        output_price=round(output_price, 4),
                        description=m.get("description", "")[:100],
                        created=m.get("created", "")
                    ))
                except:
                    continue
            
            # Sort by input price for usability
            models.sort(key=lambda x: x.input_price)
            self._models_cache = models
            return models
            
        except Exception as e:
            raise RuntimeError(f"Errore recupero modelli: {e}")
    
    def get_recommended_models(self) -> List[ModelInfo]:
        """Lista modelli consigliati (funziona anche senza API, fallback statico)."""
        try:
            if self.is_configured():
                # Se possibile, prendiamo i dati freschi filtrando quelli raccomandati
                all_models = self.get_available_models()
                rec_ids = [m.id for m in self.RECOMMENDED_MODELS]
                return [m for m in all_models if m.id in rec_ids]
        except:
            pass
        # Fallback statico
        return self.RECOMMENDED_MODELS
    
    def print_available_models(self, only_recommended: bool = True):
        """Stampa modelli disponibili con prezzi."""
        models = self.list_recommended_models() if only_recommended else self.get_available_models()
        
        print("\n" + "=" * 75)
        print("MODELLI DISPONIBILI" + (" (consigliati)" if only_recommended else ""))
        print("=" * 75)
        
        print(f"\n{'ID':<45} {'Context':<10} {'$/M Input':<12} {'$/M Output':<12}")
        print("-" * 75)
        
        for m in models:
            ctx = f"{m.context_length//1000}K"
            current = " ◄" if m.id == self.model else ""
            print(f"{m.id:<45} {ctx:<10} ${m.input_price:<11.4f} ${m.output_price:<11.4f}{current}")
        
        print("-" * 75)
        print(f"Modello attuale: {self.model}")
    
    def get_model_info(self, model_id: Optional[str] = None) -> Optional[ModelInfo]:
        """Ottiene info su un modello specifico."""
        model_id = model_id or self.model
        return next((m for m in self.get_available_models() if m.id == model_id), None)
    
    def estimate_cost(self, transcript: dict, max_clips: int = 5) -> dict:
        """Stima costo dell'analisi (usando TOON)."""
        model_info = self.get_model_info()
        if not model_info:
            return {"error": "Modello non trovato"}
        
        # Prepare content for TOON estimation
        utterances = transcript.get("utterances", [])
        simple_utts = [{"start": round(u["start"], 2), "end": round(u["end"], 2), "text": u["text"]} for u in utterances]
        toon_content = toon.encode(simple_utts, delimiter="\t")
        
        # Estimate based on TOON length (approx 3-4 chars per token)
        input_tokens = len(toon_content) // 3.5 + 1500  # +1500 system prompt
        output_tokens = max_clips * 100  # TOON output is verbose compact
        
        input_cost = (input_tokens / 1_000_000) * model_info.input_price
        output_cost = (output_tokens / 1_000_000) * model_info.output_price
        
        return {
            "model": model_info.id,
            "estimated_input_tokens": int(input_tokens),
            "estimated_output_tokens": int(output_tokens),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6),
            "context_length": model_info.context_length,
            "fits_context": input_tokens < model_info.context_length * 0.9
        }
    
    def check_credits(self) -> dict:
        """Controlla crediti OpenRouter rimanenti."""
        if not self.is_configured():
             return {"error": "API Key non configurata"}

        try:
            response = httpx.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                return {
                    "total_credits": data.get("total_credits", 0),
                    "used": data.get("total_usage", 0),
                    "remaining": data.get("total_credits", 0) - data.get("total_usage", 0),
                    "label": data.get("label", "")
                }
            return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_for_clips(
        self,
        transcript: dict,
        video_title: str,
        max_clips: int = 5,
        language: str = "it",
        target_platform: str = "all"
    ) -> dict:
        """
        Analizza trascrizione e identifica i migliori segmenti per clip virali.
        
        Args:
            transcript: Trascrizione da Deepgram
            video_title: Titolo del video originale
            max_clips: Numero massimo di clip da identificare
            language: Lingua per le risposte ("it" o "en")
            target_platform: Piattaforma target ("tiktok", "youtube", "instagram", "all")
        
        Returns:
            dict con clip identificate e metadata
        """
        if not self.is_configured():
            raise ValueError("API Key OpenRouter mancante. Configura il file .env")

        if not self.client:
             self.client = OpenAI(
                base_url=self.OPENROUTER_API_URL,
                api_key=self.api_key,
            )

        model_info = self.get_model_info()
        
        # Convert transcript to TOON for prompt
        utterances = transcript.get("utterances", [])
        # Simplify data for LLM efficiency
        simple_utts = [
            {
                "start": round(u["start"], 2), 
                "end": round(u["end"], 2), 
                "text": u["text"]
            } 
            for u in utterances
        ]
        toon_transcript = toon.encode(simple_utts, delimiter="\t")
        
        messages = self._build_messages(
            toon_transcript,
            video_title,
            transcript["duration"],
            max_clips,
            language,
            target_platform
        )
        
        print(f"\nAnalizzando trascrizione...")
        print(f"  Video: {video_title}")
        print(f"  Utterances: {len(transcript.get('utterances', []))}")
        print(f"  Durata: {format_timestamp(transcript['duration'])}")
        print(f"  Modello: {self.model}")
        print(f"  Piattaforma target: {target_platform}")
        
        if model_info:
            print(f"  Costo: ${model_info.input_price}/M input, ${model_info.output_price}/M output")
        
        estimate = self.estimate_cost(transcript, max_clips)
        print(f"  Stima costo: ~${estimate.get('total_cost', 0):.6f}")
        
        if not estimate.get("fits_context", True):
            print(f"  ⚠️  ATTENZIONE: Trascrizione potrebbe superare context window!")
        
        response = self._call_api(messages)
        clips = self._parse_response(response["content"])
        validated_clips = self._validate_clips(clips, transcript)
        
        return {
            "video_title": video_title,
            "video_duration": transcript["duration"],
            "target_platform": target_platform,
            "clips_found": len(validated_clips),
            "clips": validated_clips,
            "model_used": self.model,
            "language": language,
            "usage": response.get("usage", {}),
            "cost": response.get("cost", {})
        }
    
    def _build_messages(
        self,
        transcript: str,
        title: str,
        duration: float,
        max_clips: int,
        language: str,
        target_platform: str
    ) -> List[Dict[str, str]]:
        """Costruisce messaggi per l'API con prompt ottimizzato per viralità."""
        
        system_prompt = self._get_viral_system_prompt(language, target_platform)
        user_prompt = self._get_viral_user_prompt(
            transcript, title, duration, max_clips, language, target_platform
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _get_viral_system_prompt(self, language: str, target_platform: str) -> str:
        """
        Prompt di sistema ottimizzato per identificazione clip virali.
        Basato su ricerche e best practice del settore.
        """
        
        platform_specs = {
            "tiktok": """
🎵 SPECIFICHE TIKTOK:
- Engagement rate medio: 3.15% (il più alto tra le piattaforme)
- Completion rate target: >70%
- Prime 6 ore cruciali per l'algoritmo
- Focus su micro-storytelling e pattern sonori
- Trend audio aumentano reach 2-3x""",
            
            "youtube": """
📺 SPECIFICHE YOUTUBE SHORTS:
- Watch time ideale: >15 secondi
- Completion rate target: >80%
- Loop naturali aumentano watch time
- Keyword nel hook aumentano discoverability
- Thumbnail mentale nei primi 2 secondi""",
            
            "instagram": """
📸 SPECIFICHE INSTAGRAM REELS:
- Reach rate 2x superiore ai post standard
- Qualità visiva premium richiesta
- Caption con CTA entro 125 caratteri
- Trend estetici > trend audio
- Hashtag strategici (5-10 rilevanti)""",
            
            "all": """
🌐 OTTIMIZZAZIONE CROSS-PLATFORM:
- TikTok: Completion >70%, micro-storytelling
- YouTube: Watch time >15s, loop naturali
- Instagram: Qualità visiva, CTA chiari
- Contenuto deve funzionare su TUTTE le piattaforme"""
        }
        
        platform_info = platform_specs.get(target_platform, platform_specs["all"])
        
        if language == "it":
            return f"""Sei un esperto di contenuti virali con esperienza in YouTube Shorts, TikTok e Instagram Reels.

Il tuo compito è analizzare trascrizioni video e identificare i MOMENTI ESATTI con il più alto potenziale virale.

═══════════════════════════════════════════════════════════════════════════
📊 METRICHE DELLA VIRALITÀ
═══════════════════════════════════════════════════════════════════════════

STATISTICA CHIAVE: Il 71% degli utenti decide se continuare a guardare entro i PRIMI 3 SECONDI.
I contenuti con emotional trigger hanno 3x probabilità di diventare virali.

{platform_info}

═══════════════════════════════════════════════════════════════════════════
🎯 CRITERI DI SELEZIONE CLIP (in ordine di priorità)
═══════════════════════════════════════════════════════════════════════════

1. HOOK NEI PRIMI 3 SECONDI (CRITICO)
   ✅ Domande provocatorie ("Ma sai che...", "E se ti dicessi che...")
   ✅ Superlativi assoluti ("Il migliore", "Il più grande errore")
   ✅ Promesse di valore ("In 30 secondi capirai...")
   ✅ Conflitti/contrasti ("Quello che nessuno ti dice...")
   ✅ Pattern interrupt immediato (affermazione shock)
   ❌ MAI iniziare con "Allora...", "Quindi...", "E...", "Dunque..."

2. EMOTIONAL TRIGGERS (3x viralità)
   😲 Sorpresa/shock
   😂 Humor/ironia
   😤 Frustrazione condivisa ("Anche tu odi quando...")
   🤔 Curiosità intensa
   💪 Ispirazione/motivazione
   😱 Paura/urgenza (FOMO)

3. PATTERN INTERRUPTS
   - Cambio di ritmo ogni 2-3 secondi nel parlato
   - Picchi emotivi (voce alta, pause drammatiche)
   - Rivelazioni inaspettate
   - Domande retoriche che rompono il flusso

4. STRUTTURA NARRATIVA COMPLETA
   - Deve avere senso SENZA contesto esterno
   - Setup → Tensione → Payoff (o cliffhanger)
   - NO riferimenti a "come dicevo prima" o "vedremo dopo"

6. SEMANTIC COHERENCE (CONTEXT RULE)
   - 🚫 NEVER merge two different questions or topics into one clip.
   - 🚫 NEVER include outro/CTA ("Subscribe", "Like", "Follow me") unless it's part of the hook.
   - ✅ Clips must be self-contained within a SINGULAR scene or interaction.
   - 🔍 Check: Does the clip start in Interview A and end in Interview B? -> DISCARD or SPLIT.

═══════════════════════════════════════════════════════════════════════════
⏱️ TECHNICAL CUTTING RULES
═══════════════════════════════════════════════════════════════════════════

DURATION:
- Minimum: {MIN_CLIP_DURATION} seconds
- Maximum: {MAX_CLIP_DURATION} seconds
- Ideal: {TARGET_CLIP_DURATION} seconds (sweet spot for all platforms)

TIMESTAMPS:
- start_time = EXACT start of first word (in decimal seconds)
- end_time = EXACT end of last word (in decimal seconds)
- Use timestamps from transcript, DO NOT invent them
- I will add 0.3s buffer at start and 0.5s buffer at end

CLIP START:
- MUST start with COMPLETE sentence
- First word must be the HOOK
- MAI iniziare a metà frase

CLIP END:
- MUST end with COMPLETE sentence
- Conclusione soddisfacente O cliffhanger intenzionale
- MAI tagliare a metà concetto

═══════════════════════════════════════════════════════════════════════════
📋 OUTPUT FORMAT: TOON (Token-Oriented Object Notation)
═══════════════════════════════════════════════════════════════════════════

Respond ONLY with a TOON tabular array. Use tab indentation (2 spaces) and strict structure.
Columns: start, end, score (1-10), hook (text), reason (short text).

Example Output:
```toon
clips[2]{{start,end,score,hook,reason}}:
  120.5,150.2,9,"Did you know?", "Strong hook and payoff"
  200.0,230.5,8,"The secret is...", "High curiosity, emotional trigger"
```

Respond with NOTHING else but the code block."""

        else:  # English
            return f"""You are a viral content expert with experience in YouTube Shorts, TikTok and Instagram Reels.

Your task is to analyze video transcripts and identify the EXACT MOMENTS with the highest viral potential.

═══════════════════════════════════════════════════════════════════════════
📊 VIRALITY METRICS
═══════════════════════════════════════════════════════════════════════════

KEY STAT: 71% of users decide whether to keep watching within the FIRST 3 SECONDS.
Content with emotional triggers has 3x probability of going viral.

{platform_info}

═══════════════════════════════════════════════════════════════════════════
🎯 CLIP SELECTION CRITERIA (in priority order)
═══════════════════════════════════════════════════════════════════════════

1. HOOK IN FIRST 3 SECONDS (CRITICAL)
   ✅ Provocative questions ("Did you know that...", "What if I told you...")
   ✅ Absolute superlatives ("The best", "The biggest mistake")
   ✅ Value promises ("In 30 seconds you'll understand...")
   ✅ Conflicts/contrasts ("What nobody tells you...")
   ✅ Immediate pattern interrupt (shock statement)
   ❌ NEVER start with "So...", "And...", "Well...", "Basically..."

2. EMOTIONAL TRIGGERS (3x virality)
   😲 Surprise/shock
   😂 Humor/irony
   😤 Shared frustration ("You also hate when...")
   🤔 Intense curiosity
   💪 Inspiration/motivation
   😱 Fear/urgency (FOMO)

3. PATTERN INTERRUPTS
   - Rhythm change every 2-3 seconds in speech
   - Emotional peaks (loud voice, dramatic pauses)
   - Unexpected revelations
   - Rhetorical questions that break the flow

4. COMPLETE NARRATIVE STRUCTURE
   - Must make sense WITHOUT external context
   - Setup → Tension → Payoff (or cliffhanger)
   - NO references to "as I said before" or "we'll see later"

5. SHAREABILITY
   - Contains shareable insight?
   - Would someone think "I need to show this to..."?
   - Has a "quotable moment"?

6. SEMANTIC COHERENCE (CONTEXT RULE)
   - 🚫 NEVER merge two different questions or topics into one clip.
   - 🚫 NEVER include outro/CTA ("Subscribe", "Like", "Follow me") unless it's part of the hook.
   - ✅ Clips must be self-contained within a SINGULAR scene or interaction.
   - 🔍 Check: Does the clip start in Interview A and end in Interview B? -> DISCARD or SPLIT.

═══════════════════════════════════════════════════════════════════════════
⏱️ TECHNICAL CUTTING RULES
═══════════════════════════════════════════════════════════════════════════

DURATION:
- Minimum: {MIN_CLIP_DURATION} seconds
- Maximum: {MAX_CLIP_DURATION} seconds
- Ideal: {TARGET_CLIP_DURATION} seconds (sweet spot for all platforms)

TIMESTAMPS:
- start_time = EXACT start of first word (in decimal seconds)
- end_time = EXACT end of last word (in decimal seconds)
- Use timestamps from transcript, DO NOT invent them
- I will add 0.3s buffer at start and 0.5s buffer at end

CLIP START:
- MUST start with COMPLETE sentence
- First word must be the HOOK
- NEVER start mid-sentence

CLIP END:
- MUST end with COMPLETE sentence
- Satisfying conclusion OR intentional cliffhanger
- NEVER cut mid-concept

═══════════════════════════════════════════════════════════════════════════
📋 OUTPUT FORMAT: TOON (Token-Oriented Object Notation)
═══════════════════════════════════════════════════════════════════════════

Respond ONLY with a TOON tabular array. Use tab indentation (2 spaces) and strict structure.
Columns: start, end, score (1-10), hook (text), reason (short text).

Example Output:
```toon
clips[2]{{start,end,score,hook,reason}}:
  120.5,150.2,9,"Did you know?", "Strong hook and payoff"
  200.0,230.5,8,"The secret is...", "High curiosity, emotional trigger"
```

Respond with NOTHING else but the code block."""

    def _get_viral_user_prompt(
        self,
        transcript: str,
        title: str,
        duration: float,
        max_clips: int,
        language: str,
        target_platform: str
    ) -> str:
        """Prompt utente ottimizzato per identificazione viralità."""
        
        if language == "it":
            return f"""Analizza questa trascrizione e identifica le {max_clips} clip con il MASSIMO POTENZIALE VIRALE.

═══════════════════════════════════════════════════════════════════════════
📹 INFORMAZIONI VIDEO
═══════════════════════════════════════════════════════════════════════════

TITOLO: {title}
DURATA TOTALE: {format_timestamp(duration)} ({duration:.1f} secondi)
PIATTAFORMA TARGET: {target_platform.upper()}

═══════════════════════════════════════════════════════════════════════════
📝 TRASCRIZIONE (Formato TOON)
═══════════════════════════════════════════════════════════════════════════

{transcript}

═══════════════════════════════════════════════════════════════════════════
📋 ISTRUZIONI
═══════════════════════════════════════════════════════════════════════════

1. Leggi TUTTA la trascrizione
2. Identifica i {max_clips} momenti con il PIÙ ALTO potenziale virale
3. CRITICO: Ogni clip DEVE durare tra 15 e 90 secondi. IGNORA momenti più brevi.
4. Genera OUTPUT in formato TOON come richiesto nel system prompt.
5. Inserisci il test esatto dell'hook nella colonna 'hook'."""
        


        else:  # English
            return f"""Analyze this transcript and identify the {max_clips} clips with MAXIMUM VIRAL POTENTIAL.

═══════════════════════════════════════════════════════════════════════════
📹 VIDEO INFORMATION
═══════════════════════════════════════════════════════════════════════════

TITLE: {title}
TOTAL DURATION: {format_timestamp(duration)} ({duration:.1f} seconds)
TARGET PLATFORM: {target_platform.upper()}

═══════════════════════════════════════════════════════════════════════════
📝 TRANSCRIPT (TOON Format)
═══════════════════════════════════════════════════════════════════════════

{transcript}

═══════════════════════════════════════════════════════════════════════════
1. Read the ENTIRE transcript
2. Identify the {max_clips} moments with the HIGHEST viral potential
3. CRITICAL: Each clip MUST be between 15 and 90 seconds. IGNORE shorter moments.
4. Output strictly in TOON format as shown.
5. Provide the exact text of the hook in the 'hook' column."""
    
    def _call_api(self, messages: List[Dict[str, str]]) -> dict:
        """Chiama OpenRouter API."""
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.app_url,
                    "X-Title": self.app_name,
                },
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=8000
            )
            
            content = response.choices[0].message.content
            
            usage = {}
            cost = {}
            
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                model_info = self.get_model_info()
                if model_info:
                    input_cost = (response.usage.prompt_tokens / 1_000_000) * model_info.input_price
                    output_cost = (response.usage.completion_tokens / 1_000_000) * model_info.output_price
                    cost = {
                        "input": round(input_cost, 6),
                        "output": round(output_cost, 6),
                        "total": round(input_cost + output_cost, 6)
                    }
                
                print(f"\n  Token: {usage.get('prompt_tokens', 0)} in, {usage.get('completion_tokens', 0)} out")
                if cost:
                    print(f"  Costo: ${cost.get('total', 0):.6f}")
            
            return {"content": content, "usage": usage, "cost": cost}
            
        except Exception as e:
            err_str = str(e)
            if "No endpoints found matching your data policy" in err_str:
                 raise Exception(
                     f"\n❌ ERRORE POLICY OPENROUTER:\n"
                     f"   Il modello '{self.model}' non è disponibile con le tue attuali impostazioni privacy.\n"
                     f"   Potrebbe richiedere di abilitare 'Allow training' per i modelli gratuiti/speciali.\n"
                     f"   👉 Soluzione 1: Vai su https://openrouter.ai/settings/privacy\n"
                     f"      - Assicurati che 'ZDR Endpoints Only' sia DISATTIVATO (spesso è la causa).\n"
                     f"      - Assicurati che 'Enable paid/free endpoints that may train' sia ATTIVO.\n"
                     f"   👉 Soluzione 2: Scegli un altro modello."
                 ) from None
            raise Exception(f"Errore API OpenRouter: {e}")
    
    def _parse_response(self, response: str) -> list:
        """Parse risposta (TOON o JSON fallback)."""
        raw_clips = []
        
        # 1. Tentativo TOON
        try:
            raw_clips = toon.decode(response, strict=True)
            if not raw_clips:
                raise ValueError("Empty TOON")
        except Exception as e_toon:
            print(f"⚠️ TOON decoding failed: {e_toon}. Trying JSON fallback...")
            
            # 2. Tentativo JSON Fallback
            try:
                # Pulisci markdown code blocks se presenti
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                     clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                data = json.loads(clean_response)
                
                if isinstance(data, dict):
                    raw_clips = data.get("clips", [])
                elif isinstance(data, list):
                    raw_clips = data
                else:
                    raw_clips = []
                    
            except Exception as e_json:
                print(f"❌ Anche JSON decoding fallito: {e_json}")
                print(f"Response raw partial: {response[:200]}...")
                return []

        # Normalize keys
        normalized_clips = []
        for i, item in enumerate(raw_clips):
            # Campi comuni TOON/JSON
            start = item.get("start") or item.get("start_time") or 0
            end = item.get("end") or item.get("end_time") or 0
            score = item.get("score") or item.get("virality_score") or 5
            hook = item.get("hook") or item.get("hook_text") or ""
            reason = item.get("reason") or item.get("why_viral") or ""
            
            normalized_clips.append({
                "clip_number": i + 1,
                "start_time": float(start),
                "end_time": float(end),
                "virality_score": int(score),
                "hook_text": str(hook).strip().strip('"'),
                "why_viral": str(reason),
                # Defaults
                "hook_type": item.get("hook_type", "unknown"),
                "emotional_triggers": item.get("emotional_triggers", []),
                "primary_emotion": item.get("primary_emotion", "unknown"),
                "topic_summary": item.get("topic_summary", str(hook)[:50])
            })
        
        return normalized_clips
    
    def _validate_clips(self, clips: list, transcript: dict) -> list:
        """Valida e corregge timestamp delle clip."""
        validated = []
        words = transcript.get("words", [])
        utterances = transcript.get("utterances", [])
        
        if not utterances:
            return clips
        
        for clip in clips:
            try:
                validated_clip = self._validate_single_clip(clip, words, utterances)
                if validated_clip:
                    validated.append(validated_clip)
            except Exception as e:
                print(f"  ⚠️ Errore clip {clip.get('clip_number', '?')}: {e}")
        
        # Ordina per virality score
        validated.sort(key=lambda x: x.get("virality_score", 0), reverse=True)
        return validated
    
    def _validate_single_clip(
        self,
        clip: dict,
        words: list,
        utterances: list
    ) -> Optional[dict]:
        """Valida singola clip con i nuovi campi viralità."""
        
        start_time = float(clip.get("start_time", 0))
        end_time = float(clip.get("end_time", 0))
        
        if start_time >= end_time:
            return None
        
        # Trova utterances di confine
        start_utt = None
        for utt in utterances:
            if utt["start"] <= start_time <= utt["end"]:
                start_utt = utt
                break
            if utt["start"] > start_time:
                start_utt = utt
                break
        
        end_utt = None
        for utt in reversed(utterances):
            if utt["start"] <= end_time <= utt["end"]:
                end_utt = utt
                break
            if utt["end"] < end_time:
                end_utt = utt
                break
        
        if not start_utt or not end_utt:
            return None
        
        # Aggiusta timestamp
        adjusted_start = start_utt["start"]
        adjusted_end = end_utt["end"]
        duration = adjusted_end - adjusted_start
        
        # Verifica durata
        if duration < MIN_CLIP_DURATION or duration > MAX_CLIP_DURATION * 1.2:
            return None
        
        # Ottieni testo
        first_words = self._get_text_in_range(words, adjusted_start, adjusted_start + 4)
        last_words = self._get_text_in_range(words, adjusted_end - 4, adjusted_end)
        
        # Buffer
        final_start = max(0, adjusted_start - 0.3)
        final_end = adjusted_end + 0.5
        
        return {
            "clip_number": clip.get("clip_number", 0),
            "start_time": round(final_start, 3),
            "end_time": round(final_end, 3),
            "duration": round(final_end - final_start, 2),
            
            # Testo
            "first_words": first_words or clip.get("first_words", ""),
            "last_words": last_words or clip.get("last_words", ""),
            
            # Hook
            "hook_text": clip.get("hook_text", clip.get("hook", "")),
            "hook_type": clip.get("hook_type", ""),
            
            # Emozioni
            "emotional_triggers": clip.get("emotional_triggers", []),
            "primary_emotion": clip.get("primary_emotion", clip.get("emotion", "")),
            
            # Contenuto
            "topic_summary": clip.get("topic_summary", clip.get("topic", "")),
            "quotable_moment": clip.get("quotable_moment", ""),
            
            # Score viralità
            "virality_score": clip.get("virality_score", 5),
            "virality_breakdown": clip.get("virality_breakdown", {}),
            
            # Platform fit
            "platform_fit": clip.get("platform_fit", {}),
            
            # Motivazione
            "why_viral": clip.get("why_viral", clip.get("reason", ""))
        }
    
    def _get_text_in_range(self, words: list, start: float, end: float) -> str:
        """Ottiene parole in un range temporale."""
        if not words:
            return ""
        return " ".join(
            w.get("punctuated_word", w.get("word", ""))
            for w in words
            if start <= w.get("start", 0) <= end
        )
    
    def save_analysis(self, analysis: dict, video_id: str) -> str:
        """Salva analisi su file."""
        output_path = ANALYSIS_DIR / f"{video_id}_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Analisi salvata: {output_path}")
        return str(output_path)
    
    def load_analysis(self, video_id: str) -> Optional[dict]:
        """Carica analisi da file."""
        analysis_path = ANALYSIS_DIR / f"{video_id}_analysis.json"
        if not analysis_path.exists():
            matches = list(ANALYSIS_DIR.glob(f"*{video_id}*_analysis.json"))
            if matches:
                analysis_path = matches[0]
            else:
                return None
        with open(analysis_path, "r", encoding="utf-8") as f:
            return json.load(f)


def print_analysis_summary(analysis: dict):
    """Stampa riepilogo analisi con dettagli viralità."""
    
    print("\n" + "═" * 70)
    print("📊 ANALISI VIRALITÀ COMPLETATA")
    print("═" * 70)
    print(f"Video: {analysis['video_title']}")
    print(f"Durata: {format_timestamp(analysis['video_duration'])}")
    print(f"Piattaforma: {analysis.get('target_platform', 'all').upper()}")
    print(f"Modello: {analysis.get('model_used', 'N/A')}")
    
    if analysis.get('cost'):
        print(f"Costo analisi: ${analysis['cost'].get('total', 0):.6f}")
    
    print(f"\n🎬 Clip identificate: {analysis['clips_found']}")
    print("─" * 70)
    
    for clip in analysis["clips"]:
        score = clip.get('virality_score', 0)
        score_bar = "█" * score + "░" * (10 - score)
        
        print(f"\n📍 CLIP {clip['clip_number']} | Virality: [{score_bar}] {score}/10")
        print(f"   ⏱️  {clip['start_time']:.2f}s → {clip['end_time']:.2f}s ({clip['duration']:.1f}s)")
        
        # Hook
        hook = clip.get('hook_text', '')[:60]
        hook_type = clip.get('hook_type', '')
        print(f"   🎣 Hook ({hook_type}): \"{hook}{'...' if len(clip.get('hook_text', '')) > 60 else ''}\"")
        
        # Emozioni
        emotions = clip.get('emotional_triggers', [])
        primary = clip.get('primary_emotion', '')
        if emotions:
            print(f"   💡 Emozioni: {primary} | {', '.join(emotions)}")
        
        # Topic
        print(f"   📌 Topic: {clip.get('topic_summary', '')}")
        
        # Breakdown score
        breakdown = clip.get('virality_breakdown', {})
        if breakdown:
            print(f"   📈 Breakdown: Hook {breakdown.get('hook_strength', '-')}/10 | "
                  f"Emotion {breakdown.get('emotional_impact', '-')}/10 | "
                  f"Completion {breakdown.get('completion_likelihood', '-')}/10 | "
                  f"Share {breakdown.get('shareability', '-')}/10")
        
        # Platform fit
        platform_fit = clip.get('platform_fit', {})
        if platform_fit:
            print(f"   🎯 Platform: TikTok {platform_fit.get('tiktok', '-')}/10 | "
                  f"YT {platform_fit.get('youtube_shorts', '-')}/10 | "
                  f"IG {platform_fit.get('instagram_reels', '-')}/10")
        
        # Perché virale
        why = clip.get('why_viral', '')
        if why:
            print(f"   ✅ {why[:80]}{'...' if len(why) > 80 else ''}")
        
        # Quotable
        quotable = clip.get('quotable_moment', '')
        if quotable:
            print(f"   💬 Quote: \"{quotable[:50]}{'...' if len(quotable) > 50 else ''}\"")
    
    print("\n" + "═" * 70)
