import json
import os
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from config import DEEPGRAM_API_KEY

class DeepgramModelCache:
    """
    Scarica e cachea automaticamente i modelli Deepgram con lingue supportate.
    Aggiorna il cache ogni 24 ore o quando richiesto.
    Uses httpx for async/sync compatibility with the rest of the project.
    """
    
    CACHE_FILE = Path("deepgram_models_cache.json")
    CACHE_DURATION_HOURS = 24
    API_BASE = "https://api.deepgram.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY non configurata")
            
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        self.models_data = None
        self.last_updated = None
        
    def _is_cache_valid(self) -> bool:
        """Controlla se il cache è ancora valido."""
        if not self.CACHE_FILE.exists():
            return False
        
        try:
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                last_updated = datetime.fromisoformat(cache.get("timestamp"))
                is_valid = datetime.now() - last_updated < timedelta(hours=self.CACHE_DURATION_HOURS)
                return is_valid
        except:
            return False
    
    def _load_from_cache(self) -> Optional[Dict]:
        """Carica i modelli dal cache locale."""
        try:
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                self.models_data = cache.get("models")
                self.last_updated = cache.get("timestamp")
                # print(f"✓ Modelli caricati da cache (aggiornato: {self.last_updated})")
                return {"models": self.models_data, "timestamp": self.last_updated}
        except Exception as e:
            print(f"✗ Errore caricamento cache: {e}")
            return None
    
    def _fetch_from_api(self) -> Optional[Dict]:
        """Scarica i modelli direttamente dall'API Deepgram."""
        try:
            print("📡 Scaricamento modelli da API Deepgram...")
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.API_BASE}/models",
                    headers=self.headers
                )
                response.raise_for_status()
                
                data = response.json()
                
                models_list = []
                if "stt" in data:
                    # API returns { "stt": [model1, model2, ...], ... }
                    models_list = data["stt"]
                elif "models" in data:
                    models_list = data["models"]
                else:
                    print(f"⚠ Chiave 'stt' o 'models' non trovata nella risposta API. Keys: {list(data.keys())}")
            
            # Print success only if we found something
            print(f"✓ Ricevuti {len(models_list)} modelli da API")
            
            # Salva in cache
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "models": models_list
            }
            
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.models_data = models_list
            self.last_updated = cache_data["timestamp"]
            # print(f"✓ Cache aggiornato: {self.CACHE_FILE}")
            return models_list
            
        except httpx.HTTPError as e:
            print(f"✗ Errore API: {e}")
            return None
    
    def get_models(self, force_refresh: bool = False) -> Dict:
        """
        Ottiene la lista di modelli (da cache o API).
        """
        if not force_refresh and self._is_cache_valid():
            loaded = self._load_from_cache()
            if loaded:
                return loaded
        
        models = self._fetch_from_api()
        
        if models is None:
            print("⚠ Fallback a cache (anche se scaduto)...")
            cached = self._load_from_cache()
            if cached:
                return cached
            else:
                # print("✗ Nessun cache disponibile e API irraggiungibile")
                return {"models": []}
        
        return {"models": models, "timestamp": self.last_updated}
    
    def get_model_variants(self, family_prefix: str) -> List[str]:
        """Ritorna tutte le varianti di una famiglia di modelli."""
        if not self.models_data:
            self.get_models()
        
        return [m["name"] for m in self.models_data if m["name"].startswith(family_prefix)]
    
    def get_supported_languages(self, model: str) -> List[str]:
        """Ritorna le lingue supportate per un modello specifico."""
        if not self.models_data:
            self.get_models()
        
        for m in self.models_data:
            if m["name"] == model:
                langs = m.get("languages") or m.get("language") or m.get("supported_languages", [])
                if isinstance(langs, str):
                    return [langs] if langs != "multi" else ["multi"]
                return langs if isinstance(langs, list) else []
        
        return []
    
    def is_combination_supported(self, model: str, language: str) -> bool:
        """Verifica se una combinazione model+language è supportata."""
        # Se il modello è quello di default o generico che sappiamo funzionare sempre, potremmo skippare,
        # ma meglio controllare.
        
        if not self.models_data:
            self.get_models()
            
        supported_langs = self.get_supported_languages(model)
        
        # Se non troviamo il modello nella lista, assumiamo True (magari è nuovo e cache vecchio)?
        # O False? Meglio False se vogliamo essere strict, o True per non bloccare custom models.
        # Se la lista è vuota, significa che il modello non è stato trovato nel JSON.
        if not supported_langs:
            # Check if it's a custom model not in public list?
            # Let's verify if the model exists at least.
            # For now, return True to be safe if model not found in list (fallback).
            return True 
            
        if "multi" in supported_langs:
            return True
            
        # Normalizza
        language = language.lower().split('-')[0] # 'it-IT' -> 'it'
        
        # Controlla
        for lang in supported_langs:
            if lang.lower().startswith(language):
                return True
                
        return False

    def print_summary(self):
        """Stampa un riassunto dei modelli disponibili."""
        if not self.models_data:
            self.get_models()
        
        print("\n" + "="*60)
        print("DEEPGRAM MODELS & LANGUAGES (Dynamic)")
        print("="*60)
        
        families = {}
        for model in self.models_data:
            name = model["name"]
            family = name.split("-")[0] if "-" in name else name
            if family not in families:
                families[family] = []
            families[family].append(name)
        
        for family, variants in sorted(families.items()):
            print(f"\n{family.upper()}:")
            for variant in sorted(variants):
                langs = self.get_supported_languages(variant)
                if "multi" in langs:
                    img = "🌍"
                    langs_str = "Multilingue"
                else:
                    img = "🇬🇧" if "en" in langs and len(langs)==1 else "🏳️"
                    langs_str = f"{len(langs)} lingue ({', '.join(langs[:3])}...)"
                
                print(f"  • {variant:30} {img} {langs_str}")
        
        print("\n" + "="*60 + "\n")
